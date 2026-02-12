import pytorch_lightning as pl
import torch
import os
import yaml
import copy
import numpy as np
import pandas as pd
from .config import SciLaMAConfig
from .data import SciLaMADataModule
from .model_lit import SciLaMALightningModule
from .callbacks import (
    first_epoch_beta_reached,
    DelayedModelCheckpoint,
    DelayedEarlyStopping,
    PrintModelArchitecture,
)


class SciLaMATrainer:
    def __init__(self, config_path: str = "script/template.yaml"):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # Parse with Pydantic
        self.config = SciLaMAConfig(**config_dict)
        self.datamodule = SciLaMADataModule(self.config.data, self.config.training)
        self.module = None
        self.trainer = None

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a trained model from a checkpoint file. Restores covariate encoding state (one-hot mapping, continuous mean/std)
        from the checkpoint for consistency with training.
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.module = SciLaMALightningModule.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
            weights_only=False,
        )
        self.datamodule.set_covariate_encoding_state(self.module.covariate_encoding_state)
        self.datamodule.setup()
        print("Model loaded successfully.")

    def train(self):
        pl.seed_everything(self.config.training.seed)
        # Setup data
        self.datamodule.setup()
        
        # Determine mode
        mode = self.config.training.mode
        
        if mode == "stepwise":
            self._train_stepwise()
        else:
            self._train_standard(mode)
            
    def _save_sample_embeddings(self, module, out_path: str):
        """Save sample (cell) embeddings from module to adata.obsm and parquet."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = module.to(device)
        module.eval()
        pred_loader = self.datamodule.predict_dataloader()
        embeddings = []
        with torch.no_grad():
            for batch in pred_loader:
                x, c = batch
                x, c = x.to(device), c.to(device)
                mu, _, _ = module.sample_encoder(x, c)
                embeddings.append(mu.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        key = self.config.output.save_key
        self.datamodule.adata.obsm[key] = embeddings
        os.makedirs(self.config.output.save_dir, exist_ok=True)
        sample_ids = self.datamodule.adata.obs.index.astype(str).tolist()
        emb_list = [embeddings[i].tolist() for i in range(len(sample_ids))]
        df = pd.DataFrame({"sample_id": sample_ids, "embedding": emb_list})
        df.to_parquet(out_path)
        print(f"Stored sample embeddings in adata.obsm['{key}'] and {out_path}")

    def _save_feature_embeddings(self, module, out_path: str):
        """Save gene embeddings (feature VAE latent) from module to parquet."""
        if getattr(module, "feature_encoder", None) is None:
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        module = module.to(device)
        module.eval()
        static_mask = self.datamodule.adata.var["static_embedding"].values
        with torch.no_grad():
            if module.feature_input_embeddings and len(module.feature_input_embeddings) > 0:
                inputs = [emb.to(device) for emb in module.feature_input_embeddings.values()]
                mu_f, _, _ = module.feature_encoder(inputs, None)
                feature_emb = mu_f.cpu().numpy()
            else:
                X = self.datamodule.adata[:, static_mask].X
                Xt = torch.FloatTensor(X.toarray() if hasattr(X, "toarray") else X).t().to(device)
                mu_f, _, _ = module.feature_encoder(Xt, None)
                feature_emb = mu_f.cpu().numpy()
        gene_ids = self.datamodule.adata.var_names[static_mask].astype(str).tolist()
        embeddings = [feature_emb[i].tolist() for i in range(len(gene_ids))]
        feature_df = pd.DataFrame({"gene_id": gene_ids, "embedding": embeddings})
        os.makedirs(self.config.output.save_dir, exist_ok=True)
        feature_df.to_parquet(out_path)
        print(f"Stored feature embeddings in {out_path}")

    def _save_outputs_after_training(self, module, sample_suffix: str | None = None, feature_suffix: str | None = None):
        """Save outputs after training. Suffixes follow: beta_vae, direct_sciLaMA, intermediate, stepwise_sciLaMA."""
        os.makedirs(self.config.output.save_dir, exist_ok=True)
        if sample_suffix:
            out_path = os.path.join(self.config.output.save_dir, f"sample_embeddings_{sample_suffix}.parquet")
            self._save_sample_embeddings(module, out_path)
        if feature_suffix and getattr(module, "feature_encoder", None) is not None:
            out_path = os.path.join(self.config.output.save_dir, f"feature_embeddings_{feature_suffix}.parquet")
            self._save_feature_embeddings(module, out_path)

    def _create_module(self, phase_config: SciLaMAConfig, feature_only_phase: bool = False):
        """Create SciLaMALightningModule with standard args from datamodule."""
        return SciLaMALightningModule(
            config=phase_config,
            n_features=self.datamodule.n_features,
            n_covariates=self.datamodule.n_covariates,
            total_samples=self.datamodule.adata.n_obs,
            feature_input_embeddings=self.datamodule.feature_input_embeddings,
            covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
            feature_only_phase=feature_only_phase,
        )

    def _load_best_module(self, checkpoint_path: str, phase_config: SciLaMAConfig, feature_only_phase: bool = False):
        """Load best checkpoint and return module."""
        return SciLaMALightningModule.load_from_checkpoint(
            checkpoint_path,
            config=phase_config,
            n_features=self.datamodule.n_features,
            n_covariates=self.datamodule.n_covariates,
            total_samples=self.datamodule.adata.n_obs,
            feature_input_embeddings=self.datamodule.feature_input_embeddings,
            covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
            feature_only_phase=feature_only_phase,
            weights_only=False,
        )

    def _load_best_if_saved(self, module, best_path: str, phase_config: SciLaMAConfig, feature_only_phase: bool = False):
        """Return module loaded from best_path if it exists, else return module and print fallback message."""
        if best_path and os.path.isfile(best_path):
            return self._load_best_module(best_path, phase_config, feature_only_phase)
        print("No checkpoint saved (training ended before beta=1 or no improvement). Using in-memory model.")
        return module

    def _setup_Xt_if_needed(self, module, mode: str):
        """Setup Xt buffer when feature VAE needs transposed data (no external embeddings)."""
        if self.datamodule.feature_input_embeddings or mode == "beta_vae":
            return
        static_mask = self.datamodule.adata.var["static_embedding"].values
        X = self.datamodule.adata[:, static_mask].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        module.setup_Xt(torch.FloatTensor(X).t())

    def _create_trainer(self, callbacks: list):
        """Create pl.Trainer with standard settings."""
        return pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices=self.config.training.devices,
            strategy=self.config.training.strategy,
            val_check_interval=self.config.training.val_check_interval,
            logger=False,
        )

    def _get_callbacks(self, mode: str):
        os.makedirs(self.config.output.save_dir, exist_ok=True)
        start_epoch = first_epoch_beta_reached(self.config)
        if start_epoch > 0:
            print(f"Checkpoint saving and early-stopping patience start from epoch {start_epoch} (after beta reaches {self.config.training.beta_end}).")
        checkpoint_callback = DelayedModelCheckpoint(
            start_epoch=start_epoch,
            save_key=self.config.output.save_key,
            phase=mode,
            save_top_k=self.config.output.save_top_k,
            monitor="val_loss",
            mode="min",
            dirpath=self.config.output.save_dir,
        )
        early_stop_callback = DelayedEarlyStopping(
            start_epoch=start_epoch,
            monitor="val_loss",
            mode="min",
            patience=self.config.training.patience,
        )
        print_arch_callback = PrintModelArchitecture()
        return [checkpoint_callback, early_stop_callback, print_arch_callback]

    def _train_standard(self, mode: str):
        print(f"Initializing model in mode: {mode}")
        self.module = self._create_module(self.config)
        self._setup_Xt_if_needed(self.module, mode)

        callbacks = self._get_callbacks(mode)
        self.trainer = self._create_trainer(callbacks)

        print("Starting training...")
        self.trainer.fit(self.module, self.datamodule)

        best_path = callbacks[0].best_model_path
        if best_path:
            print(f"Best model: {best_path}")
        self.module = self._load_best_if_saved(self.module, best_path or "", self.config)
        print("\nSaving outputs...")
        if mode == "beta_vae":
            self._save_outputs_after_training(self.module, sample_suffix="beta_vae")
        else:
            self._save_outputs_after_training(self.module, sample_suffix="direct_sciLaMA", feature_suffix="direct_sciLaMA")

    def _train_stepwise(self):
        """
        Stepwise training: 3 phases (sample → feature only → joint)
        1. Train sample VAE only (beta_vae)
        2. Fix sample VAE, train feature VAE only (gamma=0)
        3. Jointly train sciLaMA (both VAEs, gamma=0.05)
        """
        print("Starting Stepwise Training (3 phases)...")
        
        # Phase 1: Sample VAE only
        print("\n=== Phase 1: Training Sample VAE ===")
        phase1_config = copy.deepcopy(self.config)
        phase1_config.training.mode = "beta_vae"

        module_phase1 = self._create_module(phase1_config)
        callbacks_p1 = self._get_callbacks("stepwise_phase1")
        trainer_p1 = self._create_trainer(callbacks_p1)
        trainer_p1.fit(module_phase1, self.datamodule)

        best_p1_path = callbacks_p1[0].best_model_path
        print(f"Phase 1 complete. Best model: {best_p1_path}")
        module_phase1 = self._load_best_if_saved(module_phase1, best_p1_path, phase1_config)
        
        # Phase 2: Feature VAE only (sample frozen); uses feature_vae_loss with gamma=0
        print("\n=== Phase 2: Training Feature VAE (Sample Frozen, feature_vae_loss, gamma=0) ===")
        phase2_config = copy.deepcopy(self.config)
        phase2_config.training.mode = "direct"
        phase2_config.training.gamma = 0.0

        module_phase2 = self._create_module(phase2_config, feature_only_phase=True)
        self._setup_Xt_if_needed(module_phase2, "direct")

        print("Transferring Sample VAE weights (frozen)...")
        module_phase2.sample_encoder.load_state_dict(module_phase1.sample_encoder.state_dict())
        module_phase2.sample_decoder.load_state_dict(module_phase1.sample_decoder.state_dict())
        module_phase2.freeze_beta_vae()

        callbacks_p2 = self._get_callbacks("stepwise_phase2")
        trainer_p2 = self._create_trainer(callbacks_p2)
        trainer_p2.fit(module_phase2, self.datamodule)

        best_p2_path = callbacks_p2[0].best_model_path
        print(f"Phase 2 complete. Best model: {best_p2_path}")
        module_phase2 = self._load_best_if_saved(module_phase2, best_p2_path, phase2_config, feature_only_phase=True)
        
        self._save_outputs_after_training(module_phase1, sample_suffix="beta_vae")
        self._save_outputs_after_training(module_phase2, feature_suffix="intermediate")
        
        # Phase 3: Joint training (both VAEs, gamma=0.05)
        print("\n=== Phase 3: Joint Training sciLaMA (gamma=0.05) ===")
        phase3_config = copy.deepcopy(self.config)
        phase3_config.training.mode = "direct"
        phase3_config.training.gamma = self.config.training.gamma  # joint: default 0.05

        module_phase3 = self._create_module(phase3_config)
        self._setup_Xt_if_needed(module_phase3, "direct")

        print("Transferring both Sample VAE and Feature VAE weights (all trainable)...")
        module_phase3.sample_encoder.load_state_dict(module_phase2.sample_encoder.state_dict())
        module_phase3.sample_decoder.load_state_dict(module_phase2.sample_decoder.state_dict())
        module_phase3.feature_encoder.load_state_dict(module_phase2.feature_encoder.state_dict())
        module_phase3.feature_decoder.load_state_dict(module_phase2.feature_decoder.state_dict())

        callbacks_p3 = self._get_callbacks("stepwise_phase3")
        trainer_p3 = self._create_trainer(callbacks_p3)
        trainer_p3.fit(module_phase3, self.datamodule)

        best_p3_path = callbacks_p3[0].best_model_path
        print(f"Phase 3 complete. Best model: {best_p3_path}")
        self.module = self._load_best_if_saved(module_phase3, best_p3_path, phase3_config)
        print("\nSaving final outputs...")
        self._save_outputs_after_training(self.module, sample_suffix="stepwise_sciLaMA", feature_suffix="stepwise_sciLaMA")
