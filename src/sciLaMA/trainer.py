import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import os
import yaml
import copy
import numpy as np
from typing import Optional
from .config import SciLaMAConfig
from .data import SciLaMADataModule
from .model_lit import SciLaMALightningModule

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
        from the checkpoint so predict() uses the same encoding as training.
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.module = SciLaMALightningModule.load_from_checkpoint(
            checkpoint_path,
            config=self.config,
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
            
    def _get_callbacks(self, mode: str):
        os.makedirs(self.config.output.save_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.output.save_dir,
            filename=f"{self.config.output.save_key}_{mode}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.config.training.patience,
            mode="min"
        )
        return [checkpoint_callback, early_stop_callback]

    def _train_standard(self, mode: str):
        # Initialize module
        print(f"Initializing model in mode: {mode}")
        self.module = SciLaMALightningModule(
            config=self.config,
            n_features=self.datamodule.n_features,
            n_covariates=self.datamodule.n_covariates,
            total_samples=self.datamodule.adata.n_obs,
            feature_input_embeddings=self.datamodule.feature_input_embeddings,
            covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
        )
        
        # Setup Xt if needed (only static_embedding genes for feature VAE)
        if not self.datamodule.feature_input_embeddings and mode != "beta_vae":
             static_mask = self.datamodule.adata.var["static_embedding"].values
             X = self.datamodule.adata[:, static_mask].X
             if hasattr(X, "toarray"):
                 X = X.toarray()
             Xt = torch.FloatTensor(X).t()
             self.module.setup_Xt(Xt)

        callbacks = self._get_callbacks(mode)
        
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            callbacks=callbacks,
            accelerator="auto",
            devices=self.config.training.devices,
            strategy=self.config.training.strategy,
            log_every_n_steps=10
        )
        
        print("Starting training...")
        self.trainer.fit(self.module, self.datamodule)
        
        # Load best
        best_path = callbacks[0].best_model_path
        if best_path:
            print(f"Loading best model from {best_path}")
            self.module = SciLaMALightningModule.load_from_checkpoint(
                best_path,
                config=self.config,
                n_features=self.datamodule.n_features,
                n_covariates=self.datamodule.n_covariates,
                total_samples=self.datamodule.adata.n_obs,
                feature_input_embeddings=self.datamodule.feature_input_embeddings,
                covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
            )

    def _train_stepwise(self):
        print("Starting Stepwise Training...")
        
        # Phase 1: Sample VAE
        print("\n=== Phase 1: Training Sample VAE ===")
        phase1_config = copy.deepcopy(self.config)
        phase1_config.training.mode = "beta_vae"
        
        module_phase1 = SciLaMALightningModule(
            config=phase1_config,
            n_features=self.datamodule.n_features,
            n_covariates=self.datamodule.n_covariates,
            total_samples=self.datamodule.adata.n_obs,
            feature_input_embeddings=self.datamodule.feature_input_embeddings,
            covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
        )
        
        callbacks_p1 = self._get_callbacks("stepwise_phase1")
        trainer_p1 = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            callbacks=callbacks_p1,
            accelerator="auto",
            devices=self.config.training.devices,
            strategy=self.config.training.strategy,
            log_every_n_steps=10
        )
        
        trainer_p1.fit(module_phase1, self.datamodule)
        
        best_p1_path = callbacks_p1[0].best_model_path
        print(f"Phase 1 complete. Best model: {best_p1_path}")
        
        # Load best phase 1 weights
        module_phase1 = SciLaMALightningModule.load_from_checkpoint(
            best_p1_path,
            config=phase1_config,
            n_features=self.datamodule.n_features,
            n_covariates=self.datamodule.n_covariates,
            total_samples=self.datamodule.adata.n_obs,
            feature_input_embeddings=self.datamodule.feature_input_embeddings,
            covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
        )
        
        # Phase 2: Joint Training
        print("\n=== Phase 2: Training Joint VAE (Sample Frozen) ===")
        phase2_config = copy.deepcopy(self.config)
        phase2_config.training.mode = "direct" 
        
        module_phase2 = SciLaMALightningModule(
            config=phase2_config,
            n_features=self.datamodule.n_features,
            n_covariates=self.datamodule.n_covariates,
            total_samples=self.datamodule.adata.n_obs,
            feature_input_embeddings=self.datamodule.feature_input_embeddings,
            covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
        )
        
        # Setup Xt if needed (only static_embedding genes)
        if not self.datamodule.feature_input_embeddings:
             static_mask = self.datamodule.adata.var["static_embedding"].values
             X = self.datamodule.adata[:, static_mask].X
             if hasattr(X, "toarray"):
                 X = X.toarray()
             Xt = torch.FloatTensor(X).t()
             module_phase2.setup_Xt(Xt)

        # Transfer weights
        print("Transferring Sample VAE weights...")
        module_phase2.sample_encoder.load_state_dict(module_phase1.sample_encoder.state_dict())
        module_phase2.sample_decoder.load_state_dict(module_phase1.sample_decoder.state_dict())
        
        # Freeze Sample VAE
        module_phase2.freeze_beta_vae()
        
        callbacks_p2 = self._get_callbacks("stepwise_phase2")
        trainer_p2 = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            callbacks=callbacks_p2,
            accelerator="auto",
            devices=self.config.training.devices,
            strategy=self.config.training.strategy,
            log_every_n_steps=10
        )
        
        trainer_p2.fit(module_phase2, self.datamodule)
        
        best_p2_path = callbacks_p2[0].best_model_path
        print(f"Phase 2 complete. Best model: {best_p2_path}")
        
        self.module = SciLaMALightningModule.load_from_checkpoint(
            best_p2_path,
            config=phase2_config,
            n_features=self.datamodule.n_features,
            n_covariates=self.datamodule.n_covariates,
            total_samples=self.datamodule.adata.n_obs,
            feature_input_embeddings=self.datamodule.feature_input_embeddings,
            covariate_encoding_state=self.datamodule.get_covariate_encoding_state(),
        )

    def predict(self):
        if self.module is None:
            raise ValueError("Model not trained yet. Call train() or load_checkpoint() first.")
        
        print("Running prediction on full dataset...")
        self.module.eval()
        
        # Ensure module is on device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        
        # Setup data if not already (handles load_checkpoint case)
        if self.datamodule.adata is None:
            self.datamodule.setup()
            
        pred_loader = self.datamodule.predict_dataloader()
        
        embeddings = []
        
        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(pred_loader, desc="Predicting"):
                x, c = batch
                x = x.to(device)
                c = c.to(device)
                # We always want sample embeddings, so use sample_encoder
                mu, sigma, z = self.module.sample_encoder(x, c)
                embeddings.append(mu.cpu().numpy())
                
        embeddings = np.concatenate(embeddings, axis=0)
        key = self.config.output.save_key
        self.datamodule.adata.obsm[key] = embeddings
        print(f"Stored sample embeddings in adata.obsm['{key}']")

        # Feature embeddings: save to parquet (static_embedding genes only)
        if self.config.output.save_feature_embeddings and getattr(self.module, "feature_encoder", None) is not None:
            import pandas as pd
            static_mask = self.datamodule.adata.var["static_embedding"].values
            with torch.no_grad():
                if self.module.feature_input_embeddings and len(self.module.feature_input_embeddings) > 0:
                    inputs = [emb.to(device) for emb in self.module.feature_input_embeddings.values()]
                    f_mu, _, _ = self.module.feature_encoder(inputs, None)
                    feature_emb = f_mu.cpu().numpy()
                else:
                    X = self.datamodule.adata[:, static_mask].X
                    Xt = torch.FloatTensor(X.toarray() if hasattr(X, "toarray") else X).t().to(device)
                    f_mu, _, _ = self.module.feature_encoder(Xt, None)
                    feature_emb = f_mu.cpu().numpy()
            gene_ids = self.datamodule.adata.var_names[static_mask].astype(str).tolist()
            embeddings = [feature_emb[i].tolist() for i in range(len(gene_ids))]
            feature_df = pd.DataFrame({"gene_id": gene_ids, "embedding": embeddings})
            out_path = os.path.join(self.config.output.save_dir, self.config.output.feature_embedding_filename)
            os.makedirs(self.config.output.save_dir, exist_ok=True)
            feature_df.to_parquet(out_path)
            print(f"Stored feature embeddings in {out_path}")
