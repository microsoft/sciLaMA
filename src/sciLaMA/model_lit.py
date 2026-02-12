import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from .model import (
    RNA_ENCODER,
    RNA_DECODER,
    MultiModalFeatureEncoder,
)
from .loss import sample_vae_loss, feature_vae_loss, joint_scilama_loss
from .metrics import pearson_reconstruction, spearman_reconstruction
from .config import SciLaMAConfig
from .utils import init_weights


class SciLaMALightningModule(pl.LightningModule):
    def __init__(self, config: SciLaMAConfig, n_features: int, n_covariates: int,
                 total_samples: int, feature_input_embeddings: None | Dict[str, torch.Tensor] = None,
                 covariate_encoding_state: None | Dict[str, Any] = None,
                 feature_only_phase: bool = False):
        """
        feature_only_phase: True only for stepwise phase 2 (feature VAE only, sample frozen).
        Uses feature_vae_loss instead of joint_scilama_loss. Not a config mode.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["feature_input_embeddings"])
        self.config = config
        self.feature_only_phase = feature_only_phase
        self.n_features = n_features
        self.n_covariates = n_covariates
        self.total_samples = total_samples
        self.feature_input_embeddings = feature_input_embeddings
        self.covariate_encoding_state = covariate_encoding_state

        hidden_dims = config.model.hidden_dims
        latent_dim = config.model.latent_dim
        dropout = config.model.dropout_rate
        bn = config.model.batchnorm
        ln = config.model.layernorm
        act_cls = getattr(nn, config.model.activation, nn.LeakyReLU)
        activation = act_cls()
        var_eps = config.model.var_eps

        # Sample VAE: cells × genes -> z_s (sample latent). Input dim = n_genes.
        self.sample_encoder = RNA_ENCODER(
            feature_dim=n_features, cov=n_covariates,
            hidden_dim=hidden_dims, latent_dim=latent_dim,
            batchnorm=bn, layernorm=ln, activation=activation,
            dropout_rate=dropout, var_eps=var_eps
        )
        self.sample_decoder = RNA_DECODER(
            cov=n_covariates, hidden_dim=hidden_dims, latent_dim=latent_dim,
            batchnorm=bn, layernorm=ln, activation=activation,
            dropout_rate=dropout, feature_dim=n_features
        )

        self.sample_encoder.apply(init_weights)
        self.sample_decoder.apply(init_weights)

        # Feature VAE (gene-level; only for direct/stepwise, not beta_vae)
        self.feature_encoder = None
        self.feature_decoder = None

        if config.training.mode != "beta_vae":
            # option A: External embeddings (e.g. LLM-derived gene_text, gene_protein)
            if self.feature_input_embeddings and len(self.feature_input_embeddings) > 0:
                feature_dims = [emb.shape[1] for emb in self.feature_input_embeddings.values()]
                self.feature_encoder = MultiModalFeatureEncoder(
                    feature_dims=feature_dims, hidden_dims=hidden_dims, latent_dim=latent_dim,
                    fuse=config.model.fusion_method, batchnorm=bn, layernorm=ln,
                    activation=activation, dropout_rate=dropout, var_eps=var_eps
                )
                # output_layer=False: decoder only produces dec_h for x_batch_prime = dec_h @ f_dec_h.T
                self.feature_decoder = RNA_DECODER(
                    cov=0, hidden_dim=hidden_dims, latent_dim=latent_dim,
                    batchnorm=bn, layernorm=ln, activation=activation,
                    dropout_rate=dropout, output_layer=False
                )
            # option B: No external embeddings: use Xt (genes × cells, transposed expression)
            else:
                self.feature_encoder = RNA_ENCODER(
                    feature_dim=total_samples, cov=0,
                    hidden_dim=hidden_dims, latent_dim=latent_dim,
                    batchnorm=bn, layernorm=ln, activation=activation,
                    dropout_rate=dropout, var_eps=var_eps
                )
                self.feature_decoder = RNA_DECODER(
                    cov=0, hidden_dim=hidden_dims, latent_dim=latent_dim,
                    batchnorm=bn, layernorm=ln, activation=activation,
                    dropout_rate=dropout, output_layer=False
                )

            if self.feature_encoder:
                self.feature_encoder.apply(init_weights)
            if self.feature_decoder:
                self.feature_decoder.apply(init_weights)

        # buffer for X.T when feature VAE has no external embeddings (option B)
        self.Xt = None

    def setup_Xt(self, Xt: torch.Tensor):
        self.register_buffer("Xt", Xt)

    def freeze_beta_vae(self):
        for param in self.sample_encoder.parameters():
            param.requires_grad = False
        for param in self.sample_decoder.parameters():
            param.requires_grad = False
        print("Sample VAE frozen.")

    def forward(self, x, c):
        return self.sample_encoder(x, c)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

    def _get_kl_weight(self):
        if self.current_epoch <= self.config.training.epochs_before_beta_warmup:
            return self.config.training.beta_start
        epochs_since_warmup = self.current_epoch - self.config.training.epochs_before_beta_warmup
        weight = self.config.training.beta_start + self.config.training.beta_warmup_rate * epochs_since_warmup
        return min(self.config.training.beta_end, weight)

    def _forward_feature_branch(self):
        """Forward through feature encoder/decoder. Returns (mu_f, sigma_f, z_f, f_dec_h)."""
        if self.feature_input_embeddings:
            inputs = [emb.to(self.device) for emb in self.feature_input_embeddings.values()]
            mu_f, sigma_f, z_f = self.feature_encoder(inputs, None)
        else:
            if self.Xt is None:
                raise ValueError("Xt (transposed data) is required for feature VAE without external embeddings.")
            mu_f, sigma_f, z_f = self.feature_encoder(self.Xt, None)
        f_dec_h, _ = self.feature_decoder(z_f, None)
        return mu_f, sigma_f, z_f, f_dec_h

    def _forward_sample_and_recon(self, x, c, mu_f, z_f, f_dec_h):
        """
        Forward sample VAE and compute x_batch_prime = dec_h @ f_dec_h.T + b,
        x_latent_prime = z_s @ z_f.T + b for joint/feature loss.
        """
        mu_s, sigma_s, z_s = self.sample_encoder(x, c)
        sample_hidden_batch, x_out = self.sample_decoder(z_s, c)
        bias = self.sample_decoder.output_mean.bias
        x_latent_batch_prime = torch.mm(z_s, z_f.t()) + bias
        x_batch_prime = torch.mm(sample_hidden_batch, f_dec_h.t()) + bias
        return mu_s, sigma_s, z_s, x_out, x_batch_prime, x_latent_batch_prime

    def _compute_loss_and_metrics(self, x, c, beta: float):
        """Compute total loss, NLL, MSE, KLD, Pearson, Spearman."""
        if self.config.training.mode == "beta_vae":
            # beta_vae or stepwise phase 1 —> optimize sample VAE only
            mu, sigma, z = self.sample_encoder(x, c)
            _, output_p = self.sample_decoder(z, c)
            total_loss, nll, mse, kld = sample_vae_loss(x, mu, sigma, output_p, beta)
            pearson = pearson_reconstruction(output_p, x)
            spearman = spearman_reconstruction(output_p, x)
            return total_loss, nll, mse, kld, pearson, spearman
        
        mu_s, sigma_s, z_s, x_out, x_batch_prime, x_latent_batch_prime = self._forward_sample_and_recon(x, c, mu_f, z_f, f_dec_h)
        mu_f, sigma_f, z_f, f_dec_h = self._forward_feature_branch()

        if self.feature_only_phase:
            # stepwise phase 2 —> optimize feature VAE only, freeze sample VAE (gamma=0)
            total_loss, nll, mse, kld = feature_vae_loss(
                x, x_batch_prime, x_latent_batch_prime,
                mu_f, sigma_f, beta, self.config.training.gamma
            )
        else:
            # direct or stepwise phase 3 —> optimize joint sample + feature VAEs (gamma = 0.05 by default)
            total_loss, nll, mse, kld = joint_scilama_loss(
                x, x_out, x_batch_prime, x_latent_batch_prime,
                mu_s, sigma_s, mu_f, sigma_f,
                beta, self.config.training.gamma
            )
        pearson = pearson_reconstruction(x_batch_prime, x)
        spearman = spearman_reconstruction(x_batch_prime, x)
        return total_loss, nll, mse, kld, pearson, spearman

    def training_step(self, batch, batch_idx):
        x, c = batch
        beta = self._get_kl_weight()
        total_loss, nll, mse, kld, pearson, spearman = self._compute_loss_and_metrics(x, c, beta)
        self.log_dict({
            "train_loss": total_loss, "train_nll": nll, "train_mse": mse, "train_kld": kld, "beta": beta,
            "train_pearson": pearson, "train_spearman": spearman,
        }, prog_bar=True, on_step=True, on_epoch=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, c = batch
        total_loss, nll, mse, kld, pearson, spearman = self._compute_loss_and_metrics(x, c, beta=1.0)
        self.log_dict({
            "val_loss": total_loss, "val_nll": nll, "val_mse": mse, "val_kld": kld, "val_pearson": pearson, "val_spearman": spearman,
        }, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
