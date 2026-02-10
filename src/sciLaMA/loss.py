import torch
import torch.nn as nn
import torch.distributions as D
from typing import Tuple


def sample_vae_loss(raw: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                    output_res: torch.Tensor, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample VAE loss: MSE + beta * KLD.
    """
    mse = nn.MSELoss(reduction="none")(output_res, raw).sum(dim=-1)
    nll = mse
    kld = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0, 1)).sum(dim=-1)
    total_loss = (nll + kld * beta).mean()
    return total_loss, nll.mean(), mse.mean(), kld.mean()


def feature_vae_loss(
    x_batch: torch.Tensor,
    x_recon_from_hidden: torch.Tensor,
    x_recon_from_latent: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    beta: float,
    gamma_: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Feature VAE loss.
    NLL: mean over batch (per-sample). KLD: mean over features. This scaling is intentional (matches original).
    """
    mse_recon_from_hidden = nn.MSELoss(reduction="none")(x_recon_from_hidden, x_batch).sum(dim=-1)
    mse_recon_from_latent = nn.MSELoss(reduction="none")(x_recon_from_latent, x_batch).sum(dim=-1)
    mse = mse_recon_from_hidden + gamma_ * mse_recon_from_latent
    nll = mse

    kld = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0, 1)).sum(dim=1)
    total_loss = nll.mean() + kld.mean() * beta
    return total_loss, nll.mean(), mse.mean(), kld.mean()


def joint_scilama_loss(
    x_batch: torch.Tensor,
    x_out: torch.Tensor,
    x_recon_from_hidden: torch.Tensor,
    x_recon_from_latent: torch.Tensor,
    sample_mu: torch.Tensor,
    sample_sigma: torch.Tensor,
    feature_mu: torch.Tensor,
    feature_sigma: torch.Tensor,
    beta: float,
    gamma_: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Joint sciLaMA loss (sample + feature VAEs).
    """
    mse_recon_from_hidden = nn.MSELoss(reduction="none")(x_recon_from_hidden, x_batch).sum(dim=-1)
    mse_recon_from_latent = nn.MSELoss(reduction="none")(x_recon_from_latent, x_batch).sum(dim=-1)
    mse = mse_recon_from_hidden + gamma_ * mse_recon_from_latent
    nll = mse

    sample_kld = D.kl_divergence(D.Normal(sample_mu, sample_sigma), D.Normal(0, 1)).sum(dim=-1)
    feature_kld = D.kl_divergence(D.Normal(feature_mu, feature_sigma), D.Normal(0, 1)).sum(dim=-1)
    kld = sample_kld.mean() + feature_kld.mean()

    total_loss = nll.mean() + kld * beta
    return total_loss, nll.mean(), mse.mean(), kld
