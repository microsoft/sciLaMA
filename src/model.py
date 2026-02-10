import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from typing import List


def FCNN(layers: List[int], batchnorm: bool = False, layernorm: bool = True,
         activation: nn.Module = nn.LeakyReLU(), dropout_rate: float = 0):
    fc_nn = []
    for i in range(1, len(layers)):
        fc_nn.append(nn.Linear(layers[i-1], layers[i]))
        if batchnorm:
            fc_nn.append(nn.BatchNorm1d(layers[i]))
        if layernorm:
            fc_nn.append(nn.LayerNorm(layers[i], elementwise_affine=False))
        if activation:
            fc_nn.append(activation)
        fc_nn.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*fc_nn)


class RNA_ENCODER(nn.Module):
    def __init__(self, feature_dim: int, cov: int,
                 hidden_dim: List[int], latent_dim: int,
                 batchnorm: bool = True, layernorm: bool = True,
                 activation: nn.Module = nn.LeakyReLU(),
                 dropout_rate: float = 0, var_eps: float = 0):
        super(RNA_ENCODER, self).__init__()
        self.var_eps = var_eps
        self.cov = cov
        self.enc_hidden = FCNN([feature_dim + cov] + hidden_dim,
                               batchnorm=batchnorm, layernorm=layernorm,
                               activation=activation, dropout_rate=dropout_rate)
        self.latent_mu = nn.Linear(hidden_dim[-1], latent_dim, bias=True)
        self.latent_sigma = nn.Linear(hidden_dim[-1], latent_dim, bias=True)

    def reparameterize(self, mu, sigma):
        z = D.Normal(mu, sigma).rsample()
        return z

    def bottleneck(self, enc_h):
        mu = self.latent_mu(enc_h)
        sigma = (torch.exp(self.latent_sigma(enc_h)) + self.var_eps).sqrt()
        return mu, sigma

    def forward(self, x, c):
        if self.cov > 0:
            if c is None:
                raise ValueError("Covariates expected but not provided.")
            enc_input = torch.cat([x, c], dim=1)
        else:
            enc_input = x

        enc_h = self.enc_hidden(enc_input)
        mu, sigma = self.bottleneck(enc_h)
        z = self.reparameterize(mu, sigma)
        return mu, sigma, z


class RNA_DECODER(nn.Module):
    def __init__(self, feature_dim: int, cov: int,
                 hidden_dim: List[int], latent_dim: int,
                 batchnorm: bool = True, layernorm: bool = True,
                 activation: nn.Module = nn.LeakyReLU(),
                 dropout_rate: float = 0):
        super(RNA_DECODER, self).__init__()
        hidden_dim_decoder = list(np.flip(hidden_dim))
        self.cov = cov

        if len(hidden_dim_decoder) == 1:
            self.dec_hidden = nn.Linear(latent_dim + cov, hidden_dim_decoder[0], bias=True)
        elif len(hidden_dim_decoder) > 1:
            self.dec_hidden = nn.Sequential(
                FCNN([latent_dim + cov] + hidden_dim_decoder[:-1],
                     batchnorm=batchnorm, layernorm=layernorm,
                     activation=activation, dropout_rate=dropout_rate),
                nn.Linear(hidden_dim_decoder[-2], hidden_dim_decoder[-1], bias=True)
            )

        self.final_layer_list = []
        if batchnorm:
            self.final_layer_list.append(nn.BatchNorm1d(hidden_dim_decoder[-1]))
        if dropout_rate > 0:
            self.final_layer_list.append(nn.Dropout(dropout_rate))
        self.final_layer = nn.Sequential(*self.final_layer_list)
        self.output_mean = nn.Linear(hidden_dim_decoder[-1], feature_dim, bias=True)

    def reconstruct_mean(self, dec_h):
        mean_x = self.output_mean(dec_h)
        return mean_x

    def forward(self, z, c):
        if self.cov > 0:
            if c is None:
                raise ValueError("Covariates expected but not provided.")
            dec_input = torch.cat([z, c], dim=1)
        else:
            dec_input = z

        dec_h = self.dec_hidden(dec_input)
        last_h = self.final_layer(dec_h)
        output_res = self.reconstruct_mean(last_h)
        return dec_h, output_res


def fuse_latents(mu_list: List[torch.Tensor], sigma_list: List[torch.Tensor], fuse: str = 'average'):
    if fuse == 'average':
        fused_mu = sum(mu_list) / len(mu_list)
        fused_sigma = sum(sigma_list) / len(sigma_list)
    elif fuse == 'MoE':
        inv_vars = [1 / (sigma ** 2 + 1e-8) for sigma in sigma_list]
        fused_mu = sum(mu * inv_var for mu, inv_var in zip(mu_list, inv_vars)) / sum(inv_vars)
        fused_sigma = torch.sqrt(1 / sum(inv_vars))
    elif fuse == 'PoE':
        precision = sum(1 / (sigma ** 2 + 1e-8) for sigma in sigma_list)
        fused_mu = sum(mu / (sigma ** 2 + 1e-8) for mu, sigma in zip(mu_list, sigma_list)) / precision
        fused_sigma = torch.sqrt(1 / precision)
    else:
        raise ValueError("Fusion method not recognized. Choose 'average', 'MoE' or 'PoE'.")
    return fused_mu, fused_sigma


class MultiModalFeatureEncoder(nn.Module):
    """Multi-modal encoder for feature-level embeddings."""
    def __init__(self, feature_dims: List[int], hidden_dims: List[int], latent_dim: int, fuse: str = 'MoE',
                 batchnorm: bool = True, layernorm: bool = True, activation: nn.Module = nn.LeakyReLU(),
                 dropout_rate: float = 0, var_eps: float = 0):
        super(MultiModalFeatureEncoder, self).__init__()
        self.modality_dim = len(feature_dims)
        self.encoders = nn.ModuleList([
            RNA_ENCODER(feature_dim, 0, hidden_dims, latent_dim,
                        batchnorm, layernorm, activation, dropout_rate, var_eps)
            for feature_dim in feature_dims
        ])
        self.fuse = fuse

    def forward(self, x_list, c_list=None):
        mu_list, sigma_list, z_list = [], [], []
        for i, encoder in enumerate(self.encoders):
            c = None
            mu, sigma, z = encoder(x_list[i], c)
            mu_list.append(mu)
            sigma_list.append(sigma)
            z_list.append(z)
        fused_mu, fused_sigma = fuse_latents(mu_list, sigma_list, fuse=self.fuse)
        fused_z = D.Normal(fused_mu, fused_sigma).rsample()
        return fused_mu, fused_sigma, fused_z


class MultiModalFeatureDecoder(nn.Module):
    """Single decoder for the fused latent."""
    def __init__(self, feature_dims: List[int], hidden_dims: List[int], latent_dim: int,
                 batchnorm: bool = True, layernorm: bool = True, activation: nn.Module = nn.LeakyReLU(),
                 dropout_rate: float = 0):
        super(MultiModalFeatureDecoder, self).__init__()
        output_dim = feature_dims[0]
        self.decoder = RNA_DECODER(output_dim, 0, hidden_dims, latent_dim,
                                   batchnorm, layernorm, activation, dropout_rate)

    def forward(self, z, c_list=None):
        dec_h, output = self.decoder(z, None)
        return dec_h, output
