import os
import copy
import random
import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset


# reproducibility
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.enabled = False
     #torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False


# initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0) if m.bias is not None else None


def add_weight_decay(model, output_layer=()):
    decay_param = []
    nodecay_param = []
    decay_name = []
    nodecay_name = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.split("_")[0] in output_layer:
            nodecay_name.append(name)
            nodecay_param.append(param)
        else:
            decay_name.append(name)
            decay_param.append(param)
    return decay_param, nodecay_param, decay_name, nodecay_name


#%%
def vae_training_mode(encoder, decoder):
    encoder.train()
    decoder.train()
    print("...")


#%%
def vae_evaluating_mode(encoder, decoder):
    encoder.eval()
    decoder.eval()
    print("...")


# Fully Connected Neural Networks
def FCNN(layers, batchnorm=False, layernorm=True, activation=nn.LeakyReLU(), dropout_rate=0):
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


# RNA Encoder
class RNA_ENCODER(nn.Module):
    def __init__(self, feature_dim:int, cov:int,
                 hidden_dim:int, latent_dim:int, 
                 batchnorm=True, layernorm=True, activation=nn.LeakyReLU(), 
                 dropout_rate=0, var_eps=0):
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
        enc_h = self.enc_hidden(torch.cat([x, c], dim=1)) if self.cov > 0 else self.enc_hidden(x)
        mu, sigma = self.bottleneck(enc_h)
        z = self.reparameterize(mu, sigma)
        return mu, sigma, z


#%%
# RNA Decoder
class RNA_DECODER(nn.Module):
    def __init__(self, feature_dim:int, cov:int,
                 hidden_dim:int, latent_dim:int, 
                 batchnorm=True, layernorm=True, activation=nn.LeakyReLU(), 
                 dropout_rate=0):
        super(RNA_DECODER, self).__init__()
        hidden_dim_decoder = list(np.flip(hidden_dim))
        self.cov = cov
        if len(hidden_dim_decoder) == 1:
            self.dec_hidden = nn.Linear(latent_dim + cov, hidden_dim_decoder[0], bias=True)
        elif len(hidden_dim_decoder) > 1:
            self.dec_hidden = nn.Sequential(FCNN([latent_dim + cov] + hidden_dim_decoder[:-1], 
                                                  batchnorm=batchnorm, layernorm=layernorm, 
                                                  activation=activation, dropout_rate=dropout_rate),
                                            nn.Linear(hidden_dim_decoder[-2], hidden_dim_decoder[-1], bias=True))
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
        dec_h = self.dec_hidden(torch.cat([z, c], dim=1)) if self.cov > 0 else self.dec_hidden(z)
        last_h = self.final_layer(dec_h)
        output_res = self.reconstruct_mean(last_h)
        return dec_h, output_res


##########################
# cell VAE loss function #
##########################
# VAE loss function: NLL + beta * KLD
def rna_vae_loss(raw, mu, sigma, output_res, beta):
    # mean squared error
    MSE = torch.nn.MSELoss(reduction="none")(output_res, raw).sum(dim=-1)
    NLL = MSE
    KLD = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0, 1)).sum(dim=-1)
    total_loss = (NLL + KLD * beta).mean()    # average over batch
    return total_loss, NLL.mean(), MSE.mean(), KLD.mean(), beta


#%%
def train_rnaVAE(inputs, cov, encoder, decoder, optimizer, kl_weight):
    mu, sigma, z = encoder(inputs, cov)
    dec_h, output_p = decoder(z, cov)
    total_loss, neg_log_likelihood, mean_square_error, kl_divergence, beta = \
    rna_vae_loss(inputs, mu, sigma, output_p, kl_weight)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#%%
def eval_rnaVAE(inputs, cov, encoder, decoder, kl_weight):
    mu, sigma, z = encoder(inputs, cov)
    dec_h, output_p = decoder(z, cov)
    eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence, beta = \
    rna_vae_loss(inputs, mu, sigma, output_p, kl_weight)
    return  eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence


##########################
# gene VAE loss function #
##########################
#%%
def rnafeature_vae_loss(x_batch, x_batch_prime, x_latent_batch_prime, 
                        mu, sigma, beta, gamma_): 
    MSE = torch.nn.MSELoss(reduction="none")(x_batch_prime, x_batch).sum(dim=-1) + \
          gamma_ * torch.nn.MSELoss(reduction="none")(x_latent_batch_prime, x_batch).sum(dim=-1)
    NLL = MSE
    KLD = (D.kl_divergence(D.Normal(mu, sigma), D.Normal(0, 1)).sum(dim=1))
    total_loss = NLL.mean() + KLD.mean() * beta  # average over batch
    return total_loss, NLL.mean(), MSE.mean(), KLD.mean(), beta


#%%
def train_rnafeatureVAE(X_t, x_batch, x_cov,
                        cell_encoder, cell_decoder, 
                        gene_encoder, gene_decoder, 
                        optimizer, kl_weight, gamma_=0.05):
    with torch.no_grad():
        cell_latent_batch = cell_encoder(x_batch, x_cov)[-1]            # B x latent
        cell_hidden_batch = cell_decoder(cell_latent_batch, x_cov)[0]   # B x last_hidden
    bias = cell_decoder.output_mean.bias
    mu, sigma, z      = gene_encoder(X_t, None)                         # G x latent
    dec_h, _          = gene_decoder(z, None)                           # G x last_hidden    
    x_latent_batch_prime = torch.mm(cell_latent_batch, z.t()) + bias    # B x G
    x_batch_prime = torch.mm(cell_hidden_batch, dec_h.t()) + bias       # B x G 
    total_loss, neg_log_likelihood, mean_square_error, kl_divergence, beta = \
    rnafeature_vae_loss(x_batch, x_batch_prime, x_latent_batch_prime, mu, sigma, beta=kl_weight, gamma_=gamma_)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#%%
def eval_rnafeatureVAE(X_t, x_batch, x_cov,
                       cell_encoder, cell_decoder, 
                       gene_encoder, gene_decoder, 
                       kl_weight, gamma_=0.05):
    with torch.no_grad():
        cell_latent_batch = cell_encoder(x_batch, x_cov)[-1]                # B x latent
        cell_hidden_batch = cell_decoder(cell_latent_batch, x_cov)[0]       # B x last_hidden
        bias = cell_decoder.output_mean.bias
        mu, sigma, z      = gene_encoder(X_t, None)                         # G x latent
        dec_h, _          = gene_decoder(z, None)                           # G x last_hidden    
        x_latent_batch_prime = torch.mm(cell_latent_batch, z.t()) + bias    # B x G
        x_batch_prime = torch.mm(cell_hidden_batch, dec_h.t()) + bias       # B x G 
        eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence, beta = \
        rnafeature_vae_loss(x_batch, x_batch_prime, x_latent_batch_prime, 
                            mu, sigma, beta=kl_weight, gamma_=gamma_)
    return eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence


###########################
# joint VAE loss function #
###########################
#%%
def si_rna_vae_loss(x_batch, x_out, 
                    x_batch_prime, x_latent_batch_prime, 
                    cmu, csigma, gmu, gsigma, beta, gamma_): 
    MSE = torch.nn.MSELoss(reduction="none")(x_batch_prime, x_batch).sum(dim=-1) + \
          gamma_ * torch.nn.MSELoss(reduction="none")(x_latent_batch_prime, x_batch).sum(dim=-1)
    NLL = MSE
    cKLD = D.kl_divergence(D.Normal(cmu, csigma), D.Normal(0, 1)).sum(dim=-1)
    gKLD = D.kl_divergence(D.Normal(gmu, gsigma), D.Normal(0, 1)).sum(dim=-1)
    KLD = cKLD.mean() + gKLD.mean()
    total_loss = NLL.mean() +  KLD * beta
    return total_loss, NLL.mean(), MSE.mean(), KLD, beta


#%%
def train_si_rnaVAE(X_t, x_batch, x_cov,
                    cell_encoder, cell_decoder, 
                    gene_encoder, gene_decoder, 
                    optimizer, kl_weight, gamma_=0.05):
    cmu, csigma, cz          = cell_encoder(x_batch, x_cov)                     # B x latent
    cell_hidden_batch, x_out = cell_decoder(cz, x_cov)                          # B x last_hidden
    bias = cell_decoder.output_mean.bias
    gmu, gsigma, gz          = gene_encoder(X_t, None)                          # G x latent
    dec_h, _                 = gene_decoder(gz, None)                           # G x last_hidden    
    x_latent_batch_prime = torch.mm(cz, gz.t()) + bias                          # B x G
    x_batch_prime = torch.mm(cell_hidden_batch, dec_h.t()) + bias               # B x G 
    total_loss, neg_log_likelihood, mean_square_error, kl_divergence, beta = \
    si_rna_vae_loss(x_batch, x_out, x_batch_prime, x_latent_batch_prime, 
                    cmu, csigma, gmu, gsigma, beta=kl_weight, gamma_=gamma_)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#%%
def eval_si_rnaVAE(X_t, x_batch, x_cov,
                    cell_encoder, cell_decoder, 
                    gene_encoder, gene_decoder, 
                    kl_weight, gamma_=0.05):
    with torch.no_grad():
        cmu, csigma, cz          = cell_encoder(x_batch, x_cov)                 # B x latent
        cell_hidden_batch, x_out = cell_decoder(cz, x_cov)                      # B x last_hidden
        bias = cell_decoder.output_mean.bias
        gmu, gsigma, gz          = gene_encoder(X_t, None)                      # G x latent
        dec_h, _                 = gene_decoder(gz, None)                       # G x last_hidden    
        x_latent_batch_prime = torch.mm(cz, gz.t()) + bias                      # B x G
        x_batch_prime = torch.mm(cell_hidden_batch, dec_h.t()) + bias           # B x G 
        eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence, beta = \
        si_rna_vae_loss(x_batch, x_out, x_batch_prime, x_latent_batch_prime, 
                        cmu, csigma, gmu, gsigma, beta=kl_weight, gamma_=gamma_)
    return eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence



