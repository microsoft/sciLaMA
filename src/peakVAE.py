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


# ATAC Encoder
class ATAC_ENCODER(nn.Module):
    def __init__(self, feature_dim:int, cov:int, 
                 hidden_dim:int, latent_dim:int, 
                 batchnorm=False, layernorm=True, activation=nn.LeakyReLU(), 
                 lib_imp=True,
                 dropout_rate=0, var_eps=0):
        super(ATAC_ENCODER, self).__init__()
        self.var_eps = var_eps
        self.cov = cov
        self.enc_hidden = FCNN([feature_dim + cov] + hidden_dim, 
                                batchnorm=batchnorm, layernorm=layernorm,
                                activation=activation, dropout_rate=dropout_rate)
        self.latent_mu = nn.Linear(hidden_dim[-1], latent_dim, bias=True)
        self.latent_sigma = nn.Linear(hidden_dim[-1], latent_dim, bias=True)
        # sample factor: d
        if lib_imp:
            self.lib_imp = nn.Sequential(FCNN([feature_dim + cov] + hidden_dim, 
                                            batchnorm=batchnorm, layernorm=layernorm,
                                            activation=activation, dropout_rate=dropout_rate),
                                        nn.Linear(hidden_dim[-1], 1, bias=True))
        else:
            self.lib_imp = None
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
        if self.lib_imp:
            d = self.lib_imp(torch.cat([x, c], dim=1)) if self.cov > 0 else self.lib_imp(x)
        else:
            d = None
        return mu, sigma, z, d


#%%
# ATAC Decoder
class ATAC_DECODER(nn.Module):
    def __init__(self, feature_dim:int, cov:int, 
                 hidden_dim:int, latent_dim:int, 
                 batchnorm=False, layernorm=True, activation=nn.LeakyReLU(), 
                 dropout_rate=0,
                 lib_imp=True,
                 feature_factor=True):
        super(ATAC_DECODER, self).__init__()
        # sigmoid for bernoulli distribution
        self.output_sigmoid = nn.Sigmoid()
        # form of the hidden layers and dimension
        hidden_dim_decoder = list(np.flip(hidden_dim))
        self.cov = cov
        # feature factor: f
        if feature_factor:
            self.feature_factor = torch.nn.Parameter(torch.zeros(feature_dim))
        else:
            self.feature_factor = None
        # data flow for: p
        if len(hidden_dim_decoder) == 1:
            self.dec_hidden = nn.Linear(latent_dim + cov, hidden_dim_decoder[0], bias=True)
        else:
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
        self.output_p = nn.Linear(hidden_dim_decoder[-1], feature_dim, bias=True)
    def reconstruct(self, dec_h):
        p_x = self.output_p(dec_h)
        return p_x
    def forward(self, z, d, c):
        dec_h = self.dec_hidden(torch.cat([z, c], dim=1)) if self.cov > 0 else self.dec_hidden(z)
        last_h = self.final_layer(dec_h)
        p = self.reconstruct(last_h)    # p (last hidden layer x the weight of the output layer)    
        if d is not None:               
            self.sample_factor = d      # d (sample factor from encoder)
            output_res = torch.sigmoid(p) * torch.sigmoid(d)                # p * d
        else:
            d = None
            self.sample_factor = None
            output_res = torch.sigmoid(p)
        if self.feature_factor is not None:
            f = self.feature_factor.clone()
            output_res = output_res * torch.sigmoid(f)
        else:
            f = None
            output_res = output_res
        return dec_h, {'p':p, 'd':d, 'f':f}, output_res
# # p, d, f, x
# # p * d * f --> output_res
# rl = torch.nn.BCELoss(reduction="none")(output_res, (x > 0).float()).sum(dim=-1)


##########################
# cell VAE loss function #
##########################
# VAE loss function: BCE + beta * KLD
def atac_vae_loss(raw, mu, sigma, output_res, beta):
        # binary cross entropy
        BCE = torch.nn.BCELoss(reduction="none")(output_res, (raw>0).float()).sum(dim=-1)
        NLL = BCE
        KLD = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0, 1)).sum(dim=-1)
        total_loss = (NLL.sum() + KLD * beta).mean() #  NLL.mean() + KLD.mean() * beta          
        # peakVI: https://github.com/scverse/scvi-tools/blob/main/src/scvi/module/_peakvae.py
        return total_loss, NLL.mean(), BCE.mean(), KLD.mean(), beta


#%%
def train_atacVAE(inputs, cov, encoder, decoder, optimizer, kl_weight):
    mu, sigma, z, d = encoder(inputs, cov)
    dec_h, _, output_p = decoder(z, d, cov)
    total_loss, neg_log_likelihood, binary_cross_entropy, kl_divergence, beta = \
    atac_vae_loss(inputs, mu, sigma, output_p, kl_weight)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#%%
def eval_atacVAE(inputs, cov, encoder, decoder, kl_weight):
    mu, sigma, z, d = encoder(inputs, cov)
    dec_h, _, output_p = decoder(z, d, cov)
    eval_total_loss, eval_neg_log_likelihood, eval_binary_cross_entropy, eval_kl_divergence, beta = \
    atac_vae_loss(inputs, mu, sigma, output_p, kl_weight)
    return  eval_total_loss, eval_neg_log_likelihood, eval_binary_cross_entropy, eval_kl_divergence


##########################
# gene VAE loss function #
##########################
#%%
# VAE loss function: BCE + beta * KLD
#%%
def atacfeature_vae_loss(x_batch, x_batch_prime, x_latent_batch_prime, 
                         mu, sigma, beta, gamma_): 
    BCE = torch.nn.BCELoss(reduction="none")(x_batch_prime, (x_batch>0).float()).sum(dim=-1) + \
          gamma_ * torch.nn.BCELoss(reduction="none")(x_latent_batch_prime, (x_batch>0).float()).sum(dim=-1)
    NLL = BCE
    KLD = (D.kl_divergence(D.Normal(mu, sigma), D.Normal(0, 1)).sum(dim=1))
    total_loss = NLL.mean() + (KLD * beta).mean() #  NLL.mean() + KLD.mean() * beta
    return total_loss, NLL.mean(), BCE.mean(), KLD.mean(), beta


#%%
def train_atacfeatureVAE(X_t, x_batch, x_cov,
                         cell_encoder, cell_decoder, 
                         peak_encoder, peak_decoder, 
                         optimizer, kl_weight, gamma_=0.05):
    with torch.no_grad():
        _, _, cell_latent_batch, cell_d = cell_encoder(x_batch, x_cov)          # B x latent
        cell_hidden_batch = cell_decoder(cell_latent_batch, cell_d, x_cov)[0]   # B x last_hidden
    bias = cell_decoder.output_p.bias
    cell_f = cell_decoder.feature_factor
    mu, sigma, z, _       = peak_encoder(X_t, None)                             # P x latent
    dec_h, _, _           = peak_decoder(z, None, None)                         # P x last_hidden 
    x_latent_p = torch.mm(cell_latent_batch, z.t()) + bias                      # B x p
    x_hidden_p = torch.mm(cell_hidden_batch, dec_h.t()) + bias                  # B x P 
    if cell_d is not None:
        x_latent_batch_prime = torch.sigmoid(x_latent_p) * torch.sigmoid(cell_d)
        x_batch_prime = torch.sigmoid(x_hidden_p) * torch.sigmoid(cell_d)
    else:
        x_latent_batch_prime = torch.sigmoid(x_latent_p)
        x_batch_prime = torch.sigmoid(x_hidden_p)
    if cell_f is not None:
        x_latent_batch_prime = x_latent_batch_prime * torch.sigmoid(cell_f)
        x_batch_prime = x_batch_prime * torch.sigmoid(cell_f)
    else:
        x_latent_batch_prime = x_latent_batch_prime
        x_batch_prime = x_batch_prime
    total_loss, neg_log_likelihood, mean_square_error, kl_divergence, beta = \
    atacfeature_vae_loss(x_batch, x_batch_prime, x_latent_batch_prime, mu, sigma, beta=kl_weight, gamma_=gamma_)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#%%
def eval_atacfeatureVAE(X_t, x_batch, x_cov,
                       cell_encoder, cell_decoder, 
                       peak_encoder, peak_decoder, 
                       kl_weight, gamma_=0.05):
    with torch.no_grad():
        _, _, cell_latent_batch, cell_d = cell_encoder(x_batch, x_cov)          # B x latent
        cell_hidden_batch = cell_decoder(cell_latent_batch, cell_d, x_cov)[0]   # B x last_hidden
        bias = cell_decoder.output_p.bias
        cell_f = cell_decoder.feature_factor
        mu, sigma, z, _       = peak_encoder(X_t, None)                         # P x latent
        dec_h, _, _           = peak_decoder(z, None, None)                     # P x last_hidden 
        x_latent_p = torch.mm(cell_latent_batch, z.t()) + bias                  # B x P
        x_hidden_p = torch.mm(cell_hidden_batch, dec_h.t()) + bias              # B x P 
        if cell_d is not None:
            x_latent_batch_prime = torch.sigmoid(x_latent_p) * torch.sigmoid(cell_d)
            x_batch_prime = torch.sigmoid(x_hidden_p) * torch.sigmoid(cell_d)
        else:
            x_latent_batch_prime = torch.sigmoid(x_latent_p)
            x_batch_prime = torch.sigmoid(x_hidden_p)
        if cell_f is not None:
            x_latent_batch_prime = x_latent_batch_prime * torch.sigmoid(cell_f)
            x_batch_prime = x_batch_prime * torch.sigmoid(cell_f)
        else:
            x_latent_batch_prime = x_latent_batch_prime
            x_batch_prime = x_batch_prime
        eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence, beta = \
        atacfeature_vae_loss(x_batch, x_batch_prime, x_latent_batch_prime, 
                            mu, sigma, beta=kl_weight, gamma_=gamma_)
    return eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_kl_divergence


###########################
# joint VAE loss function #
###########################
#%%
def si_atac_vae_loss(x_batch, x_out, 
                     x_batch_prime, x_latent_batch_prime, 
                     cmu, csigma, gmu, gsigma, beta, gamma_): 
    BCE = torch.nn.BCELoss(reduction="none")(x_batch_prime, (x_batch>0).float()).sum(dim=-1) + \
          gamma_ * torch.nn.BCELoss(reduction="none")(x_latent_batch_prime, (x_batch>0).float()).sum(dim=-1)
    NLL = BCE
    cKLD = D.kl_divergence(D.Normal(cmu, csigma), D.Normal(0, 1)).sum(dim=-1)
    pKLD = D.kl_divergence(D.Normal(gmu, gsigma), D.Normal(0, 1)).sum(dim=-1)
    total_loss = (NLL.sum() + cKLD * beta).mean() + (pKLD * beta).mean()
    # NLL.mean() + beta * (cKLD.mean() + pKLD.mean())
    return total_loss, NLL.mean(), BCE.mean(), cKLD.mean(), pKLD.mean(), beta


#%%
def train_si_atacVAE(X_t, x_batch, x_cov,
                    cell_encoder, cell_decoder, 
                    peak_encoder, peak_decoder, 
                    optimizer, kl_weight, gamma_=0.05):
    cmu, csigma, cz, cd         = cell_encoder(x_batch, x_cov)                  # B x latent
    cell_hidden_batch, _, x_out = cell_decoder(cz, cd, x_cov)                   # B x last_hidden
    bias = cell_decoder.output_p.bias
    cf = cell_decoder.feature_factor
    gmu, gsigma, gz, _          = peak_encoder(X_t, None)                       # P x latent
    dec_h, _, _                 = peak_decoder(gz, None, None)                  # P x last_hidden 
    x_latent_p = torch.mm(cz, gz.t()) + bias                                    # B x P
    x_hidden_p = torch.mm(cell_hidden_batch, dec_h.t()) + bias                  # B x P 
    if cd is not None:
        x_latent_batch_prime = torch.sigmoid(x_latent_p) * torch.sigmoid(cd)
        x_batch_prime = torch.sigmoid(x_hidden_p) * torch.sigmoid(cd)
    else:
        x_latent_batch_prime = torch.sigmoid(x_latent_p)
        x_batch_prime = torch.sigmoid(x_hidden_p)
    if cf is not None:
        x_latent_batch_prime = x_latent_batch_prime * torch.sigmoid(cf)
        x_batch_prime = x_batch_prime * torch.sigmoid(cf)
    else:
        x_latent_batch_prime = x_latent_batch_prime
        x_batch_prime = x_batch_prime
    total_loss, neg_log_likelihood, mean_square_error, ckl_divergence, pkl_divergence, beta = \
    si_atac_vae_loss(x_batch, x_out, x_batch_prime, x_latent_batch_prime, 
                    cmu, csigma, gmu, gsigma, beta=kl_weight, gamma_=gamma_)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#%%
def eval_si_atacVAE(X_t, x_batch, x_cov,
                    cell_encoder, cell_decoder, 
                    peak_encoder, peak_decoder, 
                    kl_weight, gamma_=0.05):
    with torch.no_grad():
        cmu, csigma, cz, cd         = cell_encoder(x_batch, x_cov)                  # B x latent
        cell_hidden_batch, _, x_out = cell_decoder(cz, cd, x_cov)                   # B x last_hidden
        bias = cell_decoder.output_p.bias
        cf = cell_decoder.feature_factor
        gmu, gsigma, gz, _          = peak_encoder(X_t, None)                       # P x latent
        dec_h, _, _                 = peak_decoder(gz, None, None)                  # P x last_hidden 
        x_latent_p = torch.mm(cz, gz.t()) + bias                                    # B x P
        x_hidden_p = torch.mm(cell_hidden_batch, dec_h.t()) + bias                  # B x P 
        if cd is not None:
            x_latent_batch_prime = torch.sigmoid(x_latent_p) * torch.sigmoid(cd)
            x_batch_prime = torch.sigmoid(x_hidden_p) * torch.sigmoid(cd)
        else:
            x_latent_batch_prime = torch.sigmoid(x_latent_p)
            x_batch_prime = torch.sigmoid(x_hidden_p)
        if cf is not None:
            x_latent_batch_prime = x_latent_batch_prime * torch.sigmoid(cf)
            x_batch_prime = x_batch_prime * torch.sigmoid(cf)
        else:
            x_latent_batch_prime = x_latent_batch_prime
            x_batch_prime = x_batch_prime
        eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_ckl_divergence, eval_pkl_divergence, beta = \
        si_atac_vae_loss(x_batch, x_out, x_batch_prime, x_latent_batch_prime, 
                        cmu, csigma, gmu, gsigma, beta=kl_weight, gamma_=gamma_)
    return eval_total_loss, eval_neg_log_likelihood, eval_mean_square_error, eval_ckl_divergence, eval_pkl_divergence
