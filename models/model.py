import torch
import numpy as np
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from modules import staticEncoderBase, dynamicEncoderBase, ConvSADecoder,\
    sequence_mask

class VAEbase(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.staticEncoder = staticEncoderBase(**model_config['static_encoder'])
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
    def forward(self, x, lenx, indi_ref):
        indi_mu, indi_log_std = self.staticEncoder(indi_ref)
        con_mu, con_log_std = self.dynamicEncoder(x, lenx)
        
        z_indi = self.staticEncoder.sample(indi_mu, indi_log_std)
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        
        time_len = z_con.size(2)
        z_indi_in = z_indi.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'indi_mu': indi_mu, 
                'indi_log_std': indi_log_std,}
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
        indi_mu = outputs['indi_mu']
        indi_log_std = outputs['indi_log_std']
         
        indi_kl = self.staticEncoder.kl_divergence(indi_mu, indi_log_std)
        con_kl = self.dynamicEncoder.kl_divergence(con_mu, con_log_std, lens)
        
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, mel, lens) + 0.5 * self.dec_normal_nll(x_rec, mel, lens)

        return nll, indi_kl, con_kl
        
    @staticmethod
    def dec_normal_nll(x_hat, x_gt, lens=None):
        if lens is not None:
            mask = sequence_mask(lens).to(x_hat.dtype)
        else:
            B = x_hat.size(0)
            T = x_hat.size(2)
            mask = torch.ones([B, T], dtype=x_hat.dtype, device=x_hat.device)
        dist_normal = D.Normal(x_hat, torch.ones_like(x_hat, requires_grad=False))
        logp = dist_normal.log_prob(x_gt)
        nll = - torch.sum(torch.sum(logp, dim=1) * mask, dim=1) / torch.sum(mask, dim=1)
        nll = torch.mean(nll)
        return nll    
        
    @staticmethod
    def dec_laplace_nll(x_hat, x_gt, lens=None):
        if lens is not None:
            mask = sequence_mask(lens).to(x_hat.dtype)
        else:
            B = x_hat.size(0)
            T = x_hat.size(2)
            mask = torch.ones([B, T], dtype=x_hat.dtype, device=x_hat.device)
        dist_laplace = D.Laplace(x_hat, torch.ones_like(x_hat, requires_grad=False))
        logp = dist_laplace.log_prob(x_gt)
        nll = - torch.sum(torch.sum(logp, dim=1) * mask, dim=1) / torch.sum(mask, dim=1)
        nll = torch.mean(nll)
        return nll
        
        
        
        

