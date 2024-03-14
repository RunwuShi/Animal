import torch
import numpy as np
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from modules import staticEncoderBase, staticEncoderJAVE, staticEncoderBase_Vector, dynamicEncoderBase, indi_dynamicEncoderBase, ConvSADecoder,\
    sequence_mask, LinearUnit, ConvUnit1D, ConvUnitTranspose1D, VectorQuantizerEMA, VectorQuantizer

EPS = 1e-12

class VAEbase(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.staticEncoder = staticEncoderBase(**model_config['static_encoder'])
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
    def forward(self, x, lenx, indi_ref):
        indi_mu, indi_log_std = self.staticEncoder(indi_ref) # [B, individual embedding]
        z_indi = self.staticEncoder.sample(indi_mu, indi_log_std)
        
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, content embedding, T]
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        
        time_len = z_con.size(2)
        z_indi_in = z_indi.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'indi_mu': indi_mu, 
                'indi_log_std': indi_log_std}
        
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
      
      
class VAEbase_one_static(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.staticEncoder = staticEncoderBase(**model_config['static_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
        
    def forward(self, x, lenx, indi_ref):
        
        indi_mu, indi_log_std = self.staticEncoder(indi_ref) # [B, individual embedding]
        z_indi = self.staticEncoder.sample(indi_mu, indi_log_std)
        
        # print('z_indi',z_indi.shape)
        z_indi = z_indi.unsqueeze(2).expand(-1, -1, 256)
        
        self.lenx = torch.full((z_indi.size(0),), 256, dtype=torch.int32).to(self.device)
        x_rec = self.decoder(z_indi, self.lenx)
        
        return {'x_rec': x_rec, 'con_mu': z_indi, 'con_log_std': indi_log_std,
                'z_con': z_indi, 'z_indi':z_indi, 'indi_mu': indi_mu, 
                'indi_log_std': indi_log_std, 'indi_ref': indi_ref}
        
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        indi_ref = outputs['indi_ref']
        # con_mu = outputs['con_mu']
        # con_log_std = outputs['con_log_std']
        indi_mu = outputs['indi_mu']
        indi_log_std = outputs['indi_log_std']
        
        indi_kl = self.staticEncoder.kl_divergence(indi_mu, indi_log_std)
        
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, indi_ref, self.lenx) + 0.5 * self.dec_normal_nll(x_rec, indi_ref, self.lenx)
        
        return nll, indi_kl, indi_kl
        
        
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
        
class VAEbase_one(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
    def forward(self, x, lenx, indi_ref):
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, content embedding, T]
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        
        x_rec = self.decoder(z_con, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi': z_con, 'indi_mu': con_mu, 
                'indi_log_std': con_log_std}
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']

        con_kl = self.dynamicEncoder.kl_divergence(con_mu, con_log_std, lens)
        
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, mel, lens) + 0.5 * self.dec_normal_nll(x_rec, mel, lens)

        return nll, con_kl, con_kl
        
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
        

class VAEbase_LSTM(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.LocalEncoder = EncoderBase_LSTM(**model_config['static_encoder'])
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
    def forward(self, x, lenx, indi_ref):
        # indi embedding
        indi_out= self.LocalEncoder(x) # [B, individual embedding]
        indi_mu = indi_out['f_mean']
        indi_log_std = indi_out['f_logvar']
        z_indi = indi_out['f']
        
        # content embedding
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, content embedding, T]
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        
        time_len = z_con.size(2)
        z_indi_in = z_indi.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'indi_mu': indi_mu, 
                'indi_log_std': indi_log_std}
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
        
        indi_mu = outputs['indi_mu']
        indi_log_std = outputs['indi_log_std']
         
        indi_kl = self.LocalEncoder.kl_divergence(indi_mu, indi_log_std)
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

class VectorVAEbase(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.staticEncoder = staticEncoderBase_Vector(**model_config['static_encoder'])
        self.vector_quantization = VectorQuantizerEMA(device=self.device, **model_config['VectorQuantizer'])
        
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
    def forward(self, x, lenx, indi_ref):
         # con
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, content embedding, T]
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        # indi    
        z_indi = self.staticEncoder(indi_ref) # [B, individual embedding]        
        embedding_loss, z_q, perplexity, encodings = self.vector_quantization(z_indi)
       
        time_len = z_con.size(2)
        z_indi_in = z_q.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'z_q': z_q, 'embedding_loss': embedding_loss, 'perplexity': perplexity }
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
        
        indi_loss = outputs['embedding_loss']
         
        con_kl = self.dynamicEncoder.kl_divergence(con_mu, con_log_std, lens)
        
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, mel, lens) + 0.5 * self.dec_normal_nll(x_rec, mel, lens)

        return nll, indi_loss, con_kl
        
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

class VectorVAEbase_dy(nn.Module):
    '''
    not finish
    '''
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.staticEncoder = staticEncoderBase(**model_config['static_encoder'])
        
        self.dynamicEncoder = dynamicEncoderBase_Vector(**model_config['dynamic_encoder'])
        self.vector_quantization = VectorQuantizerEMA(device=self.device, **model_config['VectorQuantizer'])
        
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
    def forward(self, x, lenx, indi_ref):
        # indi   
        indi_mu, indi_log_std = self.staticEncoder(indi_ref) # [B, individual embedding]
        z_indi = self.staticEncoder.sample(indi_mu, indi_log_std)
         
        # con
        z_con = self.dynamicEncoder(x, lenx) # [B, content embedding, T]
        embedding_loss, z_q, perplexity, encodings = self.vector_quantization(z_con)
       
        time_len = z_con.size(2)
        z_indi_in = z_q.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'z_q': z_q, 'embedding_loss': embedding_loss, 'perplexity': perplexity }
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
        
        indi_loss = outputs['embedding_loss']
         
        con_kl = self.dynamicEncoder.kl_divergence(con_mu, con_log_std, lens)
        
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, mel, lens) + 0.5 * self.dec_normal_nll(x_rec, mel, lens)

        return nll, indi_loss, con_kl
        
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

class VectorVAEbase2(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.staticEncoder = staticEncoderBase(**model_config['static_encoder'])
        self.vector_quantization = VectorQuantizer(**model_config['VectorQuantizer'], device=self.device)
        
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        self.out_channels = model_config['static_encoder']['out_channels']
        
    def forward(self, x, lenx, indi_ref):
        indi_mu, indi_log_std = self.staticEncoder(indi_ref) # [B, individual embedding]
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, content embedding, T]
        
        # indi & vq    
        z_indi = torch.cat([indi_mu, indi_log_std], dim=1)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_indi)
        
        zq_mu, zq_logs = torch.split(z_q, self.out_channels, dim=1)
        zq_indi = self.staticEncoder.sample(zq_mu, zq_logs)
        
        # con
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        
        time_len = z_con.size(2)
        z_indi_in = zq_indi.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'z_q': zq_indi, 'zq_mu': zq_mu, 'zq_logs': zq_logs,
                'embedding_loss': embedding_loss, 'perplexity': perplexity }
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
        
        indi_loss = outputs['embedding_loss']
        zq_mu = outputs['zq_mu']
        zq_log_std = outputs['zq_logs']
        
        indi_kl = self.staticEncoder.kl_divergence(zq_mu, zq_log_std)
        con_kl = self.dynamicEncoder.kl_divergence(con_mu, con_log_std, lens)
        
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, mel, lens) + 0.5 * self.dec_normal_nll(x_rec, mel, lens)

        return nll, indi_loss, indi_kl, con_kl
        
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

class DSVAEbase(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        # load model config
        self.indi_staticEncoder = staticEncoderBase(**model_config['static_encoder'])
        self.indi_dynamicEncoder = indi_dynamicEncoderBase(**model_config['indi_dynamic_encoder'])
        
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
    def forward(self, x, lenx, indi_ref):
        # indi
        indi_st_mu, indi_st_log_std = self.indi_staticEncoder(indi_ref) # [B, individual embedding], out_channels
        z_indi_st = self.indi_staticEncoder.sample(indi_st_mu, indi_st_log_std) # [B, individual embedding]
        
        indi_dy_mu, indi_dy_log_std = self.indi_dynamicEncoder(x, lenx) # [B, individual embedding]
        z_indi_dy = self.indi_dynamicEncoder.sample(indi_dy_mu, indi_dy_log_std)
        
        # content
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, content embedding, T], out_channels
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std) # [B, 80, time length]
        
        time_len = z_con.size(2)
        z_indi_st_in = z_indi_st.unsqueeze(2).expand(-1, -1, time_len) # [B, individual embedding]->[B, individual embedding, time length]
        z_indi_dy_in = z_indi_dy.unsqueeze(2).expand(-1, -1, time_len)
        
        
        dec_in = torch.cat([z_indi_st_in, z_indi_dy_in, z_con], dim=1) # [B, individual embedding * 3, time length]
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 
                'con_mu': con_mu, 'con_log_std': con_log_std, 'z_con': z_con, 
                'z_indi_st':z_indi_st, 'indi_st_mu': indi_st_mu, 'indi_st_log_std': indi_st_log_std,
                'z_indi_dy':z_indi_dy, 'indi_dy_mu': indi_dy_mu, 'indi_dy_log_std': indi_dy_log_std
                }
        
    def loss_fn(self, outputs, mel, lens):
        x_rec = outputs['x_rec']
        
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
        
        indi_st_mu = outputs['indi_st_mu']
        indi_st_log_std = outputs['indi_st_log_std']
        
        indi_dy_mu = outputs['indi_dy_mu']
        indi_dy_log_std = outputs['indi_dy_log_std']
        
        # indi kl
        indi_st_kl = self.indi_staticEncoder.kl_divergence(indi_st_mu, indi_st_log_std)
        indi_dy_kl = self.indi_dynamicEncoder.kl_divergence(indi_dy_mu, indi_dy_log_std)
        # con kl
        con_kl = self.dynamicEncoder.kl_divergence(con_mu, con_log_std, lens)
        
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, mel, lens) + 0.5 * self.dec_normal_nll(x_rec, mel, lens)

        return nll, indi_st_kl, indi_dy_kl, con_kl
        
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
        

class JOINTVAEbase(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.use_cuda = True
        
        self.temperature = 0.67
        
        # load model config
        # static Encoder
        self.staticEncoder = staticEncoderJAVE(**model_config['static_encoder'])
        
        # dynamic Encoder        
        self.dynamicEncoder = dynamicEncoderBase(**model_config['dynamic_encoder'])
        
        # decoder
        self.decoder = ConvSADecoder(**model_config['decoder'])
        
        # discrete
        self.latent_spec = model_config['latent_spec']['param']
        self.is_continuous = model_config['latent_spec']['is_continuous']
        self.is_discrete = model_config['latent_spec']['is_discrete']
        self.latent_hidden_dim = model_config['latent_spec']['hidden_dim']
        
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        
        self.latent_cont_dim = self.latent_spec['cont']
        self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        # Encode parameters of latent distribution
        if self.is_continuous:
            self.fc_mean = nn.Linear(self.latent_hidden_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self.latent_hidden_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(self.latent_hidden_dim, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)
        
        # continuous out 
  
        
    def forward(self, x, lenx, indi_ref):
        # stactic encoder
        sta_emb = self.staticEncoder(indi_ref) # [B, individual embedding]
        latent_dist = {}
        if self.is_continuous:
            latent_dist['cont'] = [self.fc_mean(sta_emb), self.fc_log_var(sta_emb)]  
        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.fc_alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(sta_emb), dim=1))
        # reparameterize for stactic encoder
        z_indi = self.reparameterize(latent_dist)
        
        # dynamic encoder
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, content embedding, T]
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        
        # change shape
        time_len = z_con.size(2)
        z_indi_in = z_indi.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'latent_dist': latent_dist}
        
    def loss_fn(self, outputs, mel, lens, latent_dist):
        '''
        outputs: output of model
        mel: mel spectrogram
        lens: length of mel spectrogram
        latent_dist: cont and disc
        '''
        x_rec = outputs['x_rec']
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
         
        indi_continuous_loss = 0.
        indi_discrete_loss = 0.
        
        if self.is_continuous:
            # Calculate KL divergence
            mean, logvar = latent_dist['cont']
            kl_cont_loss = self._kl_normal_loss(mean, logvar)
            # Calculate continuous capacity loss
            indi_continuous_loss = kl_cont_loss
            
        if self.is_discrete:
            # Calculate KL divergence
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
            # Linearly increase capacity of discrete channels
            indi_discrete_loss = kl_disc_loss
        
        # indi_kl 
        indi_kl = indi_continuous_loss + indi_discrete_loss
        # content_kl
        con_kl = self.dynamicEncoder.kl_divergence(con_mu, con_log_std, lens)
        # x_rec: rec, mel: ground truth
        nll = 0.5 * self.dec_laplace_nll(x_rec, mel, lens) + 0.5 * self.dec_normal_nll(x_rec, mel, lens)
        
        return {'nll':nll, 'indi_kl':indi_kl, 'con_kl': con_kl, 'indi_continuous_loss': indi_continuous_loss, 'indi_discrete_loss': indi_discrete_loss} 
        
    
    def reparameterize(self, latent_dist):
        """
        Samples from latent distribution using the reparameterization trick.
        
        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []

        if self.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_sample = self.sample_normal(mean, logvar)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)

        # Concatenate continuous and discrete samples into one large sample
        out = torch.cat(latent_sample, dim=1)
        return out
     
     
    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if self.use_cuda:
                eps = eps.to(self.device)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean


    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.to(self.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            if self.use_cuda:
                one_hot_samples = one_hot_samples.to(self.device)
            return one_hot_samples
        
    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)


        return kl_loss

    def _kl_multiple_discrete_loss(self, alphas):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]

        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))

        # Record losses
        # if self.model.training and self.num_steps % self.record_loss_every == 1:
        #     self.losses['kl_loss_disc'].append(kl_loss.item())
        #     for i in range(len(alphas)):
        #         self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        if self.use_cuda:
            log_dim = log_dim.to(self.device)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        
        return kl_loss


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
        
class DisentangledVAE1D(nn.Module):
    def __init__(self, f_dim=256, z_dim=32, conv_dim=2048, step=256, in_size=64, hidden_dim=512,
                 mel_dim=80, nonlinearity=None, factorised=False, shuffle_input = False, module_config = False, device=torch.device('cpu')):
    # def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.f_dim = f_dim
        self.z_dim = z_dim
        self.conv_dim = conv_dim
        self.step = step
        self.in_size = in_size
        self.hidden_dim = hidden_dim
        self.mel_dim = mel_dim
        self.factorised = factorised
        self.shuffle_input = shuffle_input
                
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity

        # Prior of content is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        
        # POSTERIOR DISTRIBUTION NETWORKS
        # -------------------------------
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        # TODO: Check if only one affine transform is sufficient. Paper says distribution is parameterised by LSTM
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        if self.factorised is True:
            # Paper says : 1 Hidden Layer MLP. Last layers shouldn't have any nonlinearities
            self.z_inter = LinearUnit(self.conv_dim, self.hidden_dim, batchnorm=False)
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        else:
            # TODO: Check if one affine transform is sufficient. Paper says distribution is parameterised by RNN over LSTM. Last layer shouldn't have any nonlinearities
            self.z_lstm = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
            self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
            # Each timestep is for each z so no reshaping and feature mixing
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # 1D in_channels, out_channels, kernel, stride=1, padding=0
        self.conv = nn.Sequential(
                ConvUnit1D(1 , step, 5, 1, 2), # 
                ConvUnit1D(step, step, 5, 2, 2), 
                ConvUnit1D(step, step, 5, 2, 2), 
                ConvUnit1D(step, step, 5, 2, 2), 
                )
        self.final_conv_size = 80 // 8
        self.conv_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))
        if self.shuffle_input is True:
            self.staticEncoder = staticEncoderBase(**module_config['static_encoder'])

        self.deconv_fc = nn.Sequential(LinearUnit(self.f_dim + self.z_dim, self.conv_dim * 2, False),
                LinearUnit(self.conv_dim * 2, step * (self.final_conv_size), False))
        self.deconv = nn.Sequential(
                ConvUnitTranspose1D(step, step, 5, 2, 2, 1),
                ConvUnitTranspose1D(step, step, 5, 2, 2, 1),
                ConvUnitTranspose1D(step, step, 5, 2, 2, 1),
                ConvUnitTranspose1D(step, 1, 5, 1, 2, 0))

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, batch_size, frames, random_sampling=True):
        z_out = None # This will ultimately store all z_s in the format [batch_size, frames, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_mean_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        z_logvar_t = torch.zeros(batch_size, self.z_dim, device=self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for _ in range(frames):
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)

        return z_means, z_logvars, z_out


    #1D 
    def encode_frames(self, x):
        # The frames are unrolled into the batch dimension for batch processing such that x goes from
        # [batch_size, size, frames] -> [batch_size * frames, 1, size]
        frames = x.size(2)       
        
        x = x.transpose(1, 2)
        x = x.unsqueeze(-2)
        x = x.reshape(-1, x.size(-2), x.size(-1))
        
        x = self.conv(x)
        x = x.reshape(-1, self.step * self.final_conv_size) # 256 * 10 卷积的输出
        x = self.conv_fc(x)
        # The frame dimension is reintroduced and x shape becomes [batch_size, frames, conv_dim]
        # This technique is repeated at several points in the code
        x = x.reshape(-1, frames, self.conv_dim)
        return x
    
    
    # 1D
    def decode_frames(self, zf):
        frames = zf.size(1)
        x = self.deconv_fc(zf)
        x = x.reshape(-1, self.step, self.final_conv_size)
        x = self.deconv(x)
        
        # to original mel [batch, mel_dim, frames]
        
        x = x.reshape(-1, frames, self.mel_dim)
        x = x.transpose(1,2)
        return x

    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)


    def encode_z(self, x, f):
        if self.factorised is True:
            features = self.z_inter(x)
        else:
            # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
            f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
            lstm_out, _ = self.z_lstm(torch.cat((x, f_expand), dim=2))
            features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)


    def forward(self, x, indi_mel = None):
        self.frames = x.size(2)
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), self.frames, random_sampling=self.training)
        conv_x = self.encode_frames(x) # conv_x: [B, time_len, dim of self.step * self.final_conv_size]
        
        # f: [B, f dim]
        if indi_mel is not None and self.shuffle_input  is True:
            f_mean, f_logvar  = self.staticEncoder(indi_mel)  
            f = self.staticEncoder.sample(f_mean, f_logvar)
            # print('indi_mel',indi_mel.shape)
            # print('f_mean',f_mean.shape)
            # print('f',f.shape)
        else:
            f_mean, f_logvar, f = self.encode_f(conv_x)
            
        # conv
        z_mean, z_logvar, z = self.encode_z(conv_x, f) 
        # print('conv_x',conv_x.shape)    
        # print('z',z.shape)
        
        f_expand = f.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z, f_expand), dim=2)
        recon_x = self.decode_frames(zf)
        
        return {'recon_x': recon_x, 'f_mean': f_mean, 'f_logvar': f_logvar, 'f': f, 'z_post_mean':z_mean, 
                'z_post_logvar': z_logvar, 'z': z, 'z_prior_mean': z_mean_prior, 
                'z_prior_logvar': z_logvar_prior}


    def loss_fn(self, original_seq, recon_seq, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar,hyperparameters=None):
        """
        Loss function consists of 3 parts, the reconstruction term that is the MSE loss between the generated and the original images
        the KL divergence of f, and the sum over the KL divergence of each z_t, with the sum divided by batch_size

        Loss = {mse + KL of f + sum(KL of z_t)} / batch_size
        Prior of f is a spherical zero mean unit variance Gaussian and the prior of each z_t is a Gaussian whose mean and variance
        are given by the LSTM
        """
        batch_size = original_seq.size(0)
        
        mse = F.mse_loss(recon_seq,original_seq,reduction='sum')
        
        kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
        z_post_var = torch.exp(z_post_logvar)
        z_prior_var = torch.exp(z_prior_logvar)
        kld_z = 0.5 * torch.sum(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
        
        if hyperparameters is not None:
            con_gamma = hyperparameters['con_gamma']
            indi_gamma = hyperparameters['indi_gamma']
            
            loss = (mse + indi_gamma * kld_f.abs() + con_gamma * kld_z.abs())/batch_size
        else:
            loss = (mse + kld_f + kld_z)/batch_size
        
        return loss, kld_f/batch_size, kld_z/batch_size


class EncoderBase_LSTM(nn.Module):
    def __init__(self, f_dim=256, z_dim=32, conv_dim=2048, step=256, in_size=64, hidden_dim=512,
                 mel_dim=80, nonlinearity=None, factorised=False, shuffle_input = False, module_config = False, device=torch.device('cpu')):
        super().__init__()
        self.f_dim = f_dim
        self.step = step
        self.hidden_dim = hidden_dim
        self.conv_dim = conv_dim

        # 1D in_channels, out_channels, kernel, stride=1, padding=0
        self.conv = nn.Sequential(
                ConvUnit1D(1 , step, 5, 1, 2), # 
                ConvUnit1D(step, step, 5, 2, 2), 
                ConvUnit1D(step, step, 5, 2, 2), 
                ConvUnit1D(step, step, 5, 2, 2), 
                )
        
        self.final_conv_size = 80 // 8
        self.conv_fc = nn.Sequential(LinearUnit(step * (self.final_conv_size), self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))
        self.f_lstm = nn.LSTM(self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        
        
    def forward(self, x, lens=None):
        self.frames = x.size(2)
        conv_x = self.encode_frames(x)  # [B, len, conv_dim (256*10)/2]
        f_mean, f_logvar, f = self.encode_f(conv_x)
        
        return {'f_mean': f_mean, 'f_logvar': f_logvar, 'f': f}
        

    def encode_frames(self, x):
        # The frames are unrolled into the batch dimension for batch processing such that x goes from
        # [batch_size, size, frames] -> [batch_size * frames, 1, size]
        frames = x.size(2)       
        
        x = x.transpose(1, 2)
        x = x.unsqueeze(-2)
        x = x.reshape(-1, x.size(-2), x.size(-1)) # [len, B, mel_dim]
        
        x = self.conv(x) # [len, step, mel_dim//8] [len, 256, 10]
        x = x.reshape(-1, self.step * self.final_conv_size) # 256 * 10 [len, 256*10]
        x = self.conv_fc(x) # [len, (256*10)/2]
        # The frame dimension is reintroduced and x shape becomes [batch_size, frames, conv_dim]
        # This technique is repeated at several points in the code
        x = x.reshape(-1, frames, self.conv_dim) # [B, len, (256*10)/2]
        
        return x
    
    
    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x) # [B, len, conv_dim (256*10)/2]
        # The features of the last timestep of the forward RNN is stored at the end of lstm_out in the first half, and the features
        # of the "first timestep" of the backward RNN is stored at the beginning of lstm_out in the second half
        # For a detailed explanation, check: https://gist.github.com/ceshine/bed2dadca48fe4fe4b4600ccce2fd6e1
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim] # [B, hidden_dim] 
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim] # [B, hidden_dim 512] 
        
        lstm_out = torch.cat((frontal, backward), dim=1) # [B, hidden_dim*2] 
        mean = self.f_mean(lstm_out) # [B, f_dim: 128]
        logvar = self.f_logvar(lstm_out) # [B, f_dim: 128]
        
        return mean, logvar, self.reparameterize(mean, logvar, self.training)
        
        
    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean
        
        
    @staticmethod
    def kl_divergence(mu, log_std, lens=None):
        """
        :param mu: [B, C]
        :param log_std: [B, C]
        :return:
        """
        post = D.Normal(mu, torch.exp(log_std))
        prior = D.Normal(torch.zeros_like(mu, requires_grad=False),
                         torch.ones_like(log_std, requires_grad=False))
        kl = D.kl.kl_divergence(post, prior)
        kl = torch.mean(torch.sum(kl, dim=1))
        
        # batch_size = mu.size(0)
        # kld_f = (-0.5 * torch.sum(1 + log_std - torch.pow(mu,2) - torch.exp(log_std))/batch_size)

        return kl


    
# test
# device = torch.device("cpu")
# vae = DisentangledVAE1D(f_dim=256, z_dim=32, step=256, factorised=True, device=device)
# for name, module in vae.named_modules():
#     print(name, module)
