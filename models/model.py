import torch
import numpy as np
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from modules import staticEncoderBase, dynamicEncoderBase, ConvSADecoder,\
    sequence_mask, LinearUnit, ConvUnit1D, ConvUnitTranspose1D


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
        con_mu, con_log_std = self.dynamicEncoder(x, lenx) # [B, individual embedding, T]
        
        z_indi = self.staticEncoder.sample(indi_mu, indi_log_std)
        z_con = self.dynamicEncoder.sample(con_mu, con_log_std)
        
        time_len = z_con.size(2)
        z_indi_in = z_indi.unsqueeze(2).expand(-1, -1, time_len)
        
        dec_in = torch.cat([z_indi_in, z_con], dim=1)
        x_rec = self.decoder(dec_in, lenx)
        
        return {'x_rec': x_rec, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'z_indi':z_indi, 'indi_mu': indi_mu, 
                'indi_log_std': indi_log_std, }
        
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


    def loss_fn(self, original_seq, recon_seq, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
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
        
        return (mse + kld_f + kld_z)/batch_size, kld_f/batch_size, kld_z/batch_size


# test
# device = torch.device("cpu")
# vae = DisentangledVAE1D(f_dim=256, z_dim=32, step=256, factorised=True, device=device)
# for name, module in vae.named_modules():
#     print(name, module)
