import torch
import torch.nn as nn
import torch.distributions as D
from transformer import get_sinusoid_encoding_table, FFTBlock
from .utils import sequence_mask, ConvResLNBlk, ConvResBNBlk,\
    get_activation, ConvINBlk, ConvBNBlk, masked_instance_norm, MaskedBatchNorm1d


class dynamicEncoderBase(nn.Module):
    def __init__(self, in_channels, h_channels, out_channels, conv_kernels, paddings, 
                 activation, sa_hidden, sa_layer, sa_head, sa_filter_size, sa_kernel_size,
                 sa_dropout, max_seq_len):
        super().__init__()
        # convolutions
        self.out_channels = out_channels
        self.activation = get_activation(activation)
        self.conv_initial = nn.Conv1d(in_channels, h_channels, kernel_size=1, padding=0)
        
        self.conv_layers = nn.ModuleList([
            ConvBNBlk(h_channels, 
                      h_channels, 
                      kernel_size=k, 
                      padding=p, 
                      activation=self.activation)
            for k, p in zip(conv_kernels, paddings)])
        
        self.out_batch_norm = MaskedBatchNorm1d(h_channels)
        
        # self-attentions
        n_position = max_seq_len + 1
        d_k = d_v = sa_hidden // sa_head
        self.max_seq_len = max_seq_len
        self.d_model = sa_hidden
        
        # position code
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, sa_hidden).unsqueeze(0),
            requires_grad=False)
        
        self.layer_stack = nn.ModuleList(
            [FFTBlock(
                sa_hidden, 
                sa_head, 
                d_k, 
                d_v, 
                sa_filter_size,
                sa_kernel_size, 
                dropout=sa_dropout)
                for _ in range(sa_layer)])
        
        self.mu_logs_linear = nn.Conv1d(sa_hidden, 2 * out_channels, kernel_size=1, padding=0)
        
    def forward(self, x, lens=None):
        """
        :param x: [B, C, T]
        :param lens: [B]
        :return: mu and log_std of shape [B, C, T]
        """
        if lens is not None:
            max_len = x.size(2)
            mask = ~sequence_mask(lens, max_length=max_len)
        else:
            B, T = x.size(0), x.size(2)
            mask = torch.zeros([B, T], dtype=torch.bool, device=x.device)
            
        # convolution
        h = self.conv_initial(x)
        for conv in self.conv_layers:
            h = conv(h, lens)
        h = self.activation(h)
        h = self.out_batch_norm(h, lens)
        
        # self-attention
        sa_input = h.permute(0, 2, 1)  # [B, C, T]->[B, T, C]
        batch_size, max_len = sa_input.shape[0], sa_input.shape[1]

        # -- Forward
        if not self.training and sa_input.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = sa_input + get_sinusoid_encoding_table(
                sa_input.shape[1], self.d_model
            )[: sa_input.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                sa_input.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = sa_input[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask)
        dec_output = dec_output.permute(0, 2, 1).contiguous()

        # get distribution parameters
        mu_logs = self.mu_logs_linear(dec_output)
        mu, log_std = torch.split(mu_logs, self.out_channels, dim=1)
        
        return mu, log_std  
    
    @staticmethod
    def sample(mu, log_std, std=1.):
        """
        :param mu: [B, C, T]
        :param log_std: [B, C, T]
        :param std: sampling std, positive float
        :return: z: [B, C, T]
        """
        assert std > 0.
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(log_std) * std)
        z = eps * torch.exp(log_std) + mu
        return z
    
    @staticmethod
    def kl_divergence(mu, log_std, lens=None):
        """
        :param mu: [B, C, T]
        :param log_std: [B, C, T]
        :param lens: [B]
        :return:
        """
        if lens is not None:
            mask = sequence_mask(lens).to(mu.dtype)
        else:
            B = mu.size(0)
            T = mu.size(2)
            mask = torch.ones([B, T], dtype=mu.dtype, device=mu.device)
        post = D.Normal(mu, torch.exp(log_std))
        prior = D.Normal(
            torch.zeros_like(mu, requires_grad=False),
            torch.ones_like(log_std, requires_grad=False))
        kl = D.kl.kl_divergence(post, prior)
        kl = torch.sum(torch.sum(kl, dim=1) * mask, dim=1) / torch.sum(mask, dim=1)
        kl = torch.mean(kl)
        return kl