
import torch
import torch.nn as nn
import numpy as np

class MultiScaleFourierFeatureEncoding(nn.Module):
    def __init__(self, input_dim, num_features, scales):
        super().__init__()
        self.num_features = num_features
        self.scales = scales
        self.out_dim = 2 * num_features * len(scales)

        self.B_list = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, num_features) * scale, requires_grad=False)
            for scale in scales
        ])

    def forward(self, x):
        features = []
        for B in self.B_list:
            proj = (2.0 * np.pi * x) @ B
            features.append(torch.cos(proj))
            features.append(torch.sin(proj))
        return torch.cat(features, dim=-1)

class PI_DeepONet_Robust(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Extraction depuis la config
        self.k_phys = cfg.physics['k']
        b = cfg.physics['bounds']
        scales = cfg.model['fourier_scales']
        latent_dim = cfg.model['latent_dim']

        # Buffers Normalisation
        self.register_buffer('A_min', torch.tensor(b['A'][0]))
        self.register_buffer('A_max', torch.tensor(b['A'][1]))
        self.register_buffer('w0_min', torch.tensor(b['w0'][0]))
        self.register_buffer('w0_max', torch.tensor(b['w0'][1]))
        self.register_buffer('f_min', torch.tensor(b['f'][0]))
        self.register_buffer('f_max', torch.tensor(b['f'][1]))
        self.register_buffer('r_max', torch.tensor(cfg.physics['r_max']))
        self.register_buffer('z_max', torch.tensor(cfg.physics['z_max']))

        self.register_buffer('epsilon_curr', torch.tensor(cfg.model['epsilon_curriculum']))

        # Branch Net
        self.branch_net = nn.Sequential(
            nn.Linear(cfg.model['branch_dim'], 200), nn.SiLU(),
            nn.Linear(200, 200), nn.SiLU(),
            nn.Linear(200, 200), nn.SiLU(),
            nn.Linear(200, 200), nn.SiLU(),
            nn.Linear(200, 2 * latent_dim)
        )

        # Trunk Net
        self.trunk_encoding = MultiScaleFourierFeatureEncoding(cfg.model['trunk_dim'], 64, scales)
        self.trunk_net = nn.Sequential(
            nn.Linear(self.trunk_encoding.out_dim, 200), nn.SiLU(),
            nn.Linear(200, 200), nn.SiLU(),
            nn.Linear(200, 200), nn.SiLU(),
            nn.Linear(200, 200), nn.SiLU(),
            nn.Linear(200, latent_dim)
        )
        self.latent_dim = latent_dim

    def normalize_linear(self, x, x_min, x_max):
        return 2.0 * (x - x_min) / (x_max - x_min + 1e-9) - 1.0

    def normalize_log(self, x, x_min, x_max):
        return self.normalize_linear(torch.log10(torch.abs(x)+1e-9), 
                                     torch.log10(torch.abs(x_min)+1e-9), 
                                     torch.log10(torch.abs(x_max)+1e-9))

    def forward(self, params, coords):
        f_raw = params[:, 2:3]
        r_raw = coords[:, 0:1]
        z_raw = coords[:, 1:2]

        params_norm = torch.cat([
            self.normalize_linear(params[:,0:1], self.A_min, self.A_max),
            self.normalize_log(params[:,1:2], self.w0_min, self.w0_max),
            self.normalize_log(params[:,2:3], self.f_min, self.f_max)
        ], dim=1)

        coords_norm = torch.cat([
            self.normalize_linear(coords[:,0:1], 0.0, self.r_max),
            self.normalize_linear(coords[:,1:2], 0.0, self.z_max)
        ], dim=1)

        B = self.branch_net(params_norm)
        T = self.trunk_net(self.trunk_encoding(coords_norm))
        B_re, B_im = torch.split(B, self.latent_dim, dim=1)

        env_re = torch.sum(B_re * T, dim=1, keepdim=True)
        env_im = torch.sum(B_im * T, dim=1, keepdim=True)

        phase_lens = - (self.k_phys * r_raw**2) / (2.0 * f_raw)
        cos_p = torch.cos(phase_lens)
        sin_p = torch.sin(phase_lens)

        geom_scale = 1.0 / torch.sqrt( (1.0 - z_raw/f_raw)**2 + self.epsilon_curr )

        out_re = geom_scale * (env_re * cos_p - env_im * sin_p)
        out_im = geom_scale * (env_re * sin_p + env_im * cos_p)

        return out_re, out_im

    def set_epsilon(self, new_eps):
        self.epsilon_curr.fill_(new_eps)
