import torch
import torch.nn as nn
import numpy as np

class MultiScaleFourierFeatureEncoding(nn.Module):
    """Encode les coordonnées (x, t) en hautes fréquences."""
    def __init__(self, in_dim, num_features, scales):
        super().__init__()
        self.in_dim = in_dim
        self.num_features = num_features
        self.scales = torch.tensor(scales).float()
        
        # Création des fréquences (B)
        # On projette in_dim vers num_features
        B = torch.randn(num_features, in_dim) * self.scales.view(-1, 1).mean()
        self.register_buffer('B', B)
        self.out_dim = num_features * 2 # Sin + Cos

    def forward(self, x):
        # x: [batch, in_dim]
        # x @ B.t() -> [batch, num_features]
        proj = 2.0 * np.pi * x @ self.B.t()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class CGL_PI_DeepONet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. Extraction Config
        b = cfg.physics['bounds']
        x_min, x_max = cfg.physics['x_domain']
        t_max = cfg.physics['t_max']
        
        latent_dim = cfg.model['latent_dim']
        branch_arch = cfg.model.get('branch_layers', [256, 256, 256, 256])
        trunk_arch = cfg.model.get('trunk_layers', [256, 256, 256, 256])
        
        # Fourier Features
        fourier_dim = cfg.model.get('fourier_dim', 64)
        scales = cfg.model.get('fourier_scales', [1.0, 10.0])

        # 2. Buffers de Normalisation (Branch Inputs)
        # On normalise tout ce qui rentre dans le Branch Net
        self.register_buffer('A_min', torch.tensor(b['A'][0]))
        self.register_buffer('A_max', torch.tensor(b['A'][1]))
        
        self.register_buffer('w0_min', torch.tensor(b['w0'][0]))
        self.register_buffer('w0_max', torch.tensor(b['w0'][1]))
        
        self.register_buffer('x0_min', torch.tensor(b['x0'][0]))
        self.register_buffer('x0_max', torch.tensor(b['x0'][1]))
        
        self.register_buffer('k_min', torch.tensor(b['k'][0]))
        self.register_buffer('k_max', torch.tensor(b['k'][1]))

        # Buffers Normalisation (Trunk Inputs)
        self.register_buffer('x_min', torch.tensor(x_min))
        self.register_buffer('x_max', torch.tensor(x_max))
        self.register_buffer('t_max', torch.tensor(t_max))

        # 3. BRANCH NET
        # Entrées : [A, w0, x0, k, type] -> 5 paramètres
        self.branch_input_dim = 5 
        
        layers_b = []
        prev_dim = self.branch_input_dim
        for h_dim in branch_arch:
            layers_b.append(nn.Linear(prev_dim, h_dim))
            layers_b.append(nn.SiLU()) 
            prev_dim = h_dim
        layers_b.append(nn.Linear(prev_dim, 2 * latent_dim)) # x2 pour Réel/Imaginaire
        self.branch_net = nn.Sequential(*layers_b)

        # 4. TRUNK NET
        # Entrées : [x, t] -> 2 coordonnées
        self.trunk_input_dim = 2
        
        self.trunk_encoding = MultiScaleFourierFeatureEncoding(self.trunk_input_dim, fourier_dim, scales)
        
        layers_t = []
        prev_dim = self.trunk_encoding.out_dim 
        for h_dim in trunk_arch:
            layers_t.append(nn.Linear(prev_dim, h_dim))
            layers_t.append(nn.SiLU())
            prev_dim = h_dim
        layers_t.append(nn.Linear(prev_dim, latent_dim))
        self.trunk_net = nn.Sequential(*layers_t)
        
        self.latent_dim = latent_dim

    def normalize_linear(self, x, x_min, x_max):
        return 2.0 * (x - x_min) / (x_max - x_min + 1e-9) - 1.0

    def normalize_log(self, x, x_min, x_max):
        # Pour w0 qui est en log-scale
        return self.normalize_linear(torch.log10(torch.abs(x) + 1e-9), 
                                     torch.log10(torch.abs(x_min) + 1e-9), 
                                     torch.log10(torch.abs(x_max) + 1e-9))

    def forward(self, params, coords):
        # params: [A, w0, x0, k, type] (Batch x 5)
        # coords: [x, t]               (Batch x 2)

        # 1. Normalisation Branch
        # A (Linéaire)
        norm_A = self.normalize_linear(params[:, 0:1], self.A_min, self.A_max)
        # w0 (Log)
        norm_w0 = self.normalize_log(params[:, 1:2], self.w0_min, self.w0_max)
        # x0 (Linéaire)
        norm_x0 = self.normalize_linear(params[:, 2:3], self.x0_min, self.x0_max)
        # k (Linéaire)
        norm_k = self.normalize_linear(params[:, 3:4], self.k_min, self.k_max)
        # Type (Déjà discret 0,1,2 -> on le laisse tel quel ou on normalise grossièrement)
        # Le mieux est de le normaliser entre -1 et 1 sachant qu'on a 3 types (0, 1, 2)
        norm_type = self.normalize_linear(params[:, 4:5], 0.0, 2.0)

        params_norm = torch.cat([norm_A, norm_w0, norm_x0, norm_k, norm_type], dim=1)

        # 2. Normalisation Trunk
        x_raw = coords[:, 0:1]
        t_raw = coords[:, 1:2]
        
        coords_norm = torch.cat([
            self.normalize_linear(x_raw, self.x_min, self.x_max),
            # t est normalisé entre [0, 1] pour l'instant (car t part de 0), 
            # mais [-1, 1] est mieux pour le réseau.
            self.normalize_linear(t_raw, 0.0, self.t_max) 
        ], dim=1)

        # 3. Forward Pass
        B = self.branch_net(params_norm)               # [Batch, 2*Latent]
        T = self.trunk_net(self.trunk_encoding(coords_norm)) # [Batch, Latent]
        
        # 4. Dot Product
        B_re, B_im = torch.split(B, self.latent_dim, dim=1)

        # u_re = Sum(B_re * T)
        # u_im = Sum(B_im * T)
        out_re = torch.sum(B_re * T, dim=1, keepdim=True)
        out_im = torch.sum(B_im * T, dim=1, keepdim=True)

        return out_re, out_im