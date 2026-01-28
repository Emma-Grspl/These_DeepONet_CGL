import torch
import torch.nn as nn
import numpy as np

class MultiScaleFourierFeatureEncoding(nn.Module):
    """Encode les coordonnées (r, z) en hautes fréquences."""
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
        proj = 2.0 * np.pi * x @ self.B.t()
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class PI_DeepONet_Robust(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # 1. Extraction Physique & Config
        self.k_phys = cfg.physics['k']
        b = cfg.physics['bounds']
        latent_dim = cfg.model['latent_dim']
        
        # 2. Paramètres Dynamiques (avec valeurs par défaut si absents du YAML)
        branch_arch = cfg.model.get('branch_layers', [256, 256, 256, 256])
        trunk_arch = cfg.model.get('trunk_layers', [256, 256, 256, 256])
        fourier_dim = cfg.model.get('fourier_dim', 64)
        scales = cfg.model.get('fourier_scales', [1.0, 10.0])

        # 3. Buffers de Normalisation (pour rester sur le GPU/CPU automatiquement)
        self.register_buffer('A_min', torch.tensor(b['A'][0]))
        self.register_buffer('A_max', torch.tensor(b['A'][1]))
        self.register_buffer('w0_min', torch.tensor(b['w0'][0]))
        self.register_buffer('w0_max', torch.tensor(b['w0'][1]))
        self.register_buffer('f_min', torch.tensor(b['f'][0]))
        self.register_buffer('f_max', torch.tensor(b['f'][1]))
        self.register_buffer('r_max', torch.tensor(cfg.physics['r_max']))
        self.register_buffer('z_max', torch.tensor(cfg.physics['z_max']))
        self.register_buffer('epsilon_curr', torch.tensor(cfg.model.get('epsilon_curriculum', 0.01)))

        # 4. BRANCH NET (Params: A, w0, f)
        layers_b = []
        prev_dim = cfg.model['branch_dim']
        for h_dim in branch_arch:
            layers_b.append(nn.Linear(prev_dim, h_dim))
            layers_b.append(nn.SiLU()) # SiLU est souvent plus stable que ReLU pour les PINNs
            prev_dim = h_dim
        layers_b.append(nn.Linear(prev_dim, 2 * latent_dim)) # x2 pour Partie Réelle et Imaginaire
        self.branch_net = nn.Sequential(*layers_b)

        # 5. TRUNK NET (Coords: r, z)
        self.trunk_encoding = MultiScaleFourierFeatureEncoding(cfg.model['trunk_dim'], fourier_dim, scales)
        
        layers_t = []
        prev_dim = self.trunk_encoding.out_dim # Sortie du Fourier (2 * fourier_dim)
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
        # Utile pour w0 et f qui varient sur plusieurs ordres de grandeur
        return self.normalize_linear(torch.log10(torch.abs(x) + 1e-9), 
                                     torch.log10(torch.abs(x_min) + 1e-9), 
                                     torch.log10(torch.abs(x_max) + 1e-9))

    def forward(self, params, coords):
        # params: [A, w0, f] | coords: [r, z]
        f_raw = params[:, 2:3]
        r_raw = coords[:, 0:1]
        z_raw = coords[:, 1:2]

        # Normalisation
        params_norm = torch.cat([
            self.normalize_linear(params[:, 0:1], self.A_min, self.A_max),
            self.normalize_log(params[:, 1:2], self.w0_min, self.w0_max),
            self.normalize_log(params[:, 2:3], self.f_min, self.f_max)
        ], dim=1)

        coords_norm = torch.cat([
            self.normalize_linear(r_raw, 0.0, self.r_max),
            self.normalize_linear(z_raw, 0.0, self.z_max)
        ], dim=1)

        # DeepONet Dot Product
        B = self.branch_net(params_norm)
        T = self.trunk_net(self.trunk_encoding(coords_norm))
        
        # Séparation Réel / Imaginaire
        B_re, B_im = torch.split(B, self.latent_dim, dim=1)

        env_re = torch.sum(B_re * T, dim=1, keepdim=True)
        env_im = torch.sum(B_im * T, dim=1, keepdim=True)

        # --- PHYSIQUE : Pré-conditionnement par la lentille ---
        # On injecte la courbure de phase analytique de la lentille mince
        phase_lens = - (self.k_phys * r_raw**2) / (2.0 * f_raw + 1e-9)
        cos_p = torch.cos(phase_lens)
        sin_p = torch.sin(phase_lens)

        # Facteur géométrique pour éviter les singularités au foyer (epsilon)
        geom_scale = 1.0 / torch.sqrt( (1.0 - z_raw/f_raw)**2 + self.epsilon_curr )

        # Reconstruction du champ complexe
        out_re = geom_scale * (env_re * cos_p - env_im * sin_p)
        out_im = geom_scale * (env_re * sin_p + env_im * cos_p)

        return out_re, out_im

    def set_epsilon(self, new_eps):
        """Ajuste l'epsilon du curriculum pendant l'entraînement."""
        self.epsilon_curr.fill_(new_eps)
