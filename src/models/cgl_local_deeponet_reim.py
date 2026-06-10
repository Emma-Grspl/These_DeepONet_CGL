import torch
import torch.nn as nn

from src.models.cgl_deeponet_amp_phase import ModifiedMLP, MultiScaleFourierFeatureEncoding


class CGL_LocalDirect_DeepONet_ReIm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        local_cfg = cfg["local_operator"] if isinstance(cfg, dict) else cfg.local_operator
        model_cfg = cfg["model_local"] if isinstance(cfg, dict) else cfg.model_local
        physics = cfg["physics"] if isinstance(cfg, dict) else cfg.physics

        sensor_nx = int(local_cfg["sensor_nx"])
        latent_dim = int(model_cfg["latent_dim"])
        branch_layers = model_cfg.get("branch_layers", [512, 512, 512])
        trunk_layers = model_cfg.get("trunk_layers", [256, 256, 256])
        fourier_dim = int(model_cfg.get("fourier_dim", 64))
        scales = model_cfg.get("fourier_scales", [1.0, 2.0, 5.0, 10.0])
        residual_cfg = model_cfg.get("local_residual_ansatz", {})

        self.branch_input_dim = 2 * sensor_nx + 9 + 1
        self.latent_dim = latent_dim
        self.use_local_residual_ansatz = bool(residual_cfg.get("enabled", False))
        self.delta_re_rate_bound = float(residual_cfg.get("delta_re_rate_bound", 20.0))
        self.delta_im_rate_bound = float(residual_cfg.get("delta_im_rate_bound", 20.0))

        x_min, x_max = physics["x_domain"]
        self.register_buffer("x_min", torch.tensor(float(x_min)))
        self.register_buffer("x_max", torch.tensor(float(x_max)))
        self.register_buffer("dt_max", torch.tensor(float(local_cfg["rollout_dt"])))

        self.branch_net = ModifiedMLP(
            input_dim=self.branch_input_dim,
            hidden_layers=branch_layers,
            output_dim=2 * latent_dim,
        )
        self.trunk_encoding = MultiScaleFourierFeatureEncoding(1, fourier_dim, scales)
        self.trunk_net = ModifiedMLP(
            input_dim=self.trunk_encoding.out_dim,
            hidden_layers=trunk_layers,
            output_dim=latent_dim,
        )

    def normalize_x(self, x):
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min + 1e-9) - 1.0

    def forward(self, branch_features, x_coords):
        if x_coords.ndim == 1:
            x_coords = x_coords.unsqueeze(1)
        x_norm = self.normalize_x(x_coords[:, 0:1])
        branch_latent = self.branch_net(branch_features)
        trunk_latent = self.trunk_net(self.trunk_encoding(x_norm))
        branch_re, branch_im = torch.split(branch_latent, self.latent_dim, dim=1)
        delta_re = torch.sum(branch_re * trunk_latent, dim=1, keepdim=True)
        delta_im = torch.sum(branch_im * trunk_latent, dim=1, keepdim=True)

        if self.use_local_residual_ansatz:
            dt_norm = branch_features[:, -1:].to(delta_re.dtype)
            dt_value = 0.5 * (dt_norm + 1.0) * self.dt_max.to(delta_re.dtype)
            re_rate = self.delta_re_rate_bound * torch.tanh(delta_re / self.delta_re_rate_bound)
            im_rate = self.delta_im_rate_bound * torch.tanh(delta_im / self.delta_im_rate_bound)
            delta_re = dt_value * re_rate
            delta_im = dt_value * im_rate

        return delta_re, delta_im
