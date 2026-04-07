import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation="silu"):
        super().__init__()
        act = nn.SiLU if activation == "silu" else nn.Tanh
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CGLLocalOperator(nn.Module):
    """
    Operateur local de type DeepONet:
    - branch: etat courant echantillonne + parametres + dt_local
    - trunk: position x
    - sortie: etat complexe au pas suivant en ce point x
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"] if isinstance(cfg, dict) else cfg.model
        operator_cfg = cfg["operator"] if isinstance(cfg, dict) else cfg.operator
        physics_cfg = cfg["physics"] if isinstance(cfg, dict) else cfg.physics

        self.sensor_nx = int(operator_cfg["sensor_nx"])
        self.branch_input_dim = 2 * self.sensor_nx + 10
        self.latent_dim = int(model_cfg["latent_dim"])

        self.branch_net = MLP(
            input_dim=self.branch_input_dim,
            hidden_layers=list(model_cfg["branch_layers"]),
            output_dim=2 * self.latent_dim,
            activation=model_cfg.get("activation", "silu"),
        )
        self.trunk_net = MLP(
            input_dim=1,
            hidden_layers=list(model_cfg["trunk_layers"]),
            output_dim=self.latent_dim,
            activation=model_cfg.get("activation", "silu"),
        )

        x_min, x_max = physics_cfg["x_domain"]
        self.register_buffer("x_min", torch.tensor(float(x_min)))
        self.register_buffer("x_max", torch.tensor(float(x_max)))

    def normalize_x(self, x):
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min + 1e-9) - 1.0

    def forward(self, branch_input, x_coord):
        x_norm = self.normalize_x(x_coord)
        b = self.branch_net(branch_input)
        t = self.trunk_net(x_norm)
        b_re, b_im = torch.split(b, self.latent_dim, dim=1)
        out_re = torch.sum(b_re * t, dim=1, keepdim=True)
        out_im = torch.sum(b_im * t, dim=1, keepdim=True)
        return out_re, out_im
