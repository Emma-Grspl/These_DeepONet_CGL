import numpy as np
import torch
import torch.nn as nn

from src.models.cgl_deeponet_amp_phase import ModifiedMLP, MultiScaleFourierFeatureEncoding


class CGL_LocalPhysics_DeepONet_AmpPhase(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        local_cfg = cfg["local_physics"] if isinstance(cfg, dict) else cfg.local_physics
        model_cfg = cfg["model_local"] if isinstance(cfg, dict) else cfg.model_local
        physics_cfg = cfg["physics"] if isinstance(cfg, dict) else cfg.physics

        sensor_nx = int(local_cfg["sensor_nx"])
        latent_dim = int(model_cfg["latent_dim"])
        branch_layers = model_cfg.get("branch_layers", [512, 512, 512])
        trunk_layers = model_cfg.get("trunk_layers", [256, 256, 256])
        fourier_dim = int(model_cfg.get("fourier_dim", 64))
        scales = model_cfg.get("fourier_scales", [1.0, 2.0, 5.0, 10.0])

        bounds_cfg = model_cfg.get("delta_bounds", {})
        self.delta_log_amp_bound = float(bounds_cfg.get("log_amp", 4.0))
        self.delta_phase_bound = float(bounds_cfg.get("phase", 12.0))

        phase_gate_cfg = model_cfg.get("phase_gate", {})
        self.phase_gate_enabled = bool(phase_gate_cfg.get("enabled", False))
        self.phase_gate_relative_floor = float(phase_gate_cfg.get("relative_floor", 0.05))
        self.phase_gate_absolute_floor = float(phase_gate_cfg.get("absolute_floor", 1.0e-6))
        self.phase_gate_exponent = float(phase_gate_cfg.get("exponent", 1.0))

        self.sensor_nx = sensor_nx
        self.latent_dim = latent_dim
        self.branch_input_dim = 3 * sensor_nx + 9 + 1

        x_min, x_max = physics_cfg["x_domain"]
        periodic = int(physics_cfg["initial_conditions"][0]) != 2
        x_sensor = self._build_sensor_grid(float(x_min), float(x_max), sensor_nx, periodic)

        self.register_buffer("x_min", torch.tensor(float(x_min)))
        self.register_buffer("x_max", torch.tensor(float(x_max)))
        self.register_buffer("x_sensor", x_sensor)
        self.register_buffer("max_window_dt", torch.tensor(float(local_cfg["window_dt"])))
        self.register_buffer("amp_floor", torch.tensor(float(local_cfg.get("amp_floor", 1.0e-6))))

        self.periodic = periodic
        self.branch_net = ModifiedMLP(
            input_dim=self.branch_input_dim,
            hidden_layers=branch_layers,
            output_dim=2 * latent_dim,
        )
        self.trunk_encoding = MultiScaleFourierFeatureEncoding(2, fourier_dim, scales)
        self.trunk_net = ModifiedMLP(
            input_dim=self.trunk_encoding.out_dim,
            hidden_layers=trunk_layers,
            output_dim=latent_dim,
        )

    @staticmethod
    def _build_sensor_grid(x_min, x_max, nx, periodic):
        if periodic:
            values = np.linspace(x_min, x_max, nx, endpoint=False, dtype=np.float32)
        else:
            values = np.linspace(x_min, x_max, nx, endpoint=True, dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)

    def _decode_window_dt(self, branch_features):
        dt_norm = branch_features[:, -1:]
        return 0.5 * (dt_norm + 1.0) * self.max_window_dt

    def _parse_state_features(self, branch_features):
        n = self.sensor_nx
        log_amp = branch_features[:, :n]
        cos_phase = branch_features[:, n : 2 * n]
        sin_phase = branch_features[:, 2 * n : 3 * n]
        amp = torch.clamp(torch.exp(log_amp) - self.amp_floor, min=0.0)
        return amp, cos_phase, sin_phase

    def _interp_sensor_field(self, values, x_query):
        x_query = x_query[:, 0:1]
        x0 = self.x_sensor[0]
        dx = self.x_sensor[1] - self.x_sensor[0]
        n = self.sensor_nx

        if self.periodic:
            period = (self.x_max - self.x_min)
            x_wrapped = torch.remainder(x_query - x0, period) + x0
            pos = (x_wrapped - x0) / dx
            idx0 = torch.floor(pos).long() % n
            frac = pos - idx0.to(pos.dtype)
            idx1 = (idx0 + 1) % n
        else:
            x_clamped = torch.clamp(x_query, min=float(self.x_min), max=float(self.x_max))
            pos = (x_clamped - x0) / dx
            idx0 = torch.floor(pos).long().clamp(min=0, max=n - 2)
            frac = pos - idx0.to(pos.dtype)
            idx1 = idx0 + 1

        v0 = values.gather(1, idx0)
        v1 = values.gather(1, idx1)
        return (1.0 - frac) * v0 + frac * v1

    def _interpolate_initial_state(self, branch_features, x_query):
        amp_sensor, cos_sensor, sin_sensor = self._parse_state_features(branch_features)
        amp0 = self._interp_sensor_field(amp_sensor, x_query)
        cos0 = self._interp_sensor_field(cos_sensor, x_query)
        sin0 = self._interp_sensor_field(sin_sensor, x_query)
        phase0 = torch.atan2(sin0, cos0)
        return amp0, phase0

    def _normalize_x(self, x):
        return 2.0 * (x - self.x_min) / (self.x_max - self.x_min + 1.0e-9) - 1.0

    def _apply_phase_gate(self, amp0, delta_phase, branch_features):
        if not self.phase_gate_enabled:
            return delta_phase
        amp_sensor, _, _ = self._parse_state_features(branch_features)
        amp_ref = torch.amax(amp_sensor, dim=1, keepdim=True)
        floor = self.phase_gate_relative_floor * amp_ref + self.phase_gate_absolute_floor
        gate = torch.pow(amp0 / (amp0 + floor + 1.0e-12), self.phase_gate_exponent)
        return delta_phase * gate

    def forward(self, branch_features, coords_local):
        if coords_local.ndim == 1:
            coords_local = coords_local.unsqueeze(1)

        x_query = coords_local[:, 0:1]
        tau_query = coords_local[:, 1:2]
        window_dt = torch.clamp(self._decode_window_dt(branch_features), min=1.0e-8)
        tau_norm = torch.clamp(tau_query / window_dt, min=0.0)

        amp0, phase0 = self._interpolate_initial_state(branch_features, x_query)

        branch_latent = self.branch_net(branch_features)
        trunk_input = torch.cat([self._normalize_x(x_query), torch.clamp(2.0 * tau_norm - 1.0, min=-1.0, max=1.0)], dim=1)
        trunk_latent = self.trunk_net(self.trunk_encoding(trunk_input))

        branch_amp, branch_phase = torch.split(branch_latent, self.latent_dim, dim=1)
        raw_delta_log_amp = torch.sum(branch_amp * trunk_latent, dim=1, keepdim=True)
        raw_delta_phase = torch.sum(branch_phase * trunk_latent, dim=1, keepdim=True)

        transition = torch.clamp(tau_norm, min=0.0, max=1.0)
        bounded_delta_log_amp = self.delta_log_amp_bound * torch.tanh(raw_delta_log_amp / self.delta_log_amp_bound)
        bounded_delta_phase = self.delta_phase_bound * torch.tanh(raw_delta_phase / self.delta_phase_bound)
        delta_log_amp = transition * bounded_delta_log_amp
        delta_phase = transition * bounded_delta_phase
        delta_phase = self._apply_phase_gate(amp0, delta_phase, branch_features)

        amp = torch.exp(torch.log(amp0 + self.amp_floor) + delta_log_amp) - self.amp_floor
        phase = phase0 + delta_phase
        psi_re = amp * torch.cos(phase)
        psi_im = amp * torch.sin(phase)
        return psi_re, psi_im
