import os

import numpy as np
import torch

from src.data.local_operator_amp_phase import (
    _cfg_get,
    build_single_case_params,
    interp_complex_field,
    normalize_linear,
    normalize_log,
    normalize_params,
    prepare_single_case_trajectory,
)


def complex_to_state_features_reim(u_complex):
    return np.concatenate(
        [
            np.real(u_complex).astype(np.float32),
            np.imag(u_complex).astype(np.float32),
        ],
        axis=0,
    )


def build_branch_features_reim(cfg, u_sensor, dt_value, params_dict):
    local_cfg = _cfg_get(cfg, "local_operator")
    state_features = complex_to_state_features_reim(u_sensor)
    dt_norm = np.asarray([normalize_linear(dt_value, 0.0, float(local_cfg["rollout_dt"]))], dtype=np.float32)
    params_norm = normalize_params(cfg, params_dict)
    return np.concatenate([state_features, params_norm, dt_norm], axis=0).astype(np.float32)


def sample_training_batch_reim(trajectory, cfg, pair_indices, batch_pairs, batch_queries, device):
    params = trajectory["params"]
    x_solver = trajectory["x_solver"]
    u_solver = trajectory["u_solver"]
    u_sensor = trajectory["u_sensor"]
    dt_value = trajectory["dt"]

    chosen_pairs = np.random.choice(pair_indices, size=batch_pairs, replace=True)
    branch_rows = []
    x_rows = []
    current_rows = []
    target_rows = []

    for pair_idx in chosen_pairs:
        state_vec = build_branch_features_reim(cfg, u_sensor[:, pair_idx], dt_value, params)
        q_idx = np.random.choice(len(x_solver), size=batch_queries, replace=True)
        branch_rows.append(np.repeat(state_vec[None, :], batch_queries, axis=0))
        x_rows.append(x_solver[q_idx, None])
        current_rows.append(u_solver[q_idx, pair_idx])
        target_rows.append(u_solver[q_idx, pair_idx + 1])

    branch = np.concatenate(branch_rows, axis=0)
    x_query = np.concatenate(x_rows, axis=0).astype(np.float32)
    current_u = np.concatenate(current_rows, axis=0).astype(np.complex64)
    target_u = np.concatenate(target_rows, axis=0).astype(np.complex64)
    delta_u = target_u - current_u

    return {
        "branch": torch.tensor(branch, dtype=torch.float32, device=device),
        "x_query": torch.tensor(x_query, dtype=torch.float32, device=device),
        "current_u_re": torch.tensor(np.real(current_u)[:, None], dtype=torch.float32, device=device),
        "current_u_im": torch.tensor(np.imag(current_u)[:, None], dtype=torch.float32, device=device),
        "target_u_re": torch.tensor(np.real(target_u)[:, None], dtype=torch.float32, device=device),
        "target_u_im": torch.tensor(np.imag(target_u)[:, None], dtype=torch.float32, device=device),
        "target_delta_re": torch.tensor(np.real(delta_u)[:, None], dtype=torch.float32, device=device),
        "target_delta_im": torch.tensor(np.imag(delta_u)[:, None], dtype=torch.float32, device=device),
    }


def rollout_local_model_reim(model, trajectory, cfg, device):
    model.eval()
    params = trajectory["params"]
    x_sensor = trajectory["x_sensor"]
    x_solver = trajectory["x_solver"]
    u_sensor_ref = trajectory["u_sensor"]
    u_solver_ref = trajectory["u_solver"]
    dt_value = trajectory["dt"]
    periodic = trajectory["periodic"]

    n_steps = u_solver_ref.shape[1] - 1
    pred_sensor = np.zeros_like(u_sensor_ref)
    pred_sensor[:, 0] = u_sensor_ref[:, 0]
    pred_solver = np.zeros_like(u_solver_ref)
    pred_solver[:, 0] = u_solver_ref[:, 0]
    rel_l2 = np.zeros(n_steps + 1, dtype=np.float64)

    with torch.no_grad():
        for step in range(n_steps):
            branch_vec = build_branch_features_reim(cfg, pred_sensor[:, step], dt_value, params)
            branch_tensor = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device)
            branch_sensor = branch_tensor.repeat(len(x_sensor), 1)
            x_sensor_tensor = torch.tensor(x_sensor[:, None], dtype=torch.float32, device=device)
            delta_re, delta_im = model(branch_sensor, x_sensor_tensor)
            current_sensor = pred_sensor[:, step]
            next_sensor = current_sensor + delta_re.cpu().numpy().reshape(-1) + 1j * delta_im.cpu().numpy().reshape(-1)
            pred_sensor[:, step + 1] = next_sensor.astype(np.complex64)
            pred_solver[:, step + 1] = interp_complex_field(x_sensor, pred_sensor[:, step + 1], x_solver, periodic)
            diff = pred_solver[:, step + 1] - u_solver_ref[:, step + 1]
            denom = np.linalg.norm(u_solver_ref[:, step + 1]) + 1e-12
            rel_l2[step + 1] = np.linalg.norm(diff) / denom

    return {
        "x_sensor": x_sensor,
        "x_solver": x_solver,
        "t_values": trajectory["t_values"],
        "u_sensor_pred": pred_sensor,
        "u_solver_pred": pred_solver,
        "u_solver_ref": u_solver_ref,
        "rel_l2": rel_l2,
    }
