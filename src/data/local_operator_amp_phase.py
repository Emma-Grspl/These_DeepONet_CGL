import os

import numpy as np
import torch

from src.utils.solver_cgl import get_ground_truth_CGL


def _cfg_get(cfg, *keys):
    obj = cfg
    for key in keys:
        obj = obj[key]
    return obj


def _is_periodic_case(params_dict):
    return int(params_dict.get("type", 0)) != 2


def build_single_case_params(cfg):
    bounds = _cfg_get(cfg, "physics", "bounds")
    eq = _cfg_get(cfg, "physics", "equation_params")
    return {
        "alpha": float(eq["alpha"][0]),
        "beta": float(eq["beta"][0]),
        "mu": float(eq["mu"][0]),
        "V": float(eq["V"][0]),
        "A": float(bounds["A"][0]),
        "w0": float(bounds["w0"][0]),
        "x0": float(bounds["x0"][0]),
        "k": float(bounds["k"][0]),
        "type": int(_cfg_get(cfg, "physics", "initial_conditions")[0]),
    }


def make_spatial_grid(x_min, x_max, nx, periodic):
    if periodic:
        return np.linspace(x_min, x_max, nx, endpoint=False)
    return np.linspace(x_min, x_max, nx, endpoint=True)


def normalize_linear(value, vmin, vmax):
    denom = float(vmax) - float(vmin)
    return 2.0 * (value - float(vmin)) / (denom + 1e-9) - 1.0


def normalize_log(value, vmin, vmax):
    return normalize_linear(
        np.log10(np.abs(value) + 1e-9),
        np.log10(np.abs(vmin) + 1e-9),
        np.log10(np.abs(vmax) + 1e-9),
    )


def normalize_params(cfg, params_dict):
    eq = _cfg_get(cfg, "physics", "equation_params")
    bounds = _cfg_get(cfg, "physics", "bounds")
    return np.asarray(
        [
            normalize_linear(params_dict["alpha"], eq["alpha"][0], eq["alpha"][1]),
            normalize_linear(params_dict["beta"], eq["beta"][0], eq["beta"][1]),
            normalize_linear(params_dict["mu"], eq["mu"][0], eq["mu"][1]),
            normalize_linear(params_dict["V"], eq["V"][0], eq["V"][1]),
            normalize_linear(params_dict["A"], bounds["A"][0], bounds["A"][1]),
            normalize_log(params_dict["w0"], bounds["w0"][0], bounds["w0"][1]),
            normalize_linear(params_dict["x0"], bounds["x0"][0], bounds["x0"][1]),
            normalize_linear(params_dict["k"], bounds["k"][0], bounds["k"][1]),
            normalize_linear(float(params_dict["type"]), 0.0, 2.0),
        ],
        dtype=np.float32,
    )


def _periodic_interp_real(x_src, y_src, x_dst):
    dx = x_src[1] - x_src[0]
    x0 = x_src[0]
    period = (x_src[-1] - x_src[0]) + dx
    x_wrapped = ((x_dst - x0) % period) + x0
    return np.interp(x_wrapped, x_src, y_src, period=period)


def interp_complex_field(x_src, u_src, x_dst, periodic):
    if periodic:
        re = _periodic_interp_real(x_src, np.real(u_src), x_dst)
        im = _periodic_interp_real(x_src, np.imag(u_src), x_dst)
    else:
        re = np.interp(x_dst, x_src, np.real(u_src))
        im = np.interp(x_dst, x_src, np.imag(u_src))
    return re + 1j * im


def complex_to_state_features(u_complex, amp_floor):
    amp = np.abs(u_complex)
    phase = np.angle(u_complex)
    log_amp = np.log(amp + amp_floor)
    return np.concatenate(
        [
            log_amp.astype(np.float32),
            np.cos(phase).astype(np.float32),
            np.sin(phase).astype(np.float32),
        ],
        axis=0,
    )


def build_branch_features(cfg, u_sensor, dt_value, params_dict):
    local_cfg = _cfg_get(cfg, "local_operator")
    state_features = complex_to_state_features(u_sensor, float(local_cfg["amp_floor"]))
    dt_norm = np.asarray([normalize_linear(dt_value, 0.0, float(local_cfg["rollout_dt"]))], dtype=np.float32)
    params_norm = normalize_params(cfg, params_dict)
    return np.concatenate([state_features, params_norm, dt_norm], axis=0).astype(np.float32)


def _phase_gate_cfg(cfg):
    model_cfg = _cfg_get(cfg, "model_local") if isinstance(cfg, dict) else cfg["model_local"]
    return model_cfg.get("local_phase_gate", {})


def apply_phase_gate_numpy(cfg, current_amp, delta_phase):
    gate_cfg = _phase_gate_cfg(cfg)
    if not gate_cfg or not bool(gate_cfg.get("enabled", False)):
        return delta_phase
    current_amp = np.asarray(current_amp, dtype=np.float32)
    delta_phase = np.asarray(delta_phase, dtype=np.float32)
    ref_amp = float(np.max(current_amp))
    relative_floor = float(gate_cfg.get("relative_floor", 0.05))
    absolute_floor = float(gate_cfg.get("absolute_floor", 1.0e-6))
    exponent = float(gate_cfg.get("exponent", 1.0))
    floor = relative_floor * ref_amp + absolute_floor
    gate = np.power(current_amp / (current_amp + floor + 1.0e-12), exponent)
    return delta_phase * gate


def apply_phase_gate_torch(cfg, current_amp, delta_phase):
    gate_cfg = _phase_gate_cfg(cfg)
    if not gate_cfg or not bool(gate_cfg.get("enabled", False)):
        return delta_phase
    ref_amp = torch.amax(current_amp.detach())
    relative_floor = float(gate_cfg.get("relative_floor", 0.05))
    absolute_floor = float(gate_cfg.get("absolute_floor", 1.0e-6))
    exponent = float(gate_cfg.get("exponent", 1.0))
    floor = relative_floor * ref_amp + absolute_floor
    gate = torch.pow(current_amp / (current_amp + floor + 1.0e-12), exponent)
    return delta_phase * gate


def _moving_average_real(values, window, periodic):
    window = int(max(1, window))
    if window <= 1:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    if periodic:
        pad = window // 2
        extended = np.pad(values, (pad, pad), mode="wrap")
        filtered = np.convolve(extended, kernel, mode="valid")
        return filtered[: len(values)].astype(np.float32, copy=False)
    return np.convolve(values, kernel, mode="same").astype(np.float32, copy=False)


def _blend_angles(prev_phase, next_phase, blend_weight):
    z = blend_weight * np.exp(1j * prev_phase) + (1.0 - blend_weight) * np.exp(1j * next_phase)
    return np.angle(z).astype(np.float32)


def apply_rollout_correction_numpy(cfg, current_sensor, next_sensor, periodic):
    correction_cfg = _cfg_get(cfg, "rollout_correction") if "rollout_correction" in cfg else {}
    if not correction_cfg or not bool(correction_cfg.get("enabled", False)):
        return next_sensor.astype(np.complex64, copy=False)

    current_sensor = np.asarray(current_sensor, dtype=np.complex64)
    next_sensor = np.asarray(next_sensor, dtype=np.complex64)

    current_amp = np.abs(current_sensor).astype(np.float32)
    current_phase = np.angle(current_sensor).astype(np.float32)
    next_amp = np.abs(next_sensor).astype(np.float32)
    next_phase = np.angle(next_sensor).astype(np.float32)

    ref_amp = float(np.max(current_amp))
    abs_floor = float(correction_cfg.get("absolute_floor", 1.0e-8))
    low_amp_threshold = float(correction_cfg.get("low_amp_relative_threshold", 0.05)) * ref_amp + abs_floor
    amp_cap = float(correction_cfg.get("amp_cap_factor", 1.25)) * max(ref_amp, abs_floor)
    phase_blend = float(correction_cfg.get("low_amp_phase_blend", 0.9))
    amp_window = int(correction_cfg.get("amp_smoothing_window", 1))
    complex_window = int(correction_cfg.get("complex_smoothing_window", 1))

    next_amp = np.clip(next_amp, 0.0, amp_cap)
    if amp_window > 1:
        next_amp = _moving_average_real(next_amp, amp_window, periodic)

    low_mask = current_amp <= low_amp_threshold
    blended_phase = _blend_angles(current_phase, next_phase, phase_blend)
    next_phase = np.where(low_mask, blended_phase, next_phase)
    corrected = next_amp * np.exp(1j * next_phase)

    if complex_window > 1 and np.any(low_mask):
        smooth_re = _moving_average_real(np.real(corrected), complex_window, periodic)
        smooth_im = _moving_average_real(np.imag(corrected), complex_window, periodic)
        smooth_complex = smooth_re + 1j * smooth_im
        corrected = np.where(low_mask, smooth_complex, corrected)

    return corrected.astype(np.complex64, copy=False)


def prepare_single_case_trajectory(cfg, t_max_override=None, dt_override=None):
    params = build_single_case_params(cfg)
    x_min, x_max = _cfg_get(cfg, "physics", "x_domain")
    local_cfg = _cfg_get(cfg, "local_operator")
    dt = float(local_cfg["train_dt"] if dt_override is None else dt_override)
    t_max = float(_cfg_get(cfg, "physics", "t_max") if t_max_override is None else t_max_override)

    n_steps = int(round(t_max / dt))
    t_max_effective = n_steps * dt
    if abs(t_max_effective - t_max) > 1e-8:
        raise ValueError(f"t_max={t_max} n'est pas divisible par dt={dt}.")

    periodic = _is_periodic_case(params)
    solver_nx = int(local_cfg["solver_nx"])
    sensor_nx = int(local_cfg["sensor_nx"])

    X, T, U = get_ground_truth_CGL(
        params,
        x_min,
        x_max,
        t_max_effective,
        Nx=solver_nx,
        Nt=n_steps + 1,
    )
    x_solver = X[:, 0]
    t_values = T[0, :]
    x_sensor = make_spatial_grid(x_min, x_max, sensor_nx, periodic)

    u_sensor = np.stack(
        [interp_complex_field(x_solver, U[:, idx], x_sensor, periodic) for idx in range(U.shape[1])],
        axis=1,
    )

    return {
        "params": params,
        "x_solver": x_solver.astype(np.float32),
        "x_sensor": x_sensor.astype(np.float32),
        "t_values": t_values.astype(np.float32),
        "u_solver": U.astype(np.complex64),
        "u_sensor": u_sensor.astype(np.complex64),
        "dt": float(dt),
        "t_max": float(t_max_effective),
        "periodic": periodic,
    }


def split_pair_indices(trajectory, train_ratio):
    n_pairs = trajectory["u_solver"].shape[1] - 1
    n_train = max(1, min(n_pairs - 1, int(round(train_ratio * n_pairs))))
    train_idx = np.arange(0, n_train, dtype=np.int64)
    valid_idx = np.arange(n_train, n_pairs, dtype=np.int64)
    if len(valid_idx) == 0:
        valid_idx = train_idx[-1:].copy()
    return train_idx, valid_idx


def sample_training_batch(trajectory, cfg, pair_indices, batch_pairs, batch_queries, device):
    amp_floor = float(_cfg_get(cfg, "local_operator", "amp_floor"))
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
        state_vec = build_branch_features(cfg, u_sensor[:, pair_idx], dt_value, params)
        q_idx = np.random.choice(len(x_solver), size=batch_queries, replace=True)
        branch_rows.append(np.repeat(state_vec[None, :], batch_queries, axis=0))
        x_rows.append(x_solver[q_idx, None])
        current_rows.append(u_solver[q_idx, pair_idx])
        target_rows.append(u_solver[q_idx, pair_idx + 1])

    branch = np.concatenate(branch_rows, axis=0)
    x_query = np.concatenate(x_rows, axis=0).astype(np.float32)
    current_u = np.concatenate(current_rows, axis=0).astype(np.complex64)
    target_u = np.concatenate(target_rows, axis=0).astype(np.complex64)

    current_amp = np.abs(current_u).astype(np.float32)[:, None]
    current_phase = np.angle(current_u).astype(np.float32)[:, None]
    target_amp = np.abs(target_u).astype(np.float32)[:, None]
    target_phase = np.angle(target_u).astype(np.float32)[:, None]
    target_delta_log_amp = np.log(target_amp + amp_floor) - np.log(current_amp + amp_floor)
    target_delta_phase = np.angle(target_u * np.conj(current_u)).astype(np.float32)[:, None]

    batch = {
        "branch": torch.tensor(branch, dtype=torch.float32, device=device),
        "x_query": torch.tensor(x_query, dtype=torch.float32, device=device),
        "current_amp": torch.tensor(current_amp, dtype=torch.float32, device=device),
        "current_phase": torch.tensor(current_phase, dtype=torch.float32, device=device),
        "target_u_re": torch.tensor(np.real(target_u)[:, None], dtype=torch.float32, device=device),
        "target_u_im": torch.tensor(np.imag(target_u)[:, None], dtype=torch.float32, device=device),
        "target_amp": torch.tensor(target_amp, dtype=torch.float32, device=device),
        "target_delta_log_amp": torch.tensor(target_delta_log_amp, dtype=torch.float32, device=device),
        "target_delta_phase": torch.tensor(target_delta_phase, dtype=torch.float32, device=device),
    }
    return batch


def rollout_local_model(model, trajectory, cfg, device):
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
            branch_vec = build_branch_features(cfg, pred_sensor[:, step], dt_value, params)
            branch_tensor = torch.tensor(branch_vec[None, :], dtype=torch.float32, device=device)
            branch_sensor = branch_tensor.repeat(len(x_sensor), 1)
            x_sensor_tensor = torch.tensor(x_sensor[:, None], dtype=torch.float32, device=device)
            delta_log_amp_sensor, delta_phase_sensor = model(branch_sensor, x_sensor_tensor)

            current_sensor = pred_sensor[:, step]
            current_amp_sensor = np.abs(current_sensor)
            current_phase_sensor = np.angle(current_sensor)
            next_amp_sensor = np.exp(
                np.log(current_amp_sensor + float(_cfg_get(cfg, "local_operator", "amp_floor")))
                + delta_log_amp_sensor.cpu().numpy().reshape(-1)
            ) - float(_cfg_get(cfg, "local_operator", "amp_floor"))
            gated_delta_phase = apply_phase_gate_numpy(cfg, current_amp_sensor, delta_phase_sensor.cpu().numpy().reshape(-1))
            next_phase_sensor = current_phase_sensor + gated_delta_phase
            pred_sensor[:, step + 1] = next_amp_sensor * np.exp(1j * next_phase_sensor)

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


def save_rollout_metrics(output_dir, rollout):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "rollout_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("time,rel_l2\n")
        for t_val, err in zip(rollout["t_values"], rollout["rel_l2"]):
            handle.write(f"{float(t_val):.8f},{float(err):.10f}\n")
    return csv_path
