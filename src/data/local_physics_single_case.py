import numpy as np

from src.utils.solver_cgl import get_ground_truth_CGL


def build_single_case_params(cfg):
    physics_cfg = cfg["physics"] if isinstance(cfg, dict) else cfg.physics
    bounds = physics_cfg["bounds"]
    eq = physics_cfg["equation_params"]
    return {
        "alpha": float(eq["alpha"][0]),
        "beta": float(eq["beta"][0]),
        "mu": float(eq["mu"][0]),
        "V": float(eq["V"][0]),
        "A": float(bounds["A"][0]),
        "w0": float(bounds["w0"][0]),
        "x0": float(bounds["x0"][0]),
        "k": float(bounds["k"][0]),
        "type": int(physics_cfg["initial_conditions"][0]),
    }


def is_periodic_case(params_dict):
    return int(params_dict.get("type", 0)) != 2


def make_spatial_grid(x_min, x_max, nx, periodic):
    if periodic:
        return np.linspace(x_min, x_max, nx, endpoint=False, dtype=np.float32)
    return np.linspace(x_min, x_max, nx, endpoint=True, dtype=np.float32)


def analytical_initial_field(params_dict, x_grid):
    A = float(params_dict["A"])
    w0 = float(params_dict["w0"])
    x0 = float(params_dict["x0"])
    k = float(params_dict["k"])
    ic_type = int(params_dict["type"])

    X = (x_grid - x0) / (w0 + 1.0e-9)
    phase = np.exp(1j * k * (x_grid - x0))
    if ic_type == 0:
        envelope = A * np.exp(-(X ** 2))
    elif ic_type == 1:
        envelope = A / np.cosh(X)
    elif ic_type == 2:
        envelope = A * np.tanh(X)
    else:
        envelope = A * np.exp(-(X ** 2))
    return (envelope * phase).astype(np.complex64)


def normalize_linear(value, vmin, vmax):
    denom = float(vmax) - float(vmin)
    return 2.0 * (value - float(vmin)) / (denom + 1.0e-9) - 1.0


def normalize_log(value, vmin, vmax):
    return normalize_linear(
        np.log10(np.abs(value) + 1.0e-9),
        np.log10(np.abs(vmin) + 1.0e-9),
        np.log10(np.abs(vmax) + 1.0e-9),
    )


def normalize_params(cfg, params_dict):
    physics_cfg = cfg["physics"] if isinstance(cfg, dict) else cfg.physics
    eq = physics_cfg["equation_params"]
    bounds = physics_cfg["bounds"]
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


def complex_to_state_features(u_complex, amp_floor):
    amp = np.abs(u_complex).astype(np.float32)
    phase = np.angle(u_complex).astype(np.float32)
    return np.concatenate(
        [
            np.log(amp + float(amp_floor)).astype(np.float32),
            np.cos(phase).astype(np.float32),
            np.sin(phase).astype(np.float32),
        ],
        axis=0,
    )


def build_branch_features(cfg, u_sensor, window_dt, params_dict):
    local_cfg = cfg["local_physics"] if isinstance(cfg, dict) else cfg.local_physics
    state_features = complex_to_state_features(u_sensor, float(local_cfg.get("amp_floor", 1.0e-6)))
    params_norm = normalize_params(cfg, params_dict)
    dt_norm = np.asarray([normalize_linear(window_dt, 0.0, float(local_cfg["window_dt"]))], dtype=np.float32)
    return np.concatenate([state_features, params_norm, dt_norm], axis=0).astype(np.float32)


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
    return (re + 1j * im).astype(np.complex64)


def build_window_schedule(t_max, window_dt):
    t_max = float(t_max)
    window_dt = float(window_dt)
    windows = []
    t_start = 0.0
    while t_start < t_max - 1.0e-12:
        t_end = min(t_start + window_dt, t_max)
        windows.append((round(t_start, 10), round(t_end, 10)))
        t_start = t_end
    return windows


def prepare_reference_trajectory(cfg, t_max_override=None, nx_override=None, nt_override=None):
    params = build_single_case_params(cfg)
    physics_cfg = cfg["physics"] if isinstance(cfg, dict) else cfg.physics
    x_min, x_max = physics_cfg["x_domain"]
    t_max = float(physics_cfg["t_max"] if t_max_override is None else t_max_override)
    local_cfg = cfg["local_physics"] if isinstance(cfg, dict) else cfg.local_physics
    nx = int(local_cfg.get("solver_nx", 256) if nx_override is None else nx_override)
    X, T, U = get_ground_truth_CGL(params, x_min, x_max, t_max, Nx=nx, Nt=nt_override)
    return {
        "params": params,
        "x": X[:, 0].astype(np.float32),
        "t": T[0, :].astype(np.float32),
        "u": U.astype(np.complex64),
    }
