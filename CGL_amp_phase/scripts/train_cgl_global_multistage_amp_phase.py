import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.models.cgl_deeponet_amp_phase import CGL_PI_DeepONet_AmpPhase
from src.utils.solver_cgl import get_ground_truth_CGL


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def build_run_dir(project_root, cfg_dict, resume_dir=None):
    configured = cfg_dict["training"]["save_dir"]
    output_root = configured if os.path.isabs(configured) else os.path.join(project_root, configured)
    os.makedirs(output_root, exist_ok=True)
    if resume_dir is not None:
        return resume_dir if os.path.isabs(resume_dir) else os.path.join(project_root, resume_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    run_name = f"run_{timestamp}" if not job_id else f"run_{timestamp}_{job_id}"
    return os.path.join(output_root, run_name)


def stage_name(stage_idx, t_start, t_end):
    return f"stage_{stage_idx:02d}_t{t_start:.1f}_{t_end:.1f}"


def fixed_case_params(cfg_dict):
    eq = cfg_dict["physics"]["equation_params"]
    bounds = cfg_dict["physics"]["bounds"]
    return {
        "alpha": float(eq["alpha"][0]),
        "beta": float(eq["beta"][0]),
        "mu": float(eq["mu"][0]),
        "V": float(eq["V"][0]),
        "A": float(bounds["A"][0]),
        "w0": float(bounds["w0"][0]),
        "x0": float(bounds["x0"][0]),
        "k": float(bounds["k"][0]),
        "type": int(cfg_dict["physics"]["initial_conditions"][0]),
    }


def prepare_single_case_reference(cfg_dict):
    params = fixed_case_params(cfg_dict)
    x_min, x_max = cfg_dict["physics"]["x_domain"]
    t_max = float(cfg_dict["physics"]["t_max"])
    nx = int(cfg_dict["data"]["solver_nx"])
    dt = 0.025
    n_steps = int(round(t_max / dt))
    X, T, U = get_ground_truth_CGL(params, x_min, x_max, t_max, Nx=nx, Nt=n_steps + 1)
    return {
        "params": params,
        "x": X[:, 0].astype(np.float32),
        "t_values": T[0, :].astype(np.float32),
        "u": U.astype(np.complex64),
    }


def sample_block_batch(reference, cfg_dict, t_start, t_end, n_queries, device):
    x = reference["x"]
    t_values = reference["t_values"]
    U = reference["u"]
    params = reference["params"]

    time_mask = (t_values >= t_start - 1e-10) & (t_values <= t_end + 1e-10)
    valid_t_idx = np.nonzero(time_mask)[0]
    chosen_t_idx = np.random.choice(valid_t_idx, size=n_queries, replace=True)
    chosen_x_idx = np.random.choice(len(x), size=n_queries, replace=True)

    coords = np.stack([x[chosen_x_idx], t_values[chosen_t_idx]], axis=1).astype(np.float32)
    u_target = U[chosen_x_idx, chosen_t_idx]
    p_vec = np.array(
        [
            params["alpha"],
            params["beta"],
            params["mu"],
            params["V"],
            params["A"],
            params["w0"],
            params["x0"],
            params["k"],
            float(params["type"]),
        ],
        dtype=np.float32,
    )
    branch = np.repeat(p_vec[None, :], n_queries, axis=0)
    return {
        "branch": torch.tensor(branch, dtype=torch.float32, device=device),
        "coords": torch.tensor(coords, dtype=torch.float32, device=device),
        "target_re": torch.tensor(np.real(u_target)[:, None], dtype=torch.float32, device=device),
        "target_im": torch.tensor(np.imag(u_target)[:, None], dtype=torch.float32, device=device),
    }


def compute_supervised_loss(model, batch):
    pred_re, pred_im = model(batch["branch"], batch["coords"])
    return torch.mean((pred_re - batch["target_re"]) ** 2 + (pred_im - batch["target_im"]) ** 2)


def save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name):
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
    }
    atomic_torch_save(state, os.path.join(ckpt_dir, name))


def load_stage_checkpoint_if_available(model, optimizer, stage_dir, device):
    ckpt_path = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
    if not os.path.exists(ckpt_path):
        return 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_valid_loss", float("inf")))


def eval_block_valid_loss(model, reference, cfg_dict, t_start, t_end, device):
    n_queries = int(cfg_dict["data"]["valid_queries"])
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(4):
            batch = sample_block_batch(reference, cfg_dict, t_start, t_end, n_queries, device)
            vals.append(float(compute_supervised_loss(model, batch).item()))
    return float(np.mean(vals))


def load_best_stage_model(cfg_dict, stage_dir, device):
    ckpt_path = os.path.join(stage_dir, "checkpoints", "model_best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(stage_dir, "checkpoints", "model_latest.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def select_stage_model(models, time_blocks, t_current):
    for (t_start, t_end), model in zip(time_blocks, models):
        if t_current <= t_end + 1e-10:
            return model
    return models[-1]


def rollout_multistage_models(models, time_blocks, reference, device):
    x = reference["x"]
    t_values = reference["t_values"]
    U_true = reference["u"]
    params = reference["params"]
    p_vec = np.array(
        [
            params["alpha"],
            params["beta"],
            params["mu"],
            params["V"],
            params["A"],
            params["w0"],
            params["x0"],
            params["k"],
            float(params["type"]),
        ],
        dtype=np.float32,
    )

    pred = np.zeros_like(U_true)
    for j, t_val in enumerate(t_values):
        model = select_stage_model(models, time_blocks, float(t_val))
        coords = torch.tensor(np.stack([x, np.full_like(x, t_val)], axis=1), dtype=torch.float32, device=device)
        branch = torch.tensor(np.repeat(p_vec[None, :], len(x), axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_re, pred_im = model(branch, coords)
        pred[:, j] = (pred_re + 1j * pred_im).cpu().numpy().reshape(-1)

    rel_l2 = np.zeros(len(t_values), dtype=np.float64)
    for j in range(len(t_values)):
        denom = np.linalg.norm(U_true[:, j]) + 1e-12
        rel_l2[j] = np.linalg.norm(pred[:, j] - U_true[:, j]) / denom

    return {
        "x": x,
        "t_values": t_values,
        "u_true": U_true,
        "u_pred": pred,
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


def plot_rollout_curve(rollout, output_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(rollout["t_values"], rollout["rel_l2"], color="#c2185b", linewidth=2.0)
    for t_sep in [1.0, 2.0, 3.0, 4.0]:
        plt.axvline(t_sep, color="black", linestyle=":", linewidth=1.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Global multistage : reconstruction piecewise vs solveur classique")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def train_one_stage(model, optimizer, reference, cfg_dict, t_start, t_end, stage_dir, device):
    num_epochs = int(cfg_dict["training"]["stage_num_epochs"])
    log_every = int(cfg_dict["training"]["log_every"])
    eval_every = int(cfg_dict["training"]["eval_every"])
    snapshot_every = int(cfg_dict["training"]["snapshot_every"])
    grad_clip = float(cfg_dict["training"]["grad_clip"])
    n_queries = int(cfg_dict["data"]["train_queries"])

    start_epoch, best_valid_loss = load_stage_checkpoint_if_available(model, optimizer, stage_dir, device)
    print(f"🔁 Reprise stage={os.path.basename(stage_dir)} epoch={start_epoch} best_valid={best_valid_loss:.6e}")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        batch = sample_block_batch(reference, cfg_dict, t_start, t_end, n_queries, device)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_supervised_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            print(f"[{os.path.basename(stage_dir)} | Epoch {epoch}] loss={loss.item():.3e}")

        if epoch % eval_every == 0 or epoch == num_epochs:
            valid_loss = eval_block_valid_loss(model, reference, cfg_dict, t_start, t_end, device)
            print(f"    📏 valid_loss={valid_loss:.3e}")
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name="model_latest.pth")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name="model_best.pth")
                print(f"    ✅ Nouveau meilleur valid_loss : {best_valid_loss:.3e}")

        if epoch % snapshot_every == 0:
            save_stage_checkpoint(model, optimizer, epoch, best_valid_loss, stage_dir, name=f"ckpt_epoch_{epoch:06d}.pth")

    save_stage_checkpoint(model, optimizer, num_epochs, best_valid_loss, stage_dir, name="model_final.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_global_multistage_amp_phase_alpha075_beta0_mu0_t5.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    run_dir = build_run_dir(PROJECT_DIR, cfg_dict, resume_dir=args.resume)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "rollout"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")

    reference = prepare_single_case_reference(cfg_dict)
    time_blocks = [tuple(map(float, block)) for block in cfg_dict["multistage"]["time_blocks"]]

    for stage_idx, (t_start, t_end) in enumerate(time_blocks):
        stage_dir = os.path.join(run_dir, stage_name(stage_idx, t_start, t_end))
        os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
        model = CGL_PI_DeepONet_AmpPhase(cfg_dict).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg_dict["training"]["learning_rate"]),
            weight_decay=float(cfg_dict["training"]["weight_decay"]),
        )
        print(f"\n🚧 Stage {stage_idx + 1}/{len(time_blocks)} | bloc=[{t_start}, {t_end}]")
        train_one_stage(model, optimizer, reference, cfg_dict, t_start, t_end, stage_dir, device)

    stage_models = [
        load_best_stage_model(cfg_dict, os.path.join(run_dir, stage_name(stage_idx, t_start, t_end)), device)
        for stage_idx, (t_start, t_end) in enumerate(time_blocks)
    ]
    rollout = rollout_multistage_models(stage_models, time_blocks, reference, device)
    csv_path = save_rollout_metrics(os.path.join(run_dir, "rollout"), rollout)
    plot_rollout_curve(rollout, os.path.join(run_dir, "rollout", "rollout_rel_l2.png"))
    with open(os.path.join(run_dir, "rollout", "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"final_rel_l2={float(rollout['rel_l2'][-1]):.10f}\n")
        handle.write(f"max_rel_l2={float(np.max(rollout['rel_l2'])):.10f}\n")
        handle.write(f"mean_rel_l2={float(np.mean(rollout['rel_l2'])):.10f}\n")
        handle.write(f"metrics_csv={csv_path}\n")
    print(f"\n🏁 Global multistage termine | final_rel_l2={float(rollout['rel_l2'][-1]):.4%}")


if __name__ == "__main__":
    main()
