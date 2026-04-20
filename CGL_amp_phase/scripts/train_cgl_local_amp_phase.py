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

from src.data.local_operator_amp_phase import (
    prepare_single_case_trajectory,
    rollout_local_model,
    sample_training_batch,
    save_rollout_metrics,
    split_pair_indices,
)
from src.models.cgl_local_deeponet_amp_phase import CGL_LocalDirect_DeepONet_AmpPhase


class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return self._dict[item]

    def get(self, key, default=None):
        return self._dict.get(key, default)


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(model, optimizer, epoch, best_rollout_l2, run_dir, name="model_latest.pth"):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_rollout_l2": best_rollout_l2,
    }
    atomic_torch_save(state, os.path.join(ckpt_dir, name))


def load_checkpoint_if_available(model, optimizer, run_dir, device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "model_latest.pth")
    if not os.path.exists(ckpt_path):
        return 0, float("inf")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_rollout_l2", float("inf")))


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


def compute_losses(model, batch, amp_floor):
    delta_log_amp, delta_phase = model(batch["branch"], batch["x_query"])
    next_amp = torch.exp(torch.log(batch["current_amp"] + amp_floor) + delta_log_amp) - amp_floor
    next_phase = batch["current_phase"] + delta_phase
    pred_re = next_amp * torch.cos(next_phase)
    pred_im = next_amp * torch.sin(next_phase)

    loss_complex = torch.mean((pred_re - batch["target_u_re"]) ** 2 + (pred_im - batch["target_u_im"]) ** 2)
    loss_amp = torch.mean((next_amp - batch["target_amp"]) ** 2)
    loss_delta = torch.mean((delta_log_amp - batch["target_delta_log_amp"]) ** 2 + (delta_phase - batch["target_delta_phase"]) ** 2)
    return loss_complex, loss_amp, loss_delta


def plot_rollout_curve(rollout, output_path):
    plt.figure(figsize=(8, 4.5))
    plt.plot(rollout["t_values"], rollout["rel_l2"], color="#c2185b", linewidth=2.0)
    plt.xlabel("Temps t")
    plt.ylabel("Erreur L2 relative")
    plt.title("Rollout local direct : erreur relative vs solveur classique")
    plt.grid(alpha=0.25)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_local_direct_amp_phase_t5.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)

    run_dir = build_run_dir(PROJECT_DIR, cfg_dict, resume_dir=args.resume)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "rollout"), exist_ok=True)

    cfg = ConfigObj(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    print(f"📂 Run dir : {run_dir}")

    trajectory = prepare_single_case_trajectory(cfg_dict)
    train_idx, valid_idx = split_pair_indices(trajectory, float(cfg_dict["training"]["train_split"]))

    model = CGL_LocalDirect_DeepONet_AmpPhase(cfg_dict).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_dict["training"]["learning_rate"]),
        weight_decay=float(cfg_dict["training"]["weight_decay"]),
    )
    start_epoch, best_rollout_l2 = load_checkpoint_if_available(model, optimizer, run_dir, device)
    print(f"🔁 Reprise epoch={start_epoch} | best_rollout_l2={best_rollout_l2:.6e}")

    amp_floor = float(cfg_dict["local_operator"]["amp_floor"])
    weights = cfg_dict["training"]["loss_weights"]
    num_epochs = int(cfg_dict["training"]["num_epochs"])
    log_every = int(cfg_dict["training"]["log_every"])
    eval_every = int(cfg_dict["training"]["eval_every"])
    snapshot_every = int(cfg_dict["training"]["snapshot_every"])
    grad_clip = float(cfg_dict["training"]["grad_clip"])

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        batch = sample_training_batch(
            trajectory,
            cfg_dict,
            pair_indices=train_idx,
            batch_pairs=int(cfg_dict["training"]["batch_pairs"]),
            batch_queries=int(cfg_dict["training"]["batch_queries"]),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_complex, loss_amp, loss_delta = compute_losses(model, batch, amp_floor)
        loss = (
            float(weights["complex"]) * loss_complex
            + float(weights["amplitude"]) * loss_amp
            + float(weights["delta"]) * loss_delta
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"[Epoch {epoch}] loss={loss.item():.3e} "
                f"| complex={loss_complex.item():.3e} "
                f"| amp={loss_amp.item():.3e} "
                f"| delta={loss_delta.item():.3e}"
            )

        if epoch % eval_every == 0 or epoch == num_epochs:
            rollout = rollout_local_model(model, trajectory, cfg_dict, device)
            csv_path = save_rollout_metrics(os.path.join(run_dir, "rollout"), rollout)
            plot_rollout_curve(rollout, os.path.join(run_dir, "rollout", "rollout_rel_l2.png"))
            final_l2 = float(rollout["rel_l2"][-1])
            print(f"    🌍 Rollout final L2(t_max)={final_l2:.4%} | csv={csv_path}")
            save_checkpoint(model, optimizer, epoch, best_rollout_l2, run_dir, name="model_latest.pth")
            if final_l2 < best_rollout_l2:
                best_rollout_l2 = final_l2
                save_checkpoint(model, optimizer, epoch, best_rollout_l2, run_dir, name="model_best.pth")
                print(f"    ✅ Nouveau meilleur rollout : {best_rollout_l2:.4%}")

        if epoch % snapshot_every == 0:
            save_checkpoint(model, optimizer, epoch, best_rollout_l2, run_dir, name=f"ckpt_epoch_{epoch:06d}.pth")

    save_checkpoint(model, optimizer, num_epochs, best_rollout_l2, run_dir, name="model_final.pth")
    print(f"🏁 Entraînement terminé | best_rollout_l2={best_rollout_l2:.4%}")


if __name__ == "__main__":
    main()
