import sys
import os
import argparse
import yaml
import torch
import glob
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.cgl_deeponet_amp_phase import CGL_PI_DeepONet_AmpPhase
from src.training.trainer_CGL import train_navigator


class ConfigObj:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, item):
        return self._dict[item]

    def get(self, key, default=None):
        return self._dict.get(key, default)


def find_latest_run_dir(base_results_dir):
    search_path = os.path.join(base_results_dir, "CGL_*")
    all_runs = glob.glob(search_path)
    all_runs = [d for d in all_runs if os.path.isdir(d)]
    if not all_runs:
        return None
    return max(all_runs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cgl_single_case_amp_phase.yaml")
    parser.add_argument("--resume", nargs='?', const="latest", default=None)
    args = parser.parse_args()

    results_root = os.path.join(project_root, "results")
    os.makedirs(results_root, exist_ok=True)

    if args.resume:
        if args.resume == "latest":
            run_dir = find_latest_run_dir(results_root)
            if not run_dir:
                raise ValueError("❌ Aucun run précédent trouvé pour --resume latest !")
        else:
            run_dir = args.resume
            if not os.path.exists(run_dir):
                run_dir = os.path.join(results_root, args.resume)
                if not os.path.exists(run_dir):
                    raise ValueError(f"❌ Dossier introuvable : {args.resume}")
        print(f"🔄 MODE REPRISE ACTIVÉ : {run_dir}")
        print("   (Le script cherchera le dernier checkpoint valide dans ce dossier)")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"CGL_AmpPhase_Run_{timestamp}"
        run_dir = os.path.join(results_root, run_name)
        print(f"🚀 NOUVEAU START : {run_dir}")

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(args.config, "r") as f:
        yaml_data = yaml.safe_load(f)
    yaml_data["training"]["save_dir"] = ckpt_dir
    cfg = ConfigObj(yaml_data)
    logic_variant = cfg["training"].get("logic_variant", "modern")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Device : {device}")
    print(f"🧠 Training logic : {logic_variant}")

    model = CGL_PI_DeepONet_AmpPhase(cfg).to(device)

    try:
        train_navigator(model, cfg, explicit_resume_path=None)
        torch.save(model.state_dict(), os.path.join(run_dir, "model_final_cgl_amp_phase.pth"))
        print("\n✅ Terminé avec succès !")
    except Exception as e:
        print(f"\n❌ Erreur Critique : {e}")
        raise e


if __name__ == "__main__":
    main()
