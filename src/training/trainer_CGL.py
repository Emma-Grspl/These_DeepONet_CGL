from src.training.trainer_CGL_modern import *  # noqa: F401,F403
from src.training import trainer_CGL_modern as _modern
from src.training import trainer_CGL_legacy as _legacy


def get_training_logic_name(cfg):
    training_cfg = cfg["training"] if isinstance(cfg, dict) else cfg.training
    return training_cfg.get("logic_variant", "modern")


def get_training_mode(cfg):
    training_cfg = cfg["training"] if isinstance(cfg, dict) else cfg.training
    return training_cfg.get("training_mode", "navigator")


def train_model(model, cfg, explicit_resume_path=None):
    logic_name = get_training_logic_name(cfg)
    training_mode = get_training_mode(cfg)
    backend = _legacy if logic_name == "legacy" else _modern

    if training_mode == "global_direct":
        if not hasattr(backend, "train_global_direct"):
            raise ValueError(f"Training mode '{training_mode}' non supporté pour logic_variant='{logic_name}'.")
        return backend.train_global_direct(model, cfg, explicit_resume_path=explicit_resume_path)
    return backend.train_navigator(model, cfg, explicit_resume_path=explicit_resume_path)


def train_navigator(model, cfg, explicit_resume_path=None):
    return train_model(model, cfg, explicit_resume_path=explicit_resume_path)
