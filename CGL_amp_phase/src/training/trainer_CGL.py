from src.training.trainer_CGL_modern import *  # noqa: F401,F403
from src.training import trainer_CGL_modern as _modern
from src.training import trainer_CGL_legacy as _legacy


def get_training_logic_name(cfg):
    training_cfg = cfg["training"] if isinstance(cfg, dict) else cfg.training
    return training_cfg.get("logic_variant", "modern")


def train_navigator(model, cfg, explicit_resume_path=None):
    logic_name = get_training_logic_name(cfg)
    if logic_name == "legacy":
        return _legacy.train_navigator(model, cfg, explicit_resume_path=explicit_resume_path)
    return _modern.train_navigator(model, cfg, explicit_resume_path=explicit_resume_path)
