import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
