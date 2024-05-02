import matplotlib.figure
import matplotlib.pyplot as plt

from .bundles import beamer_moml, jmlr

from pathlib import Path

from matplotlib.animation import TimedAnimation

targets = {
    "thesis": jmlr(),
    "talk": beamer_moml(),
}


class FigureManager:
    def __init__(self, experiments_folder: str, experiment_name: str):
        self._experiments_folder = Path(f"./{experiments_folder}")
        self._experiments_folder.mkdir(exist_ok=True)
        self._experiment_name = experiment_name
        self._targets = targets

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def targets(self) -> dict[str, dict[str, any]]:
        return self._targets

    def target_path(self, target_key: str) -> Path:
        path = self._experiments_folder / target_key / self.experiment_name
        path.mkdir(exist_ok=True, parents=True)
        return path

    def save(self, fig: matplotlib.figure.Figure, filename: str):
        for target_key in self.targets:
            plt.rcParams.update(targets[target_key])
            fig.canvas.draw()
            fig.savefig(
                self.target_path(target_key) / filename,
            )

    def save_anim(self, anim: TimedAnimation, filename: str):
        for target_key in self.targets:
            plt.rcParams.update(targets[target_key])
            anim.save(self.target_path(target_key) / filename)
