from typing import Optional
import wandb


class TrainMonitor:

    def __init__(
        self,
        project: str,
        model_name: str,
        version: str = "v1",
        experiment: str = "baseline",
        mode: str = "online",
        config: Optional[dict] = None,
        tags: Optional[list] = None,
        notes: str = "",
        finish_previous: bool = True,
    ):

        self.project = project
        self.model_name = model_name
        self.version = version
        self.experiment = experiment

        self.run = wandb.init(
            project=project,
            name=f"{model_name}_{version}_{experiment}",
            config=config,
            tags=tags,
            notes=notes,
            mode=mode,
            reinit=finish_previous,
        )

    def monitor(self, metrics: dict, step: Optional[int] = None):
        """
        Log any metrics
        """
        wandb.log(metrics, step=step)

    def update_config(self, params: dict):
        wandb.config.update(params, allow_val_change=True)

    def watch(self, model):
        wandb.watch(model)

    def finish(self):
        wandb.finish()