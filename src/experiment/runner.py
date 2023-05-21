import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import wandb

from src.datasets.data_loaders import AbstractDataLoader
from src.experiment import config
from src.experiment.reporters import ReporterList
from src.trainers.regression import TimeSeriesRegressionTrainer
from src.utils import logs, types
from src.utils.exceptions import StopSweep
from src.utils.iterables import collect_keys_with_prefix


class WandbRunner:
    def __init__(
        self,
        runner_config: dict[str, Any],
        model_from_parameters: types.TorchParameterizedModel,
        data_loader: AbstractDataLoader,
        reporters: ReporterList,
    ):
        self.config = runner_config
        self.model_from_parameters = model_from_parameters
        self.data_loader = data_loader
        self.reporters = reporters
        self.logger = logs.get_logger(name="runner")

    @property
    def default_model_store_path(self) -> Path:
        return Path("model.pt")

    def _build_model(self, parameters: dict):
        """Build model from parameters chosen by sweep agent from available configuration list"""
        model = self.model_from_parameters(parameters)
        self.logger.info("Created model with parameters...")
        return model

    def _build_optimizer(self, model, parameters: dict):
        """
        Build optimizer from parameters

        Optimizer parameters are collected by prefix `optimizer_`, since WANDB does not support nested dicts
        and sweeping through optimizer parameters is sometimes required
        """
        optimizer_parameters = collect_keys_with_prefix(parameters, prefix="optimizer_")
        optimizer = config.build_optimizer(model, name=parameters["optimizer"], parameters=optimizer_parameters)
        self.logger.info("Created optimizer...")
        return optimizer

    def _build_loss_fn(self, parameters: dict):
        """Build loss function, handles loss parameters using prefix `loss_fn_`, similarly to `build_optimizer`"""
        loss_fn_params = collect_keys_with_prefix(parameters, prefix="loss_fn_")
        loss = config.build_loss_function(name=parameters["loss_function"], parameters=loss_fn_params)
        self.logger.info("Created loss function...")
        return loss

    def _build_callback_handler(self):
        """
        Creates callback handled from parameters
        Parameters are given directly in `sweep_config` dict, since they are not swept
        """
        callback_handler = config.build_callback_handler(
            names=self.config["callback_parameters"]["names"],
            parameters=self.config["callback_parameters"]["parameters"],
        )

        self.logger.info("Created callbacks...")
        return callback_handler

    def _build_checkpoint_handler(self):
        """
        Creates checkpoint handled from parameters
        Parameters are given directly in `sweep_config` dict, since they are not swept
        """
        checkpoint_list = config.build_checkpoint_list(
            names=self.config["checkpoint_parameters"]["names"],
            parameters=self.config["checkpoint_parameters"]["parameters"],
            restore_from=self.config["checkpoint_parameters"]["restore_from"],
        )
        self.logger.info("Created checkpoints...")
        return checkpoint_list

    def _log_trained_model(self, model) -> None:
        """Log trained model as a file to WANDB interface"""
        path = Path(wandb.run.dir) / self.default_model_store_path
        torch.save(model, path)
        self.logger.info(f"Logging model from {logs.format(path, color='yellow')} to WANDB!")
        wandb.save(str(path))

    def _run_experiment(self, parameters: dict[str, Any]) -> None:
        """Runs model training for mode with given parameter"""
        try:
            wandb.log(dict(parameters))

            model = self._build_model(parameters)
            optimizer = self._build_optimizer(model, parameters)
            loss = self._build_loss_fn(parameters)
            callback_handler = self._build_callback_handler()
            checkpoint_handler = self._build_checkpoint_handler()

            trainer = TimeSeriesRegressionTrainer(
                model=model,
                optimizer=optimizer,
                loss_function=loss,
                callback_handler=callback_handler,
                checkpoint_handler=checkpoint_handler,
                name=wandb.run.name,
                n_epochs=parameters["n_epochs"],
                device=self.config["device"],
            )

            self.logger.info("Starting training...")
            trainer.train(
                data_loader=self.data_loader.training_data_loader,
                validation_data_loader=self.data_loader.validation_data_loader,
            )
            self.logger.info(logs.format("Training finished!", color="green"))
            trainer.post_train()

            self.logger.info("Starting predicting...")
            targets, predictions = trainer.predict(self.data_loader.test_data_loader)
            self.logger.info(logs.format("Prediction finished!", color="green"))

            self.logger.info("Creating report...")
            self.reporters(
                **{
                    "model": model,
                    "targets": targets,
                    "predictions": predictions,
                    "training_summary": trainer.training_summary,
                    "data_loader": self.data_loader,
                }
            )

            self._log_trained_model(model)
            self.logger.info(logs.format("Run finished!", color="green"))

        except Exception as e:
            self.logger.error(logs.format(e, color="red"))
            self.logger.error(traceback.format_exc())
            raise StopSweep from e
        except KeyboardInterrupt:
            self.logger.info(logs.format("Stopping sweep manually!", color="red"))
            wandb.finish(exit_code=1)
            sys.exit(1)

    def run_single_experiment_in_sweep(self) -> None:
        with wandb.init(reinit=True):
            parameters = wandb.config
            self._run_experiment(parameters)

    def _start_experiment(self, experiment_config: dict[str, Any], project_name: str) -> None:
        """Stard WANDB agent executing the experiment"""
        self.logger.info(f"Starting experiment for: {logs.format(project_name, color='purple')}")
        with wandb.init(project=project_name, reinit=True):
            wandb.log(experiment_config)
            self._run_experiment(experiment_config)

    def _run_multiprocessing_experiment(
        self, experiment_config: dict[str, Any], project_name: str, n_processes: int, delay: int = 5
    ) -> None:
        """Runs sweep in multiprocessing mode starting n_processes sweeps simultaneously"""
        for process_number in range(n_processes):
            self.logger.info(f"Starting process: {logs.format(process_number, color='yellow')}")
            process = mp.Process(target=self._start_experiment, args=(experiment_config, project_name))
            process.start()
            time.sleep(delay)  # delay between starting processes to allow sweep sync information correctly

    def run_sweep(
        self,
        sweep_config: dict[str, Any],
        project_name: str,
        n_runs: int = 1,
    ) -> None:
        """
        WandbRunner public entrypoint running sweep of experiments with the given config

        :param sweep_config: configuration of the sweep run
        :param project_name: name of WANDB project to sync data to
        :param n_runs: number of total runs to conduct
        """
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function=self.run_single_experiment_in_sweep, count=n_runs, project=project_name)

    def run_experiment(
        self, experiment_config: dict[str, Any], project_name: str, multiprocessing: bool = False, n_processes: int = 1
    ) -> None:
        """
        WandbRunner public entrypoint which runs experiment with given config

        :param experiment_config: config of experiment
        :param project_name: WANDB project name to sync to
        :param multiprocessing: if True use multiprocessing
        :param n_processes: number of processes to run in parallel
        """
        if multiprocessing:
            self._run_multiprocessing_experiment(experiment_config, project_name, n_processes)
        else:
            with wandb.init(project=project_name, reinit=True):
                wandb.log(experiment_config)
                self._run_experiment(experiment_config)
