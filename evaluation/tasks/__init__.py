import enum
import os
from importlib import import_module
from typing import Sequence

import yaml

from evaluation.schemas import TaskBase


class EvalMode(enum.Enum):
    CLASSIFICATION = "classification"


def load_evaluation_tasks(
    mode: EvalMode, tasks: list[str], limit: bool = False, flex: bool = False, hparam_overrides: dict = None
) -> Sequence[TaskBase]:
    """
    Load the evaluators for the given tasks.
    Args:
        mode: Evaluation mode. Classification mode is supported.

        tasks: List of tasks to evaluate the model on.
            each task can be xxx or xxx:yyy where xxx is the task name and yyy is the task config file name.
            if yyy is not provided, it will be assumed as xxx/default.yaml.

    Returns:
        List of evaluators.
    """
    evaluation_tasks = []
    tasks_dir = os.path.dirname(os.path.abspath(__file__))
    for task in tasks:
        # split by ":" to get the task name and the config file name.
        # if the config file name is not provided, it will be assumed as task/default.yaml.
        task_name = task if ":" not in task else task.split(":")[0]
        config_file_name = task.split(":")[-1] if ":" in task else "default"
        task_config_file = os.path.join(
            tasks_dir, mode.value, task_name, f"{config_file_name}.yaml"
        )

        # import the task module and check if the config file exists.
        module = import_module(f"tasks.{mode.value}.{task_name}")
        if not os.path.exists(task_config_file):
            raise FileNotFoundError(
                f"Task config file {task_config_file} does not exist."
            )

        # Load the task config file and create the task object
        with open(task_config_file, "r") as f:
            task_config = yaml.safe_load(f)

        task_type = task_config.pop("task_class")

        task_class = getattr(module, task_type)
        # Update with override
        if hparam_overrides is not None:
            for key in hparam_overrides.keys():
                task_config["hparams"][key] = hparam_overrides[key]
        process_override = {}
        if limit:
            process_override["limit"] = True 
        if flex:
            process_override["flex"] = True
        evaluation_tasks.append(task_class(**task_config, **process_override))

    return evaluation_tasks
