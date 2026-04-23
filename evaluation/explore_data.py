import fire

from schemas import ClassificationTaskBase
from tasks import load_evaluation_tasks, EvalMode


def main(task: str):
    from termcolor import colored
    try:
        evaluation_task: ClassificationTaskBase = load_evaluation_tasks(EvalMode.CLASSIFICATION, [task])[0]
    except Exception as exc:
        raise ValueError(f"Classification task {task} not found.") from exc

    print(f"Label field: {evaluation_task.label_field}")
    print(f"Required input fields: {evaluation_task.required_input_fields()}")
    for d in evaluation_task.load():
        for input_field in evaluation_task.required_input_fields():
            print(colored(f"{input_field}: {d[input_field]}", "blue"))

        print(colored(f"{evaluation_task.label_field}: {d[evaluation_task.label_field]}", "green"))
        input(f"** Press Enter to continue... **")


if __name__ == "__main__":
    fire.Fire(main)
