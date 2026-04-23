import abc
from collections import Counter

from src.classifier_models.base import SafetyClassifierBase


class TaskBase(abc.ABC):
    def __init__(self, **kwargs):
        """
        Initialize the task for evaluation.
        """
        self.data = self.put_id_on_data(self.load())
        assert len(self.data) > 0, "No data loaded for the task."
        assert all("id" in d for d in self.data), "All data must have an 'id' field."

    def put_id_on_data(self, data: list[dict]) -> list[dict]:
        """
        Put an 'id' field on the data.
        """
        return [{"id": f"{self.__class__.__name__}/{i}", **d} for i, d in enumerate(data)]

    @abc.abstractmethod
    def required_input_fields(self) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self) -> list[dict]:
        raise NotImplementedError

    @abc.abstractmethod
    def _evaluate(self, model: SafetyClassifierBase) -> tuple[dict, list[dict]]:
        raise NotImplementedError

    def evaluate(self, model: SafetyClassifierBase) -> tuple[dict, list[dict]]:
        report, individual_results = self._evaluate(model)
        if getattr(self, "thinker_eval", None) is not None and len(self.thinker_eval) > 1:
            for v in individual_results:
                assert all("id" in d for d in individual_results[v]), (
                    "All individual results must have an 'id' field."
                )
                assert len(set(d["id"] for d in individual_results[v])) == len(individual_results[v]), (
                    "All individual results must have unique 'id' field."
                )
        else:
            assert all("id" in d for d in individual_results), (
                "All individual results must have an 'id' field."
            )
            assert len(set(d["id"] for d in individual_results)) == len(individual_results), (
                "All individual results must have unique 'id' field."
            )
        return report, individual_results


class ClassificationTaskBase(TaskBase, abc.ABC):
    @abc.abstractmethod
    def required_output_fields(self) -> list[list[str]]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def label_field(self) -> str:
        raise NotImplementedError

    def display_data_stats(self):
        print(f"Field: {self.label_field}")
        counter = Counter([d[self.label_field] for d in self.data])
        for key, value in counter.items():
            print(f"{key}: {value}")

    def validate_if_model_is_compatible(self, model: SafetyClassifierBase) -> bool:
        input_compatible = all(
            field in model.get_possible_input_fields()
            for field in self.required_input_fields()
        )
        assert input_compatible, (
            f"Model input fields {model.get_possible_input_fields()} "
            f"are not compatible with the task input fields {self.required_input_fields()}"
        )
        output_compatible = any(
            all(field in model.get_output_fields() for field in fieldset)
            for fieldset in self.required_output_fields()
        )
        assert output_compatible, (
            f"Model output fields {model.get_output_fields()} "
            f"are not compatible with the task output fields {self.required_output_fields()}"
        )
        return input_compatible and output_compatible

    @abc.abstractmethod
    def _evaluate(self, model: SafetyClassifierBase) -> tuple[dict, list[dict]]:
        raise NotImplementedError
