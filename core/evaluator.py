from abc import ABC, abstractmethod

class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self, positions):
        pass

class SequentialEvaluator(FitnessEvaluator):
    def __init__(self, objective_fn):
        self.objective_fn = objective_fn

    def evaluate(self, positions):
        return [self.objective_fn(x) for x in positions]
