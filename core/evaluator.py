from abc import ABC,abstractmethod

class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(self,positions):
        pass
