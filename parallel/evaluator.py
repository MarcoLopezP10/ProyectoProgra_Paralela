"""parallel.evaluator

Evaluator implementations that enable concurrency / parallelism.

"""

from options.evaluator import FitnessEvaluator, SequentialEvaluator

__all__ = ["FitnessEvaluator", "SequentialEvaluator"]
