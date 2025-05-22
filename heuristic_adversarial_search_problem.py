from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from adversarial_search_problem import AdversarialSearchProblem, GameState

State = TypeVar("State", bound=GameState)
Action = TypeVar("Action")

class HeuristicAdversarialSearchProblem(AdversarialSearchProblem, ABC, Generic[Action, State]):
    """
    An extension on AdversarialSearchProblem to include a heuristic evaluation function.
    """
    @abstractmethod
    def heuristic(self, state: State) -> float:
        """
        Heuristic function for the adversarial search problem.
        Inputs:
            state: the state of the game to evaluate
        Outputs:
            heuristic value of the state
        """
        pass