from abc import ABC, abstractmethod
from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem
from search_algorithms import minimax, alpha_beta
from adversarial_search_problem import GameState, Action

class GameAgent(ABC):
    @abstractmethod
    def get_move(self, state: GameState) -> Action:
        pass

class MinimaxAgent(GameAgent):
    def __init__(self, searchproblem: HeuristicAdversarialSearchProblem, cutoff_depth=3):
        self.searchproblem = searchproblem
        self.cutoff_depth = cutoff_depth

    def get_move(self, state: GameState) -> Action:
        self.searchproblem.set_start_state(state)
        action, _ = minimax(self.searchproblem, cutoff_depth=self.cutoff_depth)
        return action

class AlphaBetaAgent(GameAgent):
    def __init__(self, searchproblem: HeuristicAdversarialSearchProblem, cutoff_depth=3):
        self.searchproblem = searchproblem
        self.cutoff_depth = cutoff_depth

    def get_move(self, state: GameState) -> Action:
        self.searchproblem.set_start_state(state)
        action, _ = alpha_beta(self.searchproblem, cutoff_depth=self.cutoff_depth)
        return action