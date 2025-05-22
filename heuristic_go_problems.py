from go_search_problem import GoProblem
import torch


BLACK = 0
WHITE = 1

def get_features(game_state):
    board_size = game_state.size
    features = []

    black_array = game_state.get_pieces_array(0)
    for row in black_array:
        features.extend(row)

    white_array = game_state.get_pieces_array(1)
    for row in white_array:
        features.extend(row)

    features.append(1 if game_state.player_to_move() == 0 else 0)

    return features

class GoProblemSimpleHeuristic(GoProblem):
    def __init__(self, state=None):
        super().__init__(state=state)

    def heuristic(self, state, player_index):
        """
        Very simple heuristic that just compares the number of pieces for each player
        
        Having more pieces (>1) than the opponent means that some were captured, capturing is generally good.
        """
        return len(state.get_pieces_coordinates(BLACK)) - len(state.get_pieces_coordinates(WHITE))

    def __str__(self) -> str:
        return "Simple Heuristic"


class GoProblemLearnedHeuristic(GoProblem):
    def __init__(self, model=None, state=None):
        super().__init__(state=state)
        self.model = model

    def encoding(self, state):
        features = get_features(state)
        return torch.tensor(features, dtype=torch.float32)

    def heuristic(self, state, player_index):
        features = self.encoding(state)
        with torch.no_grad():
            value = self.model(features)
        return value.item()

    def __str__(self) -> str:
        return "Learned Heuristic"