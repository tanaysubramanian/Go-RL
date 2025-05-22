import numpy as np
try:
    import pyspiel
except:
    print("pyspiel not found, using PyGo instead")
    from pygo.game import Game
    from pygo.utils import Stone
import pygame
import sys
import copy


def create_go_game(size):
    """
    load open-spiel game with provided size
    """
    if size == 5:
        komi = 0.5
    elif size == 9:
        komi = 5.5
    else:
        komi = 7.5
    try:
        game = pyspiel.load_game("go", {"board_size": size, "komi": komi})
        state = game.new_initial_state()
    except:
        state = PyGoInternalState(size, komi)
    return state


class PyGoInternalState:
    """
    Implementation of internal Go game state matching pyspiel's interface.
    """
    def __init__(self, size=5, komi=7.5):
        self.config = {
            'board_size': size,
            'black_stone': '●',
            'white_stone': '○',
            'enable_self_destruct': False,
            'komi': komi
        }
        self.game = Game(self.config)
        self.size = size
        self.consecutive_passes = 0
        self._current_player = 0  # 0 for BLACK, 1 for WHITE
        
    def current_player(self) -> int:
        return self._current_player

    def observation_tensor(self, player=0) -> list:
        board = np.array(self.game.board)
        tensor = np.zeros((4, self.size, self.size), dtype=np.int32)

        black_channel = (board == Stone.BLACK)
        tensor[0] = black_channel
        
        white_channel = (board == Stone.WHITE)
        tensor[1] = white_channel
        
        empty_channel = (board == Stone.EMPTY)
        tensor[2] = empty_channel
        
        tensor[3] += self._current_player

        return tensor.flatten()

    def returns(self) -> list:
        if not self.is_terminal():
            return [0, 0]

        scores = self.game.get_scores()
        black_score = scores[Stone.BLACK]
        white_score = scores[Stone.WHITE]
        
        if black_score > white_score:
            return [1, -1]
        else:
            return [-1, 1]

    def clone(self):
        return copy.deepcopy(self)

    def is_terminal(self) -> bool:
        """
        Game ends after two consecutive passes or when board is full
        """
        board = np.array(self.game.board)
        empty_count = np.sum(board == Stone.EMPTY)
        return self.consecutive_passes >= 2 or empty_count == 0

    def legal_actions(self) -> list:
        if self.is_terminal():
            return []

        legal = []
        board = np.array(self.game.board)
        empty_positions = np.where(board == Stone.EMPTY)
        stone = Stone.BLACK if self._current_player == 0 else Stone.WHITE
        
        # Try each empty position
        for y, x in zip(empty_positions[0], empty_positions[1]):
            action = y * self.size + x
            try:
                test_state = self.clone()
                if stone == Stone.BLACK:
                    test_state.game.place_black(y, x)
                else:
                    test_state.game.place_white(y, x)
                legal.append(action)
            except:
                continue
        
        # Pass is always legal
        pass_move = self.size * self.size
        legal.append(pass_move)
        
        return [int(l) for l in legal]

    def apply_action(self, action: int):
        if self.is_terminal():
            return
            
        pass_move = self.size * self.size
        
        if action == pass_move:
            self.consecutive_passes += 1
            self.game.pass_turn()
        else:
            y = action // self.size
            x = action % self.size
            
            if self._current_player == 0:
                self.game.place_black(y, x)
            else:
                self.game.place_white(y, x)
            self.consecutive_passes = 0
        self._current_player = 1 - self._current_player