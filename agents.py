from go_search_problem import GoProblem, GoState, Action
from adversarial_search_problem import GameState
from heuristic_go_problems import *
import random
from abc import ABC, abstractmethod
import numpy as np
import time
from game_runner import run_many
from torch import nn
from models import ValueNetwork, load_model

from search_algorithms import minimax, alpha_beta


MAXIMIZER = 0
MIMIZER = 1

class GameAgent():
    # Interface for Game agents
    @abstractmethod
    def get_move(self, game_state: GameState, time_limit: float) -> Action:
        # Given a state and time limit, return an action
        pass


class RandomAgent(GameAgent):
    # An Agent that makes random moves

    def __init__(self):
        self.search_problem = GoProblem()

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get random move for a given state
        """
        actions = self.search_problem.get_available_actions(game_state)
        return random.choice(actions)

    def __str__(self):
        return "RandomAgent"


class GreedyAgent(GameAgent):
    def __init__(self, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        get move of agent for given game state.
        Greedy agent looks one step ahead with the provided heuristic and chooses the best available action
        (Greedy agent does not consider remaining time)

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        """
        # Create new GoSearchProblem with provided heuristic
        search_problem = self.search_problem

        # Player 0 is maximizing
        if game_state.player_to_move() == MAXIMIZER:
            best_value = -float('inf')
        else:
            best_value = float('inf')
        best_action = None

        # Get Available actions
        actions = search_problem.get_available_actions(game_state)
        random.shuffle(actions)
        # Compare heuristic of every reachable next state
        for action in actions:
            new_state = search_problem.transition(game_state, action)
            value = search_problem.heuristic(new_state, new_state.player_to_move())
            if game_state.player_to_move() == MAXIMIZER:
                if value > best_value:
                    best_value = value
                    best_action = action
            else:
                if value < best_value:
                    best_value = value
                    best_action = action

        # Return best available action
        return best_action

    def __str__(self):
        """
        Description of agent (Greedy + heuristic/search problem used)
        """
        return "GreedyAgent + " + str(self.search_problem)


class MinimaxAgent(GameAgent):
    def __init__(self, depth_cutoff=2, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using minimax algorithm

        MiniMaxAgents should not consider time limit, they simply search to their specified depth_cutoff
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        self.search_problem.set_start_state(game_state)
        action, _ = minimax(self.search_problem, cutoff_depth=self.depth)
        return action

    def __str__(self):
        return f"MinimaxAgent w/ depth {self.depth} + " + str(self.search_problem)


class AlphaBetaAgent(GameAgent):
    def __init__(self, depth_cutoff=2, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.depth = depth_cutoff
        self.search_problem = search_problem

    def get_move(self, game_state: GoState, time_limit: float) -> Action:
        """
        Get move of agent for given game state using alpha-beta algorithm

        AlphaBetaAgents should not consider time limit, they simply search to their specified depth_cutoff
        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        self.search_problem.set_start_state(game_state)
        action, _ = alpha_beta(self.search_problem, cutoff_depth=self.depth)
        return action

    def __str__(self):
        return f"AlphaBeta w/ depth {self.depth} + " + str(self.search_problem)


class IterativeDeepeningAgent(GameAgent):
    def __init__(self, cutoff_time=1, search_problem=GoProblemSimpleHeuristic()):
        super().__init__()
        self.cutoff_time = cutoff_time
        self.search_problem = search_problem

    def get_move(self, game_state, time_limit):
        """
        Get move of agent for given game state using iterative deepening algorithm (+ alpha-beta).
        Iterative deepening is a search algorithm that repeatedly searches for a solution to a problem,
        increasing the depth of the search with each iteration.

        The advantage of iterative deepening is that you can stop the search based on the time limit, rather than depth.
        The recommended approach is to modify your implementation of Alpha-beta to stop when the time limit is reached
        and run IDS on that modified version.

        Args:
            game_state (GameState): current game state
            time_limit (float): time limit for agent to return a move
        Returns:
            best_action (Action): best action for current game state
        """
        self.search_problem.set_start_state(game_state)
        start_time = time.time()
        time_buffer = 0.01  # safety cushion to avoid timeout

        best_move = None
        depth = 1

        while True:
            time_remaining = time_limit - (time.time() - start_time)
            if time_remaining <= time_buffer:
                break

            try:
                move, _ = self.alpha_beta_time_limited(depth, time_remaining - time_buffer, start_time)
                if move is not None:
                    best_move = move
            except TimeoutError:
                break

            depth += 1

        return best_move
    
    def alpha_beta_time_limited(self, depth, time_limit, start_time):
        """
        Alpha-beta pruning algorithm that respects a time limit
        
        Args:
            depth: Maximum depth to search
            time_limit: Maximum allowed search time
            start_time: Time when the search started
        
        Returns:
            (move, value): The best move and its value
        
        Raises:
            TimeoutError: If the search exceeds the time limit
        """
        def check_time():
            if time.time() - start_time > time_limit:
                raise TimeoutError()
        
        def is_valid_action(state, action):
            try:
                next_state = self.search_problem.transition(state, action)
                return True
            except Exception:
                return False
        
        def ab_search(state, depth, alpha, beta, maximizing):
            check_time()
            
            if depth == 0 or self.search_problem.is_terminal_state(state):
                return None, self.search_problem.heuristic(state, state.player_to_move())
            
            best_action = None
            actions = self.search_problem.get_available_actions(state)
            
            if maximizing:
                value = float('-inf')
                for action in actions:
                    if not is_valid_action(state, action):
                        continue
                        
                    next_state = self.search_problem.transition(state, action)
                    _, child_value = ab_search(next_state, depth - 1, alpha, beta, False)
                    
                    if child_value > value:
                        value = child_value
                        best_action = action
                    
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break
                        
                return best_action, value
            else:
                value = float('inf')
                for action in actions:
                    if not is_valid_action(state, action):
                        continue
                        
                    next_state = self.search_problem.transition(state, action)
                    _, child_value = ab_search(next_state, depth - 1, alpha, beta, True)
                    
                    if child_value < value:
                        value = child_value
                        best_action = action
                    
                    beta = min(beta, value)
                    if beta <= alpha:
                        break
                        
                return best_action, value
        
        # If we can't find any valid moves, return the pass move
        try:
            return ab_search(
                self.search_problem.start_state, 
                depth, 
                float('-inf'), 
                float('inf'),
                self.search_problem.start_state.player_to_move() == MAXIMIZER
            )
        except Exception as e:
            # If something goes wrong, return the pass move as a fallback
            pass_move = self.search_problem.start_state.size * self.search_problem.start_state.size
            return pass_move, 0
    
class MCTSNode:
    def __init__(self, state, parent=None, children=None, action=None):
        # GameState for Node
        self.state = state

        # Parent (MCTSNode)
        self.parent = parent
        
        # Children List of MCTSNodes
        if children is None:
            children = []
        self.children = children
        
        # Number of times this node has been visited in tree search
        self.visits = 0
        
        # Value of node (number of times simulations from children results in black win)
        self.value = 0
        
        # Action that led to this node
        self.action = action

    def __hash__(self):
        """
        Hash function for MCTSNode is hash of state
        """
        return hash(self.state)
    
class MCTSAgent(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        """
        Args: 
            c (float): exploration constant of UCT algorithm
        """
        super().__init__()
        self.c = c
        self.search_problem = GoProblem()

    def get_move(self, state, time_limit):
        """
        Selects the best move using MCTS within the given time limit.
        """
        root_node = MCTSNode(state)
        start_time = time.time()
        num_simulations = 0

        # Keep running simulations until out of time
        while time.time() - start_time < 1.0:
            leaf_node = self.select(root_node)
            expanded_node = self.expand(leaf_node)
            simulation_result = self.simulate(expanded_node)
            self.backpropagate(expanded_node, simulation_result)
            num_simulations += 1

        # Pick move with most visits
        chosen_action = None
        most = 0
        for child_node in root_node.children:
            if child_node.visits > most and child_node.action in state.legal_actions():
                most = child_node.visits
                chosen_action = child_node.action

        return chosen_action
    
    def select(self, node):
        """
        Traverse the tree by following highest UCB until a leaf is found.
        """
        while not self.search_problem.is_terminal_state(node.state):
            available_actions = node.state.legal_actions()

            if len(node.children) < len(available_actions):
                return node

            # Pick child with best UCB score
            best_child = None
            best_ucb = -float('inf')
            for child in node.children:
                avg_value = child.value / child.visits
                exploration_term = self.c * np.sqrt(np.log(node.visits) / child.visits)
                ucb = avg_value + exploration_term

                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            node = best_child

        return node
    
    def expand(self, node):
        """
        Expand one unexplored action from the given node.
        """
        if self.search_problem.is_terminal_state(node.state):
            return node

        unvisited = set(node.state.legal_actions()) - {child.action for child in node.children}
        if not unvisited:
            return node

        chosen_move = random.choice(list(unvisited))
        next_state = self.search_problem.transition(node.state, chosen_move)
        new_child = MCTSNode(state=next_state, parent=node, action=chosen_move)
        node.children.append(new_child)

        return new_child
    
    def simulate(self, node):
        """
        Simulate a random playout starting from the given node.
        """
        output = node.state

        while not self.search_problem.is_terminal_state(output):
            possibles = output.legal_actions()
            move = random.choice(possibles)
            output = self.search_problem.transition(output, move)

        outcome = self.search_problem.get_result(output)

        return outcome
    
    def backpropagate(self, node, result):
        """
        Update stats along the path from the given node to the root.
        """
        while node is not None:
            node.visits += 1
            if (node.state.player_to_move() == 1 and result > 0) or (node.state.player_to_move() == 0 and result < 0):
                node.value += 1
            node = node.parent

class SolidAgent(MCTSAgent):
    """
    An improved MCTS agent with smarter playouts.
    Uses heuristics during simulation to prioritize captures, liberties, and extensions.
    """
    def __init__(self, explore_const=np.sqrt(2)):
        super().__init__(explore_const)
        self.problem = GoProblem()

    def __str__(self):
        return "ImprovedMCTSAgent"

    def get_move(self, state, time_limit):
        # Main MCTS loop with time cutoff
        root = MCTSNode(state)
        deadline = time.time() + time_limit - 0.05

        while time.time() < deadline:
            leaf = self.select(root)
            child = self.expand(leaf)
            result = self.simulate(child, deadline)
            self.backpropagate(child, result)

        legal = state.legal_actions()
        candidates = [c for c in root.children if c.action in legal]
        if candidates:
            return max(candidates, key=lambda c: c.visits).action
        return random.choice(legal)

    def simulate(self, node, deadline):
        # Heuristic for captures
        sim_state = node.state
        my_color = sim_state.player_to_move()
        opp_color = 1 - my_color

        while not self.problem.is_terminal_state(sim_state):
            if time.time() >= deadline:
                break

            acts = sim_state.legal_actions()

            caps = []
            for a in acts:
                nxt = self.problem.transition(sim_state, a)
                opp_before = sum(
                    1 for i in range(sim_state.size)
                    for j in range(sim_state.size)
                    if sim_state.get_pieces_array(opp_color)[i][j]
                )
                opp_after = sum(
                    1 for i in range(nxt.size)
                    for j in range(nxt.size)
                    if nxt.get_pieces_array(opp_color)[i][j]
                )
                if opp_after < opp_before:
                    caps.append(a)
            if caps:
                move = random.choice(caps)
                sim_state = self.problem.transition(sim_state, move)
                continue

            libs = []
            for a in acts:
                nxt = self.problem.transition(sim_state, a)
                my_before = sum(
                    1 for i in range(sim_state.size)
                    for j in range(sim_state.size)
                    if sim_state.get_pieces_array(my_color)[i][j]
                )
                my_after = sum(
                    1 for i in range(nxt.size)
                    for j in range(nxt.size)
                    if nxt.get_pieces_array(my_color)[i][j]
                )
                if my_after > my_before:
                    libs.append(a)
            if libs:
                move = random.choice(libs)
                sim_state = self.problem.transition(sim_state, move)
                continue

            exts = []
            for a in acts:
                x, y = a % sim_state.size, a // sim_state.size
                if any(
                    sim_state.get_pieces_array(my_color)[nx][ny]
                    for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                    if 0 <= nx < sim_state.size and 0 <= ny < sim_state.size
                ):
                    exts.append(a)
            if exts:
                move = random.choice(exts)
                sim_state = self.problem.transition(sim_state, move)
                continue

            # fallback to random move if no heuristics apply
            move = random.choice(acts)
            sim_state = self.problem.transition(sim_state, move)

        return self.problem.get_result(sim_state)
    
class Book(GameAgent):
    """
    Agent that plays the center move on the first turn, then defers to a fallback agent.
    """
    def __init__(self, fallback_agent):
        self.fallback = fallback_agent
        # Compute center move index
        self.first_move = self.fallback.problem.start_state.size // 2 \
                          + (self.fallback.problem.start_state.size // 2) * self.fallback.problem.start_state.size

    def __str__(self):
        return f"OpeningBook({self.first_move}) + {self.fallback}"

    def get_move(self, state, time_limit):
        # Play center move if board is empty, else defer to fallback agent
        blacks = len(state.get_pieces_coordinates(0))
        whites = len(state.get_pieces_coordinates(1))
        if blacks + whites == 0:
            return self.first_move
        return self.fallback.get_move(state, time_limit)

def create_value_agent_from_model():
    """
    Create agent object from saved model. This (or other methods like this) will be how your agents will be created in gradescope and in the final tournament.
    """

    model_path = "value_model.pt"
    # TODO: Update number of features for your own encoding size
    feature_size = 51
    model = load_model(model_path, ValueNetwork(feature_size))
    heuristic_search_problem = GoProblemLearnedHeuristic(model)

    # TODO: Try with other heuristic agents (IDS/AB/Minimax)
    learned_agent = AlphaBetaAgent(depth_cutoff=2, search_problem=heuristic_search_problem)

    return learned_agent

class MegaAgent(GameAgent):
    """
    MCTS agent with an opening book and custom territory-based rollout.
    Reuses tree root between moves for efficiency.
    """
    def __init__(self, c=np.sqrt(2)):
        self.search_problem = GoProblemSimpleHeuristic()
        self.c = c
        self.opening_book = {0: 12, 1: 6, 2: 18}
        self.prev_root = None  # reuse previous search tree

    def __str__(self):
        return "MegaAgent MCTS"

    def get_move(self, state, time_limit):
        # Check if opening book has a predefined move
        move_count = sum(len(state.get_pieces_coordinates(p)) for p in [0, 1])
        if move_count in self.opening_book:
            center = self.opening_book[move_count]
            if center in state.legal_actions():
                return center

        if self.prev_root and self.prev_root.state == state:
            root = self.prev_root
        else:
            root = MegaNode(state)
        self.prev_root = root

        start = time.time()
        while time.time() - start < time_limit - 0.01:
            leaf = self.select(root)
            child = self.expand(leaf)
            result = self.simulate(child)
            self.backpropagate(child, result)

        best = max(root.children, key=lambda c: c.visits, default=None)
        return best.action if best else random.choice(state.legal_actions())

    def select(self, node):
        while not self.search_problem.is_terminal_state(node.state):
            if len(node.children) < len(node.state.legal_actions()):
                return node
            node = node.best_child(self.c)
        return node

    def expand(self, node):
        legal = node.state.legal_actions()
        used = {child.action for child in node.children}
        unvisited = [a for a in legal if a not in used]
        if not unvisited:
            return node
        action = random.choice(unvisited)
        next_state = self.search_problem.transition(node.state, action)
        child = MegaNode(next_state, parent=node, action=action)
        node.children.append(child)
        return child

    def simulate(self, node):
        # Do a territory-based rollout
        state = node.state
        while not self.search_problem.is_terminal_state(state):
            legal = state.legal_actions()
            if not legal:
                break
            heuristics = []
            for a in legal:
                try:
                    next_state = self.search_problem.transition(state, a)
                    h = self.territory_heuristic(next_state)
                except:
                    h = -1000
                heuristics.append(h)
            # Softmax sampling based on territory heuristic    
            exp_vals = np.exp(heuristics - np.max(heuristics))
            probs = exp_vals / np.sum(exp_vals)
            action = np.random.choice(legal, p=probs)
            state = self.search_problem.transition(state, action)
        result = self.search_problem.get_result(state)
        return result if node.state.player_to_move() == MAXIMIZER else -result

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

    def territory_heuristic(self, state):
        board = state.get_board()
        score = 0
        for i in range(state.size):
            for j in range(state.size):
                if board[0][i][j]:
                    score += 1
                elif board[1][i][j]:
                    score -= 1
        return score

class MegaNode:
    """
    Node for MegaAgent's custom MCTS with UCB1 scoring.
    """
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    def best_child(self, c):
        best_score, best = float('-inf'), None
        for child in self.children:
            if child.visits == 0:
                continue
            exploit = child.value / child.visits
            explore = c * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        return best
    
class OptimizedMCTSAgent(GameAgent):
    """
    MCTS agent with time-based rollout heuristics: explores early and favors captures later.
    """
    def __init__(self, c=np.sqrt(2)):
        self.c = c
        self.search_problem = GoProblem()
    
    def __str__(self):
        return "OptimizedMCTSAgent"

    def get_move(self, state, time_limit):
        root_node = MCTSNode(state)
        start_time = time.time()

        while time.time() - start_time < time_limit - 0.05:
            leaf = self.select(root_node)
            child = self.expand(leaf)
            reward = self.simulate(child, move_num=self._move_number(state))
            self.backpropagate(child, reward)

        # Pick move with most visits
        legal = state.legal_actions()
        best = max(root_node.children, key=lambda c: c.visits if c.action in legal else -1, default=None)
        return best.action if best else random.choice(legal)

    def _move_number(self, state):
        return len(state.get_pieces_coordinates(0)) + len(state.get_pieces_coordinates(1))

    def select(self, node):
        # Traverse using UCB1 selection
        while not self.search_problem.is_terminal_state(node.state):
            if len(node.children) < len(node.state.legal_actions()):
                return node
            node = max(node.children, key=lambda c: (c.value / (1 + c.visits)) +
                       self.c * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-6)))
        return node

    def expand(self, node):
        used = {child.action for child in node.children}
        options = [a for a in node.state.legal_actions() if a not in used]
        if not options:
            return node
        move = random.choice(options)
        child_state = self.search_problem.transition(node.state, move)
        child = MCTSNode(child_state, parent=node, action=move)
        node.children.append(child)
        return child

    def simulate(self, node, move_num):
        # Heuristic of explore early, capture late
        state = node.state
        while not self.search_problem.is_terminal_state(state):
            legal = state.legal_actions()
            if not legal:
                break
            if move_num < 10:
                move = random.choice(legal)
            else:
                captures = []
                opp_color = 1 - state.player_to_move()
                for a in legal:
                    next_state = self.search_problem.transition(state, a)
                    opp_before = sum(sum(state.get_pieces_array(opp_color)))
                    opp_after = sum(sum(next_state.get_pieces_array(opp_color)))
                    if opp_after < opp_before:
                        captures.append(a)
                move = random.choice(captures) if captures else random.choice(legal)
            state = self.search_problem.transition(state, move)
        result = self.search_problem.get_result(state)
        return result if node.state.player_to_move() == MAXIMIZER else -result

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

class BookMCTSAgent(GameAgent):
    """
    Plays center opening move, then defers to standard MCTS.
    """
    def __init__(self, c=np.sqrt(2)):
        self.mcts = MCTSAgent(c=c)
        self.search_problem = GoProblem()
        self.first_move = 12  # (2,2) on 5x5

    def __str__(self):
        return "BookMCTSAgent"

    def get_move(self, state, time_limit):
        # Use Book logic on first move
        blacks = len(state.get_pieces_coordinates(0))
        whites = len(state.get_pieces_coordinates(1))
        if blacks + whites == 0:
            if self.first_move in state.legal_actions():
                return self.first_move

        # Fall back to MCTS
        return self.mcts.get_move(state, time_limit)
    
class BookMCTSAgentWithTT(GameAgent):
    def __init__(self, c=np.sqrt(2)):
        self.first_move = 12
        self.search_problem = GoProblem()
        self.transposition_table = {}  # Shared cache across simulations
        self.c = c

    def __str__(self):
        return "BookMCTSAgent + TranspositionTable"

    def get_move(self, state, time_limit):
        if len(state.get_pieces_coordinates(0)) + len(state.get_pieces_coordinates(1)) == 0:
            if self.first_move in state.legal_actions():
                return self.first_move
        return self._mcts_with_tt(state, time_limit)

    def _mcts_with_tt(self, root_state, time_limit):
        root = MCTSNode(root_state)
        start_time = time.time()

        while time.time() - start_time < time_limit - 0.05:
            leaf = self._select(root)
            child = self._expand(leaf)
            result = self._simulate(child)
            self._backpropagate(child, result)

        legal = root_state.legal_actions()
        best = max(root.children, key=lambda c: c.visits if c.action in legal else -1, default=None)
        return best.action if best else random.choice(legal)

    def _select(self, node):
        while not self.search_problem.is_terminal_state(node.state):
            if len(node.children) < len(node.state.legal_actions()):
                return node
            node = max(node.children, key=lambda c: (c.value / (1 + c.visits)) +
                       self.c * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-6)))
        return node

    def _expand(self, node):
        # Cache previously seen state values
        legal = node.state.legal_actions()
        used = {child.action for child in node.children}
        unvisited = [a for a in legal if a not in used]
        if not unvisited:
            return node
        action = random.choice(unvisited)
        next_state = self.search_problem.transition(node.state, action)

        # Reuse result if already seen
        key = hash(next_state)
        if key in self.transposition_table:
            result = self.transposition_table[key]
        else:
            result = 0

        child = MCTSNode(next_state, parent=node, action=action)
        child.value = result
        node.children.append(child)
        return child

    def _simulate(self, node):
        # Play random rollout and cache result
        sim_state = node.state
        while not self.search_problem.is_terminal_state(sim_state):
            legal = sim_state.legal_actions()
            if not legal:
                break
            move = random.choice(legal)
            sim_state = self.search_problem.transition(sim_state, move)
        result = self.search_problem.get_result(sim_state)
        self.transposition_table[hash(node.state)] = result
        return result

    def _backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

class BookMCTSFlexTimeAgent(GameAgent): ############################## best agent
    """
    Book + MCTS with time scaled to board complexity.
    """
    def __init__(self, c=np.sqrt(2), base_time=0.4, max_time=1.2):
        self.mcts = MCTSAgent(c=c)
        self.search_problem = GoProblem()
        self.first_move = 12  # center move for 5x5
        self.base_time = base_time
        self.max_time = max_time

    def __str__(self):
        return "BookMCTSFlexTime"

    def get_move(self, state, time_limit):
        # Play opening move if it's the first turn
        blacks = len(state.get_pieces_coordinates(0))
        whites = len(state.get_pieces_coordinates(1))
        if blacks + whites == 0 and self.first_move in state.legal_actions():
            return self.first_move

        # Estimate dynamic time based on board complexity
        num_legal = len(state.legal_actions())
        dynamic_time = min(self.base_time + 0.05 * num_legal, self.max_time, time_limit - 0.05)

        return self.mcts.get_move(state, dynamic_time)
    
class BookMCTSFlexTimeTTAgent(GameAgent):
    """
    Book + MCTS + transposition table + dynamic time allocation.
    """
    def __init__(self, c=np.sqrt(2), base_time=0.4, max_time=1.2):
        self.mcts = MCTSAgent(c=c)
        self.search_problem = GoProblem()
        self.first_move = 12  # center move for 5x5
        self.base_time = base_time
        self.max_time = max_time

    def __str__(self):
        return "BookMCTSFlexTimeTT"

    def get_move(self, state, time_limit):
        # Opening book
        blacks = len(state.get_pieces_coordinates(0))
        whites = len(state.get_pieces_coordinates(1))
        if blacks + whites == 0 and self.first_move in state.legal_actions():
            return self.first_move

        # Flexible time allocation
        num_legal = len(state.legal_actions())
        dynamic_time = min(self.base_time + 0.05 * num_legal, self.max_time, time_limit - 0.05)

        # Transposition cache
        original_simulate = self.mcts.simulate
        cache = {}

        def cached_simulate(node):
            key = node.state
            if key in cache:
                return cache[key]
            result = original_simulate(node)
            cache[key] = result
            return result

        self.mcts.simulate = cached_simulate
        move = self.mcts.get_move(state, dynamic_time)
        self.mcts.simulate = original_simulate  # restore original

        return move
    
class BookMCTSFlexTimeHeuristicRolloutAgent(GameAgent):
    """
    Book + MCTS with dynamic time and rollout using piece heuristic.
    """
    def __init__(self, c=np.sqrt(2), base_time=0.4, max_time=1.2):
        self.c = c
        self.base_time = base_time
        self.max_time = max_time
        self.search_problem = GoProblem()
        self.first_move = 12  # center move on 5x5

    def __str__(self):
        return "BookMCTSFlexTimeHeuristicRollout"

    def get_move(self, state, time_limit):
        blacks = len(state.get_pieces_coordinates(0))
        whites = len(state.get_pieces_coordinates(1))
        if blacks + whites == 0 and self.first_move in state.legal_actions():
            return self.first_move

        num_legal = len(state.legal_actions())
        dynamic_time = min(self.base_time + 0.05 * num_legal, self.max_time, time_limit - 0.05)

        root = MCTSNode(state)
        deadline = time.time() + dynamic_time

        while time.time() < deadline:
            leaf = self.select(root)
            child = self.expand(leaf)
            result = self.heuristic_rollout(child)
            self.backpropagate(child, result)

        legal = state.legal_actions()
        best = max((c for c in root.children if c.action in legal), key=lambda c: c.visits, default=None)
        return best.action if best else random.choice(legal)

    def select(self, node):
        while not self.search_problem.is_terminal_state(node.state):
            if len(node.children) < len(node.state.legal_actions()):
                return node
            node = max(
                node.children,
                key=lambda c: c.value / (1 + c.visits) +
                self.c * np.sqrt(np.log(node.visits + 1) / (c.visits + 1e-6))
            )
        return node

    def expand(self, node):
        used = {child.action for child in node.children}
        unvisited = [a for a in node.state.legal_actions() if a not in used]
        if not unvisited:
            return node
        action = random.choice(unvisited)
        next_state = self.search_problem.transition(node.state, action)
        new_child = MCTSNode(next_state, parent=node, action=action)
        node.children.append(new_child)
        return new_child

    def heuristic_rollout(self, node):
        # Prefer actions that gain more pieces and reduce opponent's
        state = node.state
        my_color = state.player_to_move()
        opp_color = 1 - my_color

        while not self.search_problem.is_terminal_state(state):
            legal = state.legal_actions()
            best_action = None
            best_score = -float('inf')

            for action in legal:
                next_state = self.search_problem.transition(state, action)
                my_pieces = np.sum(next_state.get_pieces_array(my_color))
                opp_pieces = np.sum(next_state.get_pieces_array(opp_color))
                diff = my_pieces - opp_pieces

                if diff > best_score:
                    best_score = diff
                    best_action = action

            if best_action:
                state = self.search_problem.transition(state, best_action)
            else:
                state = self.search_problem.transition(state, random.choice(legal))

        result = self.search_problem.get_result(state)
        return result if node.state.player_to_move() == MAXIMIZER else -result

    def backpropagate(self, node, result):
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

def get_final_agent_5x5():
    """Called to construct agent for final submission for 5x5 board"""
    return BookMCTSFlexTimeAgent()

def get_final_agent_9x9():
    """Called to construct agent for final submission for 9x9 board"""
    return MCTSAgent()