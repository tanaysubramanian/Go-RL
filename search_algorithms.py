import random
from typing import Dict, Tuple
from adversarial_search_problem import Action, State as GameState
from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem

def minimax(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
    state = asp.get_start_state()
    stats = {"expanded_nodes": [0]}

    if state.player_to_move() == 0:
        _, action = maximize(asp, state, 0, cutoff_depth, stats["expanded_nodes"])

    else:
        _, action = minimize(asp, state, 0, cutoff_depth, stats["expanded_nodes"])
        
    return action, {"states_expanded": stats["expanded_nodes"][0]}

def maximize(asp, state, depth, cutoff, expanded):
    expanded[0] += 1

    if depth == cutoff:
        return asp.heuristic(state, state.player_to_move()), None
    
    if asp.is_terminal_state(state):
        return asp.heuristic(state, state.player_to_move()), None
    
    best_val, best_act = float('-inf'), None

    for action in asp.get_available_actions(state):
        child = asp.transition(state, action)
        val, _ = minimize(asp, child, depth + 1, cutoff, expanded)

        if val > best_val:
            best_val, best_act = val, action

    return best_val, best_act

def minimize(asp, state, depth, cutoff, expanded):
    expanded[0] += 1

    if depth == cutoff:
        return asp.heuristic(state, state.player_to_move()), None
    
    if asp.is_terminal_state(state):
        return asp.heuristic(state, state.player_to_move()), None
    
    best_val, best_act = float('inf'), None

    for action in asp.get_available_actions(state):
        child = asp.transition(state, action)
        val, _ = maximize(asp, child, depth + 1, cutoff, expanded)

        if val < best_val:
            best_val, best_act = val, action

    return best_val, best_act

def alpha_beta(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
    state = asp.get_start_state()
    stats = {"expanded_nodes": [0]}

    if state.player_to_move() == 0:
        _, action = alpha_beta_max(asp, state, float('-inf'), float('inf'), 0, cutoff_depth, stats["expanded_nodes"])

    else:
        _, action = alpha_beta_min(asp, state, float('-inf'), float('inf'), 0, cutoff_depth, stats["expanded_nodes"])

    return action, {"states_expanded": stats["expanded_nodes"][0]}

def alpha_beta_max(asp, state, alpha, beta, depth, cutoff, expanded):
    expanded[0] += 1

    if asp.is_terminal_state(state):
        return asp.heuristic(state, state.player_to_move()), None
    
    if depth == cutoff:
        return asp.heuristic(state, state.player_to_move()), None
    
    best_val, best_act = float('-inf'), None
    actions = asp.get_available_actions(state)
    random.shuffle(actions)

    for action in actions:
        child = asp.transition(state, action)
        val, _ = alpha_beta_min(asp, child, alpha, beta, depth + 1, cutoff, expanded)

        if val > best_val:
            best_val, best_act = val, action

        alpha = max(alpha, best_val)

        if best_val >= beta:
            break

    return best_val, best_act

def alpha_beta_min(asp, state, alpha, beta, depth, cutoff, expanded):
    expanded[0] += 1

    if asp.is_terminal_state(state):
        return asp.heuristic(state, state.player_to_move()), None
    
    if depth == cutoff:
        return asp.heuristic(state, state.player_to_move()), None
    
    best_val, best_act = float('inf'), None
    actions = asp.get_available_actions(state)
    random.shuffle(actions)

    for action in actions:
        child = asp.transition(state, action)
        val, _ = alpha_beta_max(asp, child, alpha, beta, depth + 1, cutoff, expanded)

        if val < best_val:
            best_val, best_act = val, action

        beta = min(beta, best_val)
        
        if best_val <= alpha:
            break

    return best_val, best_act