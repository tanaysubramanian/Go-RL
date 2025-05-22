import time
from go_search_problem import GoProblem
import tqdm
import numpy as np
from go_gui import GoGUI
from agents import *
import pygame
import argparse

pygame.init()
clock = pygame.time.Clock()

BLACK = MAXIMIZER = 0
WHITE = MINIMIZER = 1


def run_game(agent1, agent2, time_limit=15, time_increment=1, hard_time_cutoff=True, size=5):
    """
    Run a single game between two agents.
    :param agent1: The first agent
    :param agent2: The second agent
    :param time_limit: The time limit for each player (starting time)
    :param time_increment: The time increment for each player (additional time per move)
    :param hard_time_cutoff: If true, will terminate the game when a player runs out of time
                                If false, will continue to play until the game is over.
    :return: The result of the game (1 for agent1 win, -1 for agent2 win)
    """
    my_go = GoProblem(size=size)
    state = my_go.start_state
    player1_time = time_limit
    player2_time = time_limit
    player1_durations = []
    player2_durations = []
    while (not my_go.is_terminal_state(state)):
        start_time = time.time()
        # Clone so as to avoid side effects from agents
        player1_action = agent1.get_move(state.clone(), player1_time)
        move_duration = time.time() - start_time
        player1_time -= move_duration
        player1_durations.append(move_duration)
        if (player1_time <= 0):
            if hard_time_cutoff:
                info = {
    "Agent 1 End Time": player1_time,
    "Agent 2 End Time": player2_time,
    "Agent 1 Average Duration": np.mean(player1_durations) if player1_durations else 0,
    "Agent 2 Average Duration": np.mean(player2_durations) if player2_durations else 0,
    "Agent 1 Longest Duration": np.max(player1_durations) if player1_durations else 0,
    "Agent 2 Longest Duration": np.max(player2_durations) if player2_durations else 0,
    "Agent 1 Score": -1,
    "Agent 2 Score": 1
}
                return -1, info
        player1_time += time_increment
        state = my_go.transition(state, player1_action)
        if (my_go.is_terminal_state(state)):
            break
        start_time = time.time()
        player2_action = agent2.get_move(state.clone(), player2_time)
        duration = time.time() - start_time
        player2_durations.append(duration)
        player2_time -= duration
        if (player2_time <= 0):
            print("Player 2 over time")
            if hard_time_cutoff:
                info = {"Agent 1 End Time": player1_time, "Agent 2 End Time": player2_time,
                        "Agent 1 Average Duration": np.mean(player1_durations),
                        "Agent 2 Average Duration": np.mean(player2_durations),
                        "Agent 1 Longest Duration": np.max(player1_durations),
                        "Agent 2 Longest Duration": np.max(player2_durations),
                        "Agent 1 Score": -1, "Agent 2 Score": 1}
                return 1, info
        else:
            player2_time += time_increment
        state = my_go.transition(state, player2_action)
    info = {"Agent 1 End Time": player1_time, "Agent 2 End Time": player2_time,
            "Agent 1 Average Duration": np.mean(player1_durations),
            "Agent 2 Average Duration": np.mean(player2_durations),
            "Agent 1 Longest Duration": np.max(player1_durations),
            "Agent 2 Longest Duration": np.max(player2_durations),
            "Agent 1 Score": -1, "Agent 2 Score": 1}
    return my_go.get_result(state), info


def run_many(agent1, agent2, num_games=10, verbose=True, size=5):
    agent1_score = 0
    agent2_score = 0
    agent1_score_black = 0
    agent2_score_black = 0
    agent1_average_duration = 0
    agent2_average_duration = 0

    agent1_longest_duration = 0
    agent2_longest_duration = 0

    agent1_average_time_remaining = 0
    agent2_average_time_remaining = 0

    agent1_min_time_remaining = float('inf')
    agent2_min_time_remaining = float('inf')

    for _ in tqdm.tqdm(range(int(num_games / 2))):
        result, info = run_game(agent1, agent2, size=size)
        agent1_score += result
        agent2_score += -result
        agent1_score_black += result

        agent1_average_duration += info["Agent 1 Average Duration"] / num_games
        agent2_average_duration += info["Agent 2 Average Duration"] / num_games

        agent1_longest_duration = max(
            agent1_longest_duration, info["Agent 1 Longest Duration"])
        agent2_longest_duration = max(
            agent2_longest_duration, info["Agent 2 Longest Duration"])

        agent1_average_time_remaining += info["Agent 1 End Time"] / num_games
        agent2_average_time_remaining += info["Agent 2 End Time"] / num_games

        agent1_min_time_remaining = min(
            agent1_min_time_remaining, info["Agent 1 End Time"])
        agent2_min_time_remaining = min(
            agent2_min_time_remaining, info["Agent 2 End Time"])

        result, info = run_game(agent2, agent1, size=size)

        # Note that since player 2 goes first in the second game,
        # The stats will look backwards
        agent2_score_black += result
        agent1_score += -result
        agent2_score += result

        agent1_average_duration += info["Agent 2 Average Duration"] / num_games
        agent2_average_duration += info["Agent 1 Average Duration"] / num_games

        agent1_longest_duration = max(
            agent1_longest_duration, info["Agent 2 Longest Duration"])
        agent2_longest_duration = max(
            agent2_longest_duration, info["Agent 1 Longest Duration"])

        agent1_average_time_remaining += info["Agent 2 End Time"] / num_games
        agent2_average_time_remaining += info["Agent 1 End Time"] / num_games

        agent1_min_time_remaining = min(
            agent1_min_time_remaining, info["Agent 2 End Time"])
        agent2_min_time_remaining = min(
            agent2_min_time_remaining, info["Agent 1 End Time"])

    if verbose:
        print("Agent 1: " + str(agent1) + " Score: " + str(agent1_score))
        print("Agent 2: " + str(agent2) + " Score: " + str(agent2_score))
        print("Agent 1: " + str(agent1) + " Score with Black (first move): " +
              str(agent1_score_black))
        print("Agent 2: " + str(agent2) + " Score with Black (first move): " +
              str(agent2_score_black))
        print("Agent 1: " + str(agent1) + " Average Duration: " +
              str(agent1_average_duration))
        print("Agent 2: " + str(agent2) + " Average Duration: " +
              str(agent2_average_duration))
        print("Agent 1: " + str(agent1) + " Longest Duration: " +
              str(agent1_longest_duration))
        print("Agent 2: " + str(agent2) + " Longest Duration: " +
              str(agent2_longest_duration))
        print("Agent 1: " + str(agent1) + " Average Time Remaining: " +
              str(agent1_average_time_remaining))
        print("Agent 2: " + str(agent2) + " Average Time Remaining: " +
              str(agent2_average_time_remaining))
        print("Agent 1: " + str(agent1) + " Min Time Remaining: " +
              str(agent1_min_time_remaining))
        print("Agent 2: " + str(agent2) + " Min Time Remaining: " +
              str(agent2_min_time_remaining))

    return agent1_score, agent2_score

def print_state(state):
    """
    Print the state of the game in a human-readable format.
    """
    print("  ", end="")
    for i in range(len(state.get_board()[0])):
        print(i, end=" ")
    print()
    for i in range(len(state.get_board()[0])):
        print(i, end=" ")
        for j in range(len(state.get_board()[0])):
            if state.get_board()[0][i][j] == 1:
                print("B", end=" ")
            elif state.get_board()[1][i][j] == 1:
                print("W", end=" ")
            else:
                print(".", end=" ")
        print()
    print()


def run_game_with_text(agent, size=5):
    """
    Run a single game between a human and an agent with text output.
    :param agent: The agent to play against (must be a subclass of GameAgent)
    """
    print("Starting game...", size)
    my_go = GoProblem(size=size)
    state = my_go.start_state
    while (not my_go.is_terminal_state(state)):
        player1_action = agent.get_move(state.clone(), 1)
        print("Agent Action:", player1_action, ", which corresponds to coordinate ", my_go.action_index_to_string(player1_action))
        state = my_go.transition(state, player1_action)
        if (my_go.is_terminal_state(state)):
            break
        action = None
        print_state(state)
        print("Human's turn")
        while action is None:
            while action not in state.legal_actions():
                col = int(input("Enter col of action: "))
                row = int(input("Enter row of action: "))
                action = row * len(state.get_pieces_array(0)) + col
            print("Human Action:", action, ", which corresponds to coordinate ", my_go.action_index_to_string(action))
        state = my_go.transition(state, action)
        print_state(state)
    print("Done!")
    print("Winner: " + str(my_go.get_result(state)))

def run_game_with_gui(agent, size=5):
    """
    Run a single game between a human and an agent with a GUI.
    :param agent: The agent to play against (must be a subclass of GameAgent)
    """
    print("Starting game...", size)
    my_go = GoProblem(size=size)
    state = my_go.start_state
    gui = GoGUI(my_go)
    while (not my_go.is_terminal_state(state)):
        player1_action = agent.get_move(state.clone(), 1)
        state = my_go.transition(state, player1_action)
        gui.update_state(state)
        gui.render()
        if (my_go.is_terminal_state(state)):
            break
        action = None
        while action is None:
            while action not in state.legal_actions():
                action = gui.get_user_input_action()
                gui.render()
                clock.tick(60)
            print("Human Action:", action, ", which corresponds to coordinate ", my_go.action_index_to_string(action))
            gui.render()
            clock.tick(60)
        state = my_go.transition(state, action)
        gui.update_state(state)
        gui.render()
        clock.tick(60)
    print("Done!")
    if my_go.get_result(state) == 1:
        print("Agent wins!")
    else:
        print("You won!")

def create_agent(agent_type: str, **kwargs):
    """
    Factory function to create agents based on command line arguments

    :param agent_type: The type of agent to create (string)
    :param kwargs: Additional arguments for the agent (e.g., depth, parameters, etc.)
    """
    if agent_type.lower() == "alphabeta":
        depth = kwargs.get('depth', 2)
        return AlphaBetaAgent(depth_cutoff=depth)
    elif agent_type.lower() == "minimax":
        depth = kwargs.get('depth', 2)
        return MinimaxAgent(depth_cutoff=depth)
    elif agent_type.lower() == "random":
        return RandomAgent()
    elif agent_type.lower() == "greedy":
        return GreedyAgent()
    elif agent_type.lower() == "ids":
        return IterativeDeepeningAgent()
    elif agent_type.lower() == "mcts":
        return MCTSAgent()
    elif agent_type.lower() == "solid":
        return SolidAgent()
    elif agent_type.lower() == "book":
        return Book(SolidAgent())
    elif agent_type.lower() == "mega":
        return MegaAgent()
    elif agent_type.lower() == "mcts1":
        return OptimizedMCTSAgent()
    elif agent_type.lower() == "bookmcts":
        return BookMCTSAgent()
    elif agent_type.lower() == "bookmctstt":
        return BookMCTSAgentWithTT()
    elif agent_type.lower() == "bookmctsflex":
        return BookMCTSFlexTimeAgent()
    elif agent_type.lower() == "bookmctsflextt":
        return BookMCTSFlexTimeTTAgent()
    elif agent_type.lower() == "bookmctsflexheuristic":
        return BookMCTSFlexTimeHeuristicRolloutAgent()
    # Add more agent types here as needed
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def parse_args():
    parser = argparse.ArgumentParser(description='Go Game Runner')
    
    # Mode selection
    parser.add_argument('--mode', choices=['gui', 'vs', 'tournament', 'text'], default='gui',
                      help='Run mode: gui (play against AI), vs (single game between agents), tournament (multiple games)')

    # Agent configuration
    parser.add_argument('--agent1-type', default='alphabeta',
                      help='Type of agent 1 (e.g., alphabeta)')
    parser.add_argument('--agent1-depth', type=int, default=2,
                      help='Depth limit for agent 1 if applicable')
    
    parser.add_argument('--agent2-type', default='alphabeta',
                      help='Type of agent 2 (e.g., alphabeta)')
    parser.add_argument('--agent2-depth', type=int, default=2,
                      help='Depth limit for agent 2 if applicable')
    
    # Game settings
    parser.add_argument('--time-limit', type=float, default=15,
                      help='Time limit per player in seconds')
    parser.add_argument('--time-increment', type=float, default=1,
                      help='Time increment per move in seconds')
    parser.add_argument('--soft-time', action='store_true',
                      help='Continue game even if time limit is exceeded')
    parser.add_argument('--size', type=int, default=5,
                      help='Size of the Go board')
    
    # Tournament settings
    parser.add_argument('--num-games', type=int, default=10,
                      help='Number of games to play in tournament mode')
    parser.add_argument('--quiet', action='store_true',
                      help='Suppress detailed output in tournament mode')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Create agents based on arguments
    agent1 = create_agent(args.agent1_type, depth=args.agent1_depth)
    if args.mode == 'gui':
        run_game_with_gui(agent1, size=args.size)
    elif args.mode == 'text':
        run_game_with_text(agent1, size=args.size)
    else:
        agent2 = create_agent(args.agent2_type, depth=args.agent2_depth)
        if args.mode == 'vs':
            result, info = run_game(agent1, agent2, 
                                  time_limit=args.time_limit,
                                  time_increment=args.time_increment,
                                  hard_time_cutoff=not args.soft_time,
                                  size=args.size)
            print("Game Info:", info)
        elif args.mode == 'tournament':
            run_many(agent1, agent2, 
                    num_games=args.num_games,
                    verbose=not args.quiet,
                    size=args.size)


if __name__ == "__main__":
    main()