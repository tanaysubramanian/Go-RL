import matplotlib.pyplot as plt
import numpy as np

def plot_visit_heatmap(root, board_size):
    """
    Plot a heatmap showing how many times each action was visited.
    """
    heatmap = np.zeros((board_size, board_size))

    for child in root.children:
        if child.action != board_size * board_size:
            x = child.action % board_size
            y = child.action // board_size
            heatmap[y, x] = child.visits

    plt.figure(figsize=(6, 5))
    plt.title("BookMCTSFlexTimeAgent Visit Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Number of Visits")
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    import time
    from agents import MCTSAgent, MCTSNode, BookMCTSFlexTimeAgent
    from go_search_problem import GoProblem

    problem = GoProblem(size=5)
    start_state = problem.start_state

    agent = BookMCTSFlexTimeAgent()
    mcts_agent = agent.mcts
    root = MCTSNode(start_state)

    # Estimate dynamic time just like the agent would
    num_legal = len(start_state.legal_actions())
    dynamic_time = min(agent.base_time + 0.05 * num_legal, agent.max_time, 1.0)

    deadline = time.time() + dynamic_time
    while time.time() < deadline:
        leaf = mcts_agent.select(root)
        child = mcts_agent.expand(leaf)
        result = mcts_agent.simulate(child)
        mcts_agent.backpropagate(child, result)

    plot_visit_heatmap(root, board_size=5)