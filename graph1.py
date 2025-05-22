import matplotlib.pyplot as plt

# Agent data
agent_names = ['Greedy', 'MCTS', 'AlphaBeta', 'Book', 'BookMCTS', 'BookMCTSFlex']
wins = [2, 4, 2, 2, 6, 6]
losses = [6, 4, 2, 4, 4, 2]
win_ratios = [w / (w + l) for w, l in zip(wins, losses)]

x = range(len(agent_names))

plt.figure(figsize=(10, 8))
plt.bar(x, wins, label='Wins', alpha=0.7)
plt.bar(x, losses, bottom=wins, label='Losses', alpha=0.7)
plt.xticks(x, agent_names, rotation=45)
plt.ylabel("Games Played")
plt.title("Agent Performance: Wins and Losses")
plt.legend()

# Add win ratio annotations
for i, (x_pos, ratio) in enumerate(zip(x, win_ratios)):
    total = wins[i] + losses[i]
    plt.text(x_pos, total + 0.3, f"WR: {ratio:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()