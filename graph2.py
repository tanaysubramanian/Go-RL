import matplotlib.pyplot as plt
import numpy as np

# Agent names
agents = ['Greedy', 'MCTS', 'AlphaBeta', 'Book', 'BookMCTS', 'BookMCTSFlex']

avg_durations = [0.00848010392772117, 1.088936979820331, 0.06258121177554131, 1.6607287289115829, 1.0555501516212473, 1.0586630327045927]
max_durations = [0.022003889083862305, 1.3862149715423584, 0.14934301376342773, 15.979519844055176, 1.3008415699005127, 1.3887012004852295]
avg_time_remaining = [86.39055454730988, 13.26521921157837, 35.873987317085266, 1.0388250350952148, 13.951247990131378, 9.907788276672363]

x = np.arange(len(agents))
width = 0.25

# Create subplots
plt.figure(figsize=(12, 6))
plt.bar(x - width, avg_durations, width=width, label='Avg Move Duration (s)')
plt.bar(x, max_durations, width=width, label='Max Move Duration (s)')
plt.bar(x + width, avg_time_remaining, width=width, label='Avg Time Remaining (s)')

# Axis labels and legend
plt.xticks(x, agents, rotation=30)
plt.ylabel("Seconds")
plt.title("Agent Timing Metrics")
plt.legend()
plt.tight_layout()
plt.show()