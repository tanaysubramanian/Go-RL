# Reinforcement Learning Agents - Playing the Board Game Go

## Project Description
In this project, I designed and implemented multiple AI agents to play the game of Go. Starting with agents based on simple heuristics like Greedy, Minimax, Alpha-Beta, and Iterative Deepening search algorithms, I eventually developed a competitive hybrid agent combining Monte Carlo Tree Search (MCTS), an opening book strategy, and dynamic time allocation based on board complexity. This agent consistently outperformed others by mimicking human-like play - thinking longer in complex mid-game scenarios and acting quickly in simpler ones. While I explored transposition tables and heuristic rollouts, I found that simpler strategies with consistent rollout quality yielded better results.

## Design Process
#### Phase 1: Foundational Agents
- Implemented GreedyAgent, MinimaxAgent, and AlphaBetaAgent.
- While informative, these agents lacked efficiency and scalability for deeper searches.

#### Phase 2: Anytime Agents
- Built IterativeDeepeningAgent with alpha-beta pruning and time constraints.
- Developed MCTSAgent, which outperformed all previous agents with deeper simulations.

#### Phase 3: Experiments and Innovations
- SolidAgent: Heuristic rollouts that prioritized captures and extensions.
- OptimizedMCTSAgent: Adaptive rollout policy based on game stage.
- BookAgent: A surprisingly strong agent that hardcoded the center move (2,2).
- BookMCTSAgent: Combined the opening book with standard MCTS — improved performance.
- MegaAgent & SuperAgent: Complex strategies using territory heuristics and deeper tree memory — underperformed.
- BookMCTSFlexTimeAgent: My best agent, which combined opening book, MCTS with c = √2, and time per move scaled by board complexity

## Results

My final agent, BookMCTSFlexTimeAgent, achieved the strongest performance among all agents I developed. As shown in the bar chart below, it achieved a win rate of 0.75 against the other agents, including MCTSAgent (0.50), AlphaBetaAgent (0.50), and BookAgent (0.33). The GreedyAgent, which relies on a one-step heuristic, had the lowest win rate at 0.25. These results demonstrate that combining an effective opening strategy with adaptive search timing leads to more consistent victories, particularly in the mid-game where decision complexity peaks.

<img src="https://github.com/user-attachments/assets/e93bba91-29be-4393-b9dd-58108eedbaa7" alt="Image" width="540" height="412.5"> <br />

The heatmap below compares board exploration patterns between MCTSAgent and BookMCTSFlexTimeAgent. The MCTS heatmap shows scattered simulations across the board, indicating uniform exploration without clear prioritization. In contrast, BookMCTSFlexTimeAgent concentrated its visits on high-value positions, especially in the center and certain edges, reflecting the influence of the opening book and smarter mid-game targeting. This focused behavior resulted in stronger board control and improved win rates.

<img src="https://github.com/user-attachments/assets/36695c7d-b6ba-4ace-b8cc-3e9be685c13c" alt="Image" width="705" height="290"> <br />

The plot below highlights how agents allocated their computation time. GreedyAgent had the shortest average move duration and the most time left on the clock, underscoring its simplistic and shallow evaluation. In contrast, BookMCTSFlexTimeAgent used significantly more time per move on average but still managed to conserve enough total time, demonstrating effective time budgeting. Compared to the baseline MCTS and Book agents, my optimized agent showed more strategic use of time, spending more when it mattered without risking timeout.

<img src="https://github.com/user-attachments/assets/681ff8bc-17fa-4d8c-aa2a-43a0fa4bf0d5" alt="Image" width="652" height="309"> <br />

Together, these results illustrate that deliberate integration of an opening strategy, adaptive timing, and structured exploration can significantly boost the performance of MCTS-based Go agents.
