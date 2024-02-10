# Q-Learning-Robot-exploring-unknown-terrain

![image](https://github.com/AlmazErmilov/Q-Learning-Robot-exploring-unknown-terrain/assets/64559090/cb67e5df-d955-4469-9276-9878b64ae02c)

## Summary

The most important thing in this task was to play around with and understand the parameters described below, and also to find the balance between all these parameters for this task ("Fine-Tuning"):

1.	Learning Rate (alpha):
•	Determines how much new information overrides old information. A high learning rate means the robot learns quickly (but can potentially ignore useful previous information). 
•	I chose at the end alpha=0.9: during the Q-learning updates, the algorithm will be strongly affected by new information, but it will not completely overwrite the old value, weighted average, with 90% of the new value and 10% of the old value (too high value for a stable environment maybe, but that's why I don't need so many epochs for training).

2.	Discount Factor (gamma):
•	determines the importance of future rewards. A factor close to 1 values future rewards almost as highly as immediate rewards, favouring long-term optimal policies. A lower discount factor makes the agent short-sighted and focuses more on immediate rewards.
•	I chose at the end gamma=0.9: quite high, meaning that future rewards are considered almost as important as immediate rewards. This encourages the algorithm to find routes that provide high rewards in the long term, rather than just looking for immediate gains (too high a value for a stable environment, but that's why I don't need so many epochs for training).

3.	Exploration vs. Exploitation (epsilon):
•	Fine-tuning epsilon is about balancing exploration and exploitation. Higher epsilon encourages exploration to learn more about the environment.
•	I chose epsilon=0.5: With a value of 0.5, the robot chooses to explore (select a random action) half the time and exploit (select the best known action) the other half. This balance can be particularly effective because it allows the algorithm to discover new paths ("exploration") while utilising the knowledge it has already acquired ("exploitation").

4.	Number of Episodes (num_episodes):
•	The number of episodes affects how much experience the agent gains. Too few episodes may not be enough for the agent to learn effectively, while too many episodes may be wasted or lead to over-adaptation to the environment.
•	I chose at the end num_episodes=100 (or much less due to large alpha and gamma): Although more episodes generally provide better learning, even a lower number like 100 can be sufficient if the learning rate (alpha) is high, because the Q values are updated more aggressively.

5.	Reward:
I chose at the end: self.rewards = {'M': -300, 'W': -500, 'L': 10, 'E': 1000}
•	'M': -300 and 'W': -500: The penalty for mountain (M) and water (W) makes these terrains much more undesirable. This causes the robot to avoid these conditions to a greater extent, making it less likely that the route will pass through them. 
•	'L': 10: Keeping the regular land reward (L) at 10 provides a small positive reward that encourages the robot to stay on land whenever possible.
•	'E': 1000, End (E): Such a large positive reward gives the algorithm a strong incentive to reach the goal, making the end state very attractive. This can speed up the learning process of the path leading to the goal.

The combination of these parameters and reward settings created a learning environment where the robot could quickly learn a fine balance between exploring new paths and utilising familiar paths, while largely avoiding negative terrain and aiming for the target. The high learning rate allowed the robot to quickly adapt to the increased rewards and penalties, resulting in an efficient route to the target.
