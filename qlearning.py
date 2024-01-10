# You can import matplotlib or numpy, if needed
# You can also import any module included in Python 3.10, for example "random"
# See https://docs.python.org/3.10/py-modindex.html for included modules

import random

class Robot:
    
    def __init__(self):
        """
        Initializes the Robot class with a predefined terrain map, actions, and state indices 
        (maps each (row, column) position on the grid to a unique linear index).
        It creates a reward matrix (R-matrix) based on the terrain and initializes a Q-matrix
        for Q-learning with zero values.
        """        
        self.terrain = [
            # 1    2    3    4    5    6
            ['W', 'M', 'M', 'L', 'L', 'W'],  # A
            ['W', 'W', 'L', 'M', 'L', 'L'],  # B
            ['M', 'L', 'L', 'M', 'L', 'M'],  # C
            ['M', 'L', 'L', 'L', 'L', 'L'],  # D
            ['M', 'L', 'M', 'L', 'M', 'L'],  # E
            ['E', 'W', 'W', 'W', 'W', 'W']   # F
        ]
        self.actions = ['up', 'down', 'left', 'right']
        self.state_indices = {(i, j): i * 6 + j for i in range(6) for j in range(6)}
        
        self.start_state = (0, 3) # Row A, Column 4 (A4)
        self.end_state = (5, 0)   # Row F, Column 1 (F1)

        # rewards dict for each type of terrain
        self.rewards = {'M': -300, 'W': -500, 'L': 10, 'E': 1000}

        # R-matrix 6x6
        self.R_matrix = [[self.rewards.get(self.terrain[i][j], 0) for j in range(6)] for i in range(6)]
  
        # Q-matrix 36x4
        self.Q_matrix = [[0 for _ in range(4)] for _ in range(36)]
        
    def get_next_state_mc(self, current_state):
        """
        Randomly selects a valid action (up, down, left, right) from the current state and returns the resulting state.
        Ensures that the robot does not move off the grid. If no valid actions are available, the robot stays in the current state.
        
        Parameters:
        - current_state (tuple): The current state of the robot as (row, column).
        
        Returns:
        - tuple: The next state of the robot as (row, column).
        """        
        actions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }

        row, col = current_state
        
        possible_actions = []
        for action, (dr, dc) in actions.items():
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 6 and 0 <= new_col < 6:
                possible_actions.append(action)
        
        if possible_actions:
            chosen_action = random.choice(possible_actions)
            dr, dc = actions[chosen_action]
            next_state = (row + dr, col + dc)
        else:
            next_state = current_state
        
        return next_state

    def monte_carlo_exploration(self, num_simulations):
        """
        Performs Monte Carlo simulations to explore the map randomly. 
        Tracks and returns the best route and highest reward encountered.
        
        Parameters:
        - num_simulations (int): The number of Monte Carlo simulations to run.
        
        Returns:
        - tuple: A pair containing the best route (as a list of states) and the highest reward obtained across all simulations.
        """
        best_route = None
        highest_reward = float('-inf')

        for simulation in range(num_simulations):
            current_state = self.start_state
            total_reward = 0
            route = [current_state]

            while current_state != self.end_state:
                next_state = self.get_next_state_mc(current_state)
                total_reward += self.R_matrix[next_state[0]][next_state[1]]
                route.append(next_state)
                current_state = next_state

                if current_state == self.end_state:
                    break

            if total_reward > highest_reward:
                best_route = route
                highest_reward = total_reward

        return best_route, highest_reward

    def get_next_state(self, state_index, action_index):
        """
        Calculates the next state index given the current state and an action.
        Ensures that the state index does not go off the grid. If the action would go off the grid,
        the current state index is returned instead.
        
        Parameters:
        - state_index (int): The linear index of the current state in the Q-matrix.
        - action_index (int): The index of the action taken in the actions list.
        
        Returns:
        - int: The linear index of the next state in the Q-matrix.
        """
        row, col = state_index // 6, state_index % 6
        if action_index == 0: row -= 1  # up
        elif action_index == 1: row += 1  # down
        elif action_index == 2: col -= 1  # left
        elif action_index == 3: col += 1  # right
        return self.state_indices.get((row, col), state_index) 

    def get_next_state_eg(self, state_index, epsilon):
        """
        Chooses the next state index using an epsilon-greedy policy. With probability epsilon,
        a random action is selected; otherwise, the action with the highest Q-value is chosen.
        This method is used to balance exploration and exploitation during the learning process.
        
        Parameters:
        - state_index (int): The index of the current state in the Q-matrix.
        - epsilon (float): The probability of choosing a random action.
        
        Returns:
        - tuple: The index of the next state and the index of the chosen action.
        """
        if random.uniform(0, 1) < epsilon:
            action_index = random.choice(range(4))  # Choose any random action
        else:
            max_q_value = max(self.Q_matrix[state_index])
            best_actions = [action for action, q in enumerate(self.Q_matrix[state_index]) if q == max_q_value]
            action_index = random.choice(best_actions if best_actions else range(4))
        next_state_index = self.get_next_state(state_index, action_index)
        return next_state_index, action_index

    def get_possible_actions(self, state_index):
        """
        Determines the list of possible actions from a given state index based on the grid layout.
        
        Parameters:
        - state_index (int): The index of the current state in the Q-matrix.
        
        Returns:
        - list: A list of indices representing possible actions from the given state.
        """
        row, col = state_index // 6, state_index % 6
        possible_actions = []
        if row > 0: possible_actions.append(0)  # up
        if row < 5: possible_actions.append(1)  # down
        if col > 0: possible_actions.append(2)  # left
        if col < 5: possible_actions.append(3)  # right
        return possible_actions
    
    def q_learning(self, num_episodes, epsilon=0.5, alpha=0.9, gamma=0.9, max_steps=100):
        """
        Runs Q-learning for a specified number of episodes, updating the Q-matrix based on the agent's experiences.
        Each episode starts at a random state on the grid, and the agent uses an epsilon-greedy policy for action selection.
        The episode ends when the agent reaches the goal state or the maximum number of steps is exceeded.
        
        Parameters:
        - num_episodes (int): The number of episodes to run the Q-learning algorithm.
        - epsilon (float): The exploration rate for the epsilon-greedy policy.
        - alpha (float): The learning rate.
        - gamma (float): The discount factor for future rewards.
        - max_steps (int): The maximum number of steps in each episode.
        
        No explicit return value; updates the Q-matrix internally.
        """
        end_state_index = self.state_indices[self.end_state]
        for episode in range(num_episodes):
            # Start at a random state that is not the end state
            current_state_index = random.choice([state for state in range(36) if state != end_state_index])
            for step in range(max_steps):
                if current_state_index == end_state_index:
                    break  # End the episode if the end state is reached
                next_state_index, action_index = self.get_next_state_eg(current_state_index, epsilon)
                row, col = next_state_index // 6, next_state_index % 6
                reward = self.R_matrix[row][col]
                # Q-Learning update
                next_q_values = self.Q_matrix[next_state_index]
                max_next_q_value = max(next_q_values) if next_q_values else 0
                current_q_value = self.Q_matrix[current_state_index][action_index]
                self.Q_matrix[current_state_index][action_index] += alpha * (reward + gamma * max_next_q_value - current_q_value)
                current_state_index = next_state_index

    def greedy_path(self):
        """
        Determines the best path from the start state to the end state based on the highest Q-values in the trained Q-matrix.
        The method follows a greedy policy, selecting the action with the maximum Q-value at each state, until it reaches the goal.
        
        Returns:
        - tuple: The method returns two items;
            1. A list representing the best path as a sequence of states (as (row, column) tuples).
            2. The total reward accumulated along this best path.
        """
        current_state_index = self.state_indices[self.start_state]
        path = [self.start_state]
        total_reward = 0

        while current_state_index != self.state_indices[self.end_state]:
            q_values = self.Q_matrix[current_state_index]
            possible_actions = [(action, self.get_next_state(current_state_index, action)) for action in self.get_possible_actions(current_state_index)]

            if not possible_actions:
                print("No unvisited actions available, breaking out.")
                break

            best_action, best_next_state_index = max(possible_actions, key=lambda action: q_values[action[0]])
            next_state = (best_next_state_index // 6, best_next_state_index % 6)
            path.append(next_state)
            total_reward += self.R_matrix[next_state[0]][next_state[1]]

            current_state_index = best_next_state_index

        return path, total_reward

if __name__ == "__main__":
    # evrth is for testing here
    robot = Robot()
    
    # R-matrix
    print("Initial R-Matrix:")
    for i in robot.R_matrix:
        print(i)
    
    print()
    print("Start State:", robot.start_state)
    print("End State:", robot.end_state)
    print()
    
    # Monte Carlo 100 times
    best_route, highest_reward = robot.monte_carlo_exploration(100)
    print("Best Route from Monte Carlo Exploration:", best_route)
    print("Highest Reward from Monte Carlo Exploration:", highest_reward)
    
    print()
    
    # Q-learning 100 episodes
    robot.q_learning(100)
    print("Q-Matrix after Q-Learning:")
    rounded_q_matrix = [[round(value) for value in row] for row in robot.Q_matrix]
    for row in rounded_q_matrix:
        print(row)
    
    # best route using the greedy policy based on the learned Q-matrix
    print()
    best_route, total_reward = robot.greedy_path()
    print("Best Route from Greedy Path based on Q-Matrix:", best_route)
    print("Total Reward from Greedy Path:", total_reward)
