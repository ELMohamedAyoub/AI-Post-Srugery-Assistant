"""
Purpose:
Implements the Q-learning reinforcement learning agent.
"""

import numpy as np

class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        # Q-table for storing Q-values; state is represented as a 3D array (mood, adherence, engagement)
        self.q_table = np.zeros((11, 11, 11, action_size))  # Assuming state range is 0-10 for each component
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def act(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Explore (choose random action)
        return np.argmax(self.q_table[tuple(state)])  # Exploit (choose the best action based on current knowledge)

    def learn(self, state, action, reward, next_state):
        """Update Q-table using the Q-learning formula."""
        q_predict = self.q_table[tuple(state)][action]  # Current Q-value
        q_target = reward + self.gamma * np.max(self.q_table[tuple(next_state)])  # Bellman equation
        # Update the Q-value using the learning rate
        self.q_table[tuple(state)][action] += self.lr * (q_target - q_predict)
        # Decay epsilon for exploration-exploitation tradeoff
        self.epsilon *= self.epsilon_decay
