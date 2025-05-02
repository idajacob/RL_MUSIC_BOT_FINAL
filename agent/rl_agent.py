import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount=0.95, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Antag hver note er mellem 0-12 => tilstanden repræsenteres som tuple af noter (0 hvis tom)
        self.q_table = {}  # Lazy init: gem kun sete tilstande

    def get_action(self, state):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        old_value = self.q_table[state][action]
        future_estimate = 0 if done else np.max(self.q_table[next_state])
        new_value = old_value + self.lr * (reward + self.gamma * future_estimate - old_value)
        self.q_table[state][action] = new_value

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
