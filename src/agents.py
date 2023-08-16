import torch.optim as optim
from collections import deque
import random

class WordleAgent:
    def __init__(self, input_size, output_size, hidden_dim):
        self.qnetwork = QNetwork(input_size, output_size, hidden_dim)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # for epsilon-greedy action selection

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(possible_words)
        else:
            with torch.no_grad():
                return self.qnetwork(state).argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self):
        ...
        # Sample from memory, calculate Q-values using the Bellman equation, and update the network.
