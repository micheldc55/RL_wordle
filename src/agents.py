import random
from collections import deque
import pickle

import torch
import torch.optim as optim
import torch.nn as nn

from src.networks import QNetwork


class WordleAgent:
    def __init__(self, input_size: int, output_size: int, hidden_dim: int, word_pool: list):
        self.word_pool = word_pool
        self.qnetwork = QNetwork(input_size, output_size, hidden_dim)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # for epsilon-greedy action selection

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.word_pool)
        else:
            with torch.no_grad():
                action_index = self.qnetwork(state).argmax().item()
                return self.word_pool[action_index]

    def store_experience(self, state, action_word, reward, next_state, done):
        action_index = self.word_pool.index(action_word)  # Get index of the chosen word
        self.memory.append((state, action_index, reward, next_state, done))

    
    def learn(self, batch_size=64, gamma=0.99):
        # Check if we have enough experiences in memory
        if len(self.memory) < batch_size:
            return
        
        # Sample a batch of experiences from the memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert the batch data to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get current Q-values
        current_q_values = self.qnetwork(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Get next Q-values
        next_q_values = self.qnetwork(next_states).max(1)[0]

        # Compute the target Q-values
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        # Compute the loss between current and target Q-values
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optionally decay the epsilon
        self.epsilon = max(self.epsilon * 0.998, 0.1)

    def load_agent(self, state_dict_path: str, config_path: str):
        self.qnetwork.load_state_dict(torch.load(state_dict_path))

        with open(config_path, 'rb') as config_file:
            saved_config = pickle.load(config_file)

        self.epsilon = saved_config['epsilon']
        self.word_pool = saved_config['word_pool']
