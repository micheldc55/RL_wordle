import torch
import pickle

from src.vocabulary import parse_vocabulary
from src.environments import WordleEnv
from src.agents import WordleAgent


# Parameters
input_size = 15  # Change this if necessary based on the state representation
hidden_dim = 128
word_pool = parse_vocabulary("data/reduced_vocab.txt")
output_size = len(word_pool)  # Assuming word_pool contains all valid 5-letter words
num_episodes = 100_000
batch_size = 64

# Initialize environment and agent
env = WordleEnv(word_pool)
agent = WordleAgent(input_size, output_size, hidden_dim, word_pool)

# Main training loop
for episode in range(num_episodes):
    # Reset the environment for a new game
    state = env.reset()
    
    # Convert the state to a tensor
    state = torch.tensor(state, dtype=torch.float32)

    total_reward = 0
    done = False

    while not done:
        # Agent picks an action
        action_word = agent.choose_action(state)

        # Execute the action in the environment
        next_state, reward, done = env.step(action_word)

        # Convert the next_state to a tensor
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Store the experience in memory
        agent.store_experience(state, action_word, reward, next_state, done)

        # Move to the next state
        state = next_state

        # Learn from a batch of experiences
        agent.learn(batch_size)

        total_reward += reward

    # Print episode stats
    if episode % 1000 == 0:
        print(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.2f}")

print("Training complete!")

torch.save(agent.qnetwork.state_dict(), 'wordle_agent_model.pth')

config = {
    'epsilon': agent.epsilon,
    'word_pool': word_pool
}

with open('data/agent_config.pkl', 'wb') as config_file:
    pickle.dump(config, config_file)