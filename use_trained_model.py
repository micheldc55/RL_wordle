import pickle

from src.environments import WordleEnv
from src.agents import WordleAgent
from src.playing_game import reproduce_game


# Initialize environment and agent
config_path = "data/agent_config.pkl"

with open(config_path, 'rb') as config_file:
    saved_config = pickle.load(config_file)

word_pool = saved_config['word_pool']
input_size = saved_config['input_size']
hidden_dim = saved_config['hidden_dim']
output_size = saved_config['output_size']

agent = WordleAgent(input_size, output_size, hidden_dim, word_pool)
agent.load_agent('wordle_agent_model.pth', 'data/agent_config.pkl')

env = WordleEnv(word_pool)

# Play the game
for i in range(10):
    print(f"Game: {i+1}")
    score = reproduce_game(agent, env)
    print(score)
    print("\n")