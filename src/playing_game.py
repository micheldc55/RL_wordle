import torch


def reproduce_game(agent, env, verbose=True):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    total_reward = 0

    while not done:
        with torch.inference_mode():
            agent.qnetwork.eval()
            action_word = agent.choose_action(state)
            print(f"Env. before step: {env.done}")
            next_state, reward, done = env.step(action_word)
            print(f"Env. after step: {env.done}")
            feedback = env.feedback_for_guess(action_word)

        if verbose:
            print(f"True word: {env.true_word}")
            print("Is done: ", done)
            print(f"State: {next_state}")
            print(f"Action: {action_word}")
            print(f"Reward: {reward}")
            print(f"Feedback: {feedback}")
            print("----------------------------")

        state = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

    return total_reward
