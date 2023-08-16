import random


class WordleEnv:
    def __init__(self, word_pool, true_word=None):
        self.word_pool = word_pool
        self.true_word = true_word or random.choice(self.word_pool)
        self.state = [0, 0, 1] * len(self.true_word)  # Start with all Gray.
        self.moves_taken = 0
        self.done = False

    def reset(self, true_word=None):
        self.true_word = true_word or random.choice(self.word_pool)
        self.state = [0, 0, 1] * len(self.true_word)
        self.done = False
        return self.state

    def encode_feedback(self, feedback):
        mapping = {
            "Green": [1, 0, 0],
            "Yellow": [0, 1, 0],
            "Gray": [0, 0, 1]
        }
        encoded = []
        for f in feedback:
            encoded.extend(mapping[f])
        return encoded

    def feedback_for_guess(self, guess):
        feedback = []
        temp_true_word = list(self.true_word)
        temp_guess = list(guess)

        # First, check for 'Green' letters.
        for i in range(len(temp_true_word)):
            if temp_true_word[i] == temp_guess[i]:
                feedback.append("Green")
                temp_true_word[i] = None
                temp_guess[i] = None
            else:
                feedback.append(None)

        # Next, check for 'Yellow' letters.
        for i, letter in enumerate(temp_guess):
            if letter and letter in temp_true_word:
                feedback[i] = "Yellow"
                temp_true_word[temp_true_word.index(letter)] = None

        # Any remaining None values in the feedback are 'Gray'.
        for i in range(len(feedback)):
            if not feedback[i]:
                feedback[i] = "Gray"
                
        return feedback

    def step(self, guess_word):
        feedback = self.feedback_for_guess(guess_word)
        self.state = self.encode_feedback(feedback)

        reward = self.calculate_reward(feedback)

        win_condition = all([color == "Green" for color in feedback])
        
        if win_condition or self.moves_taken >= 6:
            self.done = True

        self.moves_taken += 1
        
        return self.state, reward, self.done

    def calculate_reward(self, feedback):
        reward = 0
        
        # Assign rewards based on feedback
        for f in feedback:
            if f == "Green":
                reward += 1
            elif f == "Yellow":
                reward += 0.5
            # No need to check for "Gray" as its reward is 0

        if reward == 0:
            reward = -1

        # Introduce a penalty if game is not done after 6 moves
        if self.moves_taken >= 5 and not self.done:
            reward -= 5
        else:
            win_condition = all([color == "Green" for color in feedback])
            if win_condition:
                reward += 6
            
        return reward

if __name__ == "__main__":
    test_env = WordleEnv(["hello", "world", "python", "computer", "science"], true_word="hello")
    test_env.step("bratz")
    test_env.step("bratz")
    test_env.step("bratz")
    test_env.step("bratz")
    state, reward, done = test_env.step("bratz")

    print(state, reward, done)