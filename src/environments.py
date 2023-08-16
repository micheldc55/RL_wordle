class WordleEnv:
    def __init__(self, true_word):
        self.true_word = true_word
        self.state = ... # Initial state before any guess is made.
        self.done = False

    def reset(self, true_word):
        self.__init__(true_word)

    def make_guess(self, guess_word):
        ...
        # Return the new state, reward, and whether the game is over or not.
        return self.state, reward, self.done
