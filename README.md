# RL for wordle

Repository for implementing Deep Reinforcement Learning for beating Wordle

Very basic implementation of a Reinforcement Learning algorithm for beating a very simplified version of Wordle with only 100 words. The algorithm is a simple Q-learning algorithm with a neural network as the Q-function approximator. The neural network is a simple feed-forward network with 2 hidden layers. The input is a one-hot encoding of the current state (the letters that have been guessed correctly and the letters that have been guessed incorrectly) and the output is a vector of size 100 with the Q-values for each word in the vocabulary. The algorithm is trained on 100.000 games.

This is a **very** rough Reinforcement Learning implementation. The code is not optimized and the algorithm is not very sophisticated. The goal of this project was to get a better understanding of how Reinforcement Learning works and how to implement it. The algorithm is not very good at beating the game, but it does learn to beat the game to some extent. Mainly by learning some type of posterior distribution of possible outcomes given the initial word it always chooses.