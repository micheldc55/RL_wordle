import torch
import torch.nn as nn

def compute_q_average(q_values):
    """Compute the average Q-value over the batch.

    :param q_values: Q-values
    :return: average Q-value
    """
    return torch.mean(q_values, dim=1)