# Abstract Base Class for Neural Architectures
import torch
import torch.nn as nn


class BaseSepsisModel(nn.Module):
    """
    Abstract Base Sepsis model. Exposes helper functions for counting
    trainable parameters to document complexity in research evaluations.
    """
    def __init__(self):
        super(BaseSepsisModel, self).__init__()

    def count_trainable_parameters(self):
        """
        Returns the total number of trainable model weights.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self, logger=None):
        """
        Logs model summary parameters.
        """
        n_params = self.count_trainable_parameters()
        msg = f"Model Architecture: {self.__class__.__name__} | Trainable Parameters: {n_params:,}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
