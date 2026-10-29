from torch import nn


class QNetwork(nn.Module):
    """Maps a game state to one Q-value per action.

    Q-values are estimates of total future reward ("if I take this action now and
    play well afterwards, how much reward will I collect?"). They are NOT
    probabilities: they can be negative, don't sum to 1, and we just pick the largest.
    """

    def __init__(self, inputs=14, hidden=128, outputs=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, outputs),  # no activation: Q-values can be any real number
        )

    def forward(self, x):
        return self.layers(x)
