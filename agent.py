import random
from collections import deque
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from model import QNetwork


@dataclass
class Config:
    hidden: int = 128               # hidden units in the network
    gamma: float = 0.9              # discount: how much future reward counts vs. immediate reward
    lr: float = 1e-3                # Adam learning rate (size of each weight update)
    batch_size: int = 64            # experiences per training update
    memory_size: int = 100_000      # replay buffer capacity; oldest experiences drop out
    train_start: int = 1_000        # don't train until the buffer has this many experiences
    target_update: int = 1_000      # copy online net -> target net every N training updates
    epsilon_start: float = 1.0      # start fully random
    epsilon_min: float = 0.01       # always keep a little exploration during training
    epsilon_decay: float = 0.99     # multiply epsilon by this after every episode


class Agent:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.epsilon = self.cfg.epsilon_start
        # Online network: the one we train and use to pick actions.
        self.net = QNetwork(hidden=self.cfg.hidden)
        # Target network: a frozen copy used to compute training targets. Updating it
        # only occasionally keeps the targets from shifting on every single step.
        self.target = QNetwork(hidden=self.cfg.hidden)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        # Replay buffer: past (state, action, reward, next_state, terminated) tuples.
        # Training on random samples breaks the correlation between consecutive frames.
        self.memory = deque(maxlen=self.cfg.memory_size)
        self.updates = 0

    def act(self, state, explore=True):
        """Epsilon-greedy: with probability epsilon pick a random action, else the best Q-value."""
        if explore and random.random() < self.epsilon:
            return random.randrange(3)
        with torch.no_grad():
            q = self.net(torch.tensor(state, dtype=torch.float32))
        return int(q.argmax())

    def remember(self, state, action, reward, next_state, terminated):
        # Store `terminated`, not `truncated`: a time-limit cut is not a real ending.
        self.memory.append((state, action, reward, next_state, terminated))

    def train_step(self):
        """One minibatch update. Returns the loss, or None if the buffer is still too small."""
        if len(self.memory) < max(self.cfg.train_start, self.cfg.batch_size):
            return None
        batch = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, terminated = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        terminated = torch.tensor(terminated, dtype=torch.float32)

        # Q(s, a) for the action that was actually taken
        q = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a'), but just r if the game really ended.
        # no_grad: targets are fixed labels, we don't backpropagate through them.
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            target = rewards + self.cfg.gamma * next_q * (1 - terminated)

        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.updates += 1
        if self.updates % self.cfg.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def save(self, path):
        torch.save({"model": self.net.state_dict(), "config": asdict(self.cfg), "epsilon": self.epsilon}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path)
        agent = cls(Config(**data["config"]))
        agent.net.load_state_dict(data["model"])
        agent.target.load_state_dict(data["model"])
        agent.epsilon = data["epsilon"]
        return agent
