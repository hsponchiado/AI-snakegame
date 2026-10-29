# AI-snakegame

Snake in pygame, plus a Deep Q-Network (DQN) in PyTorch that learns to play it.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Commands

```bash
python3 snake.py                          # play yourself (WASD / arrow keys, Space to restart)
python3 evaluate.py --games 100           # random-agent check: 100 episodes, no learning
python3 train.py --episodes 500           # train; saves the best checkpoint to model.pth
python3 train.py --episodes 500 --render  # watch training live (slower); add --fps 30 to slow it down
python3 evaluate.py --model model.pth     # model vs. random baseline over 100 games, no exploration
python3 evaluate.py --model model.pth --games 5 --watch   # watch the trained snake play
python3 evaluate.py --model model.pth --games 5 --watch --think 30   # + draw its planned path and Q-values
python3 test_snake.py                     # game-logic and training checks
```

`python3 train.py --help` lists the tunable options (`--lr`, `--gamma`, `--batch-size`,
`--epsilon-decay`, `--max-steps-without-food`, `--cells`, `--seed`, `--render`, `--fps`).
`--cells` sets the board size (default 20); `evaluate.py` takes it too.

## Files

| File | Role |
|------|------|
| `snake.py` | `SnakeGame` environment (`reset`, `step`, `get_state`, `render`) and human play |
| `model.py` | `QNetwork`: 14 inputs, 128 hidden (ReLU), 3 outputs |
| `agent.py` | DQN agent: epsilon-greedy, replay buffer, target network, Adam, `Config` |
| `train.py` | Training loop and logging |
| `evaluate.py` | Random baseline, greedy model evaluation, watching a model |
| `test_snake.py` | Focused checks |

## How it works

**Actions** (relative to where the snake faces): `0` straight, `1` turn right, `2` turn left.

**Observation** (14 values):

- *Danger straight / right / left*: would the next cell in that direction be a wall
  or a body segment? The tail cell counts as safe, because the tail moves away on
  the same step (unless the snake eats).
- *Direction left / right / up / down*: one-hot of the current heading.
- *Food left / right / up / down*: the apple's position compared to the head.

- *Free space straight / right / left* (0.0 to 1.0): if the head moved there, what
  fraction of the empty board could it still reach? Computed with a flood fill
  (breadth-first search) through empty cells. 0 means instant death; a small value
  means the move enters a pocket that may trap the snake later.

The first 11 values only look one cell ahead, so alone they can't see a pocket
forming. The free-space values add that, but they are still a snapshot: they don't
know the tail will keep moving and open space up later, so very long snakes still
trap themselves. Seeing the whole grid would be the next step.

**Rewards**: +10 for an apple, -10 for dying, 0 otherwise. Filling the board is a win.

**Episode endings**:

- *terminated*: the game really ended (death or win). There is no future, so the
  training target is just the reward.
- *truncated*: the snake went `--max-steps-without-food` moves (default 100) without
  eating, so we cut the episode to stop endless circling. The game itself could have
  continued, so the target still includes the estimated future value.

**One step of training**:

1. `state = game.get_state()` — observe 14 numbers.
2. `action = agent.act(state)` — with probability epsilon pick a random action
   (explore), otherwise the action with the highest Q-value (exploit).
3. `next_state, reward, terminated, truncated = game.step(action)`.
4. `agent.remember(...)` — store the experience in the replay buffer.
5. `agent.train_step()` — sample 64 random past experiences and compute
   `target = reward + gamma * max Q_target(next_state)` (just `reward` if terminated),
   then nudge `Q(state, action)` toward the target with MSE loss and Adam.
6. Every 1000 updates, copy the online network into the target network.
7. After each episode, epsilon shrinks (`epsilon *= epsilon_decay`), so the agent
   explores less as it learns.

The network outputs **Q-values**: estimated total future reward for each action.
They are not probabilities; we just pick the largest.

**Hyperparameters** (`agent.Config`):

| Name | Default | Role |
|------|---------|------|
| `gamma` | 0.9 | How much future reward counts. Higher = plans further ahead, learns slower. |
| `lr` | 0.001 | Step size of Adam updates. Too high = unstable, too low = slow. |
| `batch_size` | 64 | Experiences per update. |
| `memory_size` | 100000 | Replay buffer size. |
| `train_start` | 1000 | Experiences collected before training starts. |
| `target_update` | 1000 | Updates between target-network syncs. |
| `epsilon_start / min / decay` | 1.0 / 0.01 / 0.99 | Exploration schedule, per episode. |

Results vary with the random seed and settings; scores are not guaranteed.
