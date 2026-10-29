import argparse
import random
from collections import deque

import torch

from agent import Agent, Config
from snake import SnakeGame


def train(episodes, save_path, max_steps_without_food=100, render=False, seed=None, config=None, log=True, fps=0, cells=20):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    game = SnakeGame(render=render, max_steps_without_food=max_steps_without_food, cells=cells)
    if render:
        import pygame
        clock = pygame.time.Clock()
    closed = False
    agent = Agent(config)
    recent = deque(maxlen=100)  # scores of the last 100 episodes
    best_avg = float("-inf")

    for episode in range(1, episodes + 1):
        state = game.reset()
        losses = []
        while True:
            # 1. observe -> 2. act -> 3. reward -> 4. store -> 5. learn
            action = agent.act(state)
            next_state, reward, terminated, truncated = game.step(action)
            agent.remember(state, action, reward, next_state, terminated)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            if render:
                pygame.display.set_caption(f"Training  episode {episode}  epsilon {agent.epsilon:.3f}")
                game.render()
                # Handle window events so it stays responsive; closing it stops training.
                closed = any(e.type == pygame.QUIT for e in pygame.event.get())
                if fps:
                    clock.tick(fps)
            if terminated or truncated or closed:
                break
        if closed:
            print("window closed, stopping training")
            break

        agent.decay_epsilon()
        recent.append(game.score)
        avg = sum(recent) / len(recent)
        mean_loss = sum(losses) / len(losses) if losses else float("nan")
        if log:
            end = "won" if game.won else "truncated" if truncated else "died"
            print(f"ep {episode:5d}  score {game.score:3d}  avg100 {avg:6.2f}  "
                  f"eps {agent.epsilon:.3f}  loss {mean_loss:8.4f}  ({end})")
        # Keep the checkpoint with the best rolling average (only once 100 episodes exist).
        if len(recent) == recent.maxlen and avg > best_avg:
            best_avg = avg
            agent.save(save_path)
    if best_avg == float("-inf"):  # short run: save whatever we have
        agent.save(save_path)
    return agent


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train a DQN to play Snake.")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--save", default="model.pth")
    p.add_argument("--max-steps-without-food", type=int, default=100)
    p.add_argument("--render", action="store_true", help="show the window while training (slow)")
    p.add_argument("--fps", type=int, default=0, help="speed limit with --render (0 = as fast as possible)")
    p.add_argument("--cells", type=int, default=20, help="board size (cells per side)")
    p.add_argument("--seed", type=int)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--gamma", type=float, default=Config.gamma)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--epsilon-decay", type=float, default=Config.epsilon_decay)
    a = p.parse_args()
    cfg = Config(lr=a.lr, gamma=a.gamma, batch_size=a.batch_size, epsilon_decay=a.epsilon_decay)
    train(a.episodes, a.save, a.max_steps_without_food, a.render, a.seed, cfg, fps=a.fps, cells=a.cells)
