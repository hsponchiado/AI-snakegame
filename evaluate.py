import argparse
import random

from agent import Agent
from snake import SnakeGame


def run(policy, games, max_steps_without_food=100, render=False, fps=15):
    """Play `games` episodes with `policy(state) -> action`. No learning happens here."""
    game = SnakeGame(render=render, max_steps_without_food=max_steps_without_food)
    if render:
        import pygame
        clock = pygame.time.Clock()
    scores, ends = [], {"died": 0, "truncated": 0, "won": 0}
    for _ in range(games):
        state = game.reset()
        while True:
            state, _, terminated, truncated = game.step(policy(state))
            if render:
                game.render()
                pygame.event.pump()  # keep the window responsive
                clock.tick(fps)
            if terminated or truncated:
                break
        scores.append(game.score)
        ends["won" if game.won else "truncated" if truncated else "died"] += 1
    return scores, ends


def report(name, scores, ends):
    print(f"{name:7s} games {len(scores)}  avg apples {sum(scores) / len(scores):.2f}  "
          f"max {max(scores)}  endings {ends}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a random agent and/or a saved model.")
    p.add_argument("--model", help="checkpoint to evaluate, e.g. model.pth")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--watch", action="store_true", help="open a window and watch the model play")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--max-steps-without-food", type=int, default=100)
    a = p.parse_args()

    report("random", *run(lambda s: random.randrange(3), a.games, a.max_steps_without_food))
    if a.model:
        agent = Agent.load(a.model)
        agent.net.eval()
        # explore=False: always take the highest Q-value. We never call remember/train_step,
        # so the agent and its (empty) replay memory are untouched.
        policy = lambda s: agent.act(s, explore=False)
        report("model", *run(policy, a.games, a.max_steps_without_food, a.watch, a.fps))
