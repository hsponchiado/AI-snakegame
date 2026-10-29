import argparse
import copy
import random

import torch

from agent import Agent
from snake import SnakeGame


def think(agent, game, steps):
    """What the agent expects to do next: Q-values now, plus the path it would follow.

    Plays ahead on a copy of the game. Stops at the next apple (where the new apple
    spawns is unknown) or at death. The random state is restored so the real game
    gets the same apples it would have without thinking.
    """
    with torch.no_grad():
        q = agent.net(torch.tensor(game.get_state(), dtype=torch.float32)).tolist()
    sim = copy.copy(game)  # shallow is enough: step() replaces snake/direction, never mutates them
    sim.window = None
    rng = random.getstate()
    plan = []
    for _ in range(steps):
        _, reward, terminated, _ = sim.step(agent.act(sim.get_state(), explore=False))
        if terminated:
            break
        plan.append(sim.snake[0])
        if reward > 0:
            break
    random.setstate(rng)
    return plan, q


def run(policy, games, max_steps_without_food=100, render=False, fps=15, cells=20, thinker=None):
    """Play `games` episodes with `policy(state) -> action`. No learning happens here.

    thinker(game) -> (plan, q) is drawn on top of the board when rendering.
    """
    game = SnakeGame(render=render, max_steps_without_food=max_steps_without_food, cells=cells)
    if render:
        import pygame
        clock = pygame.time.Clock()
    scores, ends = [], {"died": 0, "truncated": 0, "won": 0}
    for _ in range(games):
        state = game.reset()
        while True:
            state, _, terminated, truncated = game.step(policy(state))
            if render:
                game.render(*(thinker(game) if thinker and not game.game_over else ()))
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
    p.add_argument("--think", type=int, default=0, metavar="N",
                   help="with --watch: draw the next N moves the model plans + its Q-values")
    p.add_argument("--max-steps-without-food", type=int, default=100)
    p.add_argument("--cells", type=int, default=20, help="board size (cells per side)")
    a = p.parse_args()

    report("random", *run(lambda s: random.randrange(3), a.games, a.max_steps_without_food, cells=a.cells))
    if a.model:
        agent = Agent.load(a.model)
        agent.net.eval()
        # explore=False: always take the highest Q-value. We never call remember/train_step,
        # so the agent and its (empty) replay memory are untouched.
        policy = lambda s: agent.act(s, explore=False)
        thinker = (lambda g: think(agent, g, a.think)) if a.think else None
        report("model", *run(policy, a.games, a.max_steps_without_food, a.watch, a.fps, a.cells, thinker))
