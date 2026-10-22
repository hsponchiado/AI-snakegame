"""Run with: python test_snake.py"""
import os
import random
import tempfile

import torch

from agent import Agent, Config
from snake import CELLS, LEFT, RIGHT, STRAIGHT, SnakeGame, to_action
from train import train


def test_wall_collision():
    g = SnakeGame()
    g.snake, g.direction, g.apple = [(CELLS - 1, 5), (CELLS - 2, 5), (CELLS - 3, 5)], (1, 0), (0, 0)
    assert g.get_state()[0] == 1  # danger straight
    _, reward, terminated, truncated = g.step(STRAIGHT)
    assert (reward, terminated, truncated) == (-10, True, False)


def test_self_collision():
    g = SnakeGame()
    # head at (5,5) moving left; body wraps around below it so turning left (south) hits (5,6)
    g.snake = [(5, 5), (6, 5), (6, 6), (5, 6), (4, 6)]
    g.direction, g.apple = (-1, 0), (0, 0)
    assert g.get_state()[2] == 1  # danger left
    _, reward, terminated, _ = g.step(LEFT)
    assert reward == -10 and terminated


def test_moving_into_vacated_tail_is_safe():
    g = SnakeGame()
    # 2x2 loop: the head moves into the cell the tail is leaving this same step
    g.snake = [(5, 5), (6, 5), (6, 6), (5, 6)]
    g.direction, g.apple = (-1, 0), (0, 0)
    assert g.get_state()[2] == 0  # left (south, the tail cell) is not danger
    _, reward, terminated, _ = g.step(LEFT)
    assert not terminated and g.snake[0] == (5, 6)


def test_eating_grows_and_rewards():
    g = SnakeGame()
    g.apple = (11, 10)
    _, reward, terminated, _ = g.step(STRAIGHT)
    assert reward == 10 and not terminated and len(g.snake) == 4 and g.score == 1


def test_win_by_filling_board():
    g = SnakeGame()
    # snake fills every cell except (0, 0); head at (1, 0) facing left
    # zig-zag body, head first: row 0 from (1,0) rightward, row 1 leftward, row 2 rightward, ...
    order = lambda r: range(1, CELLS) if r == 0 else range(CELLS - 1, -1, -1) if r % 2 else range(CELLS)
    g.snake = [(c, r) for r in range(CELLS) for c in order(r)]
    g.apple, g.direction = (0, 0), (-1, 0)
    assert g.snake[0] == (1, 0) and g.apple == (0, 0)
    _, reward, terminated, truncated = g.step(STRAIGHT)
    assert reward == 10 and terminated and not truncated and g.won


def test_truncation_after_steps_without_food():
    g = SnakeGame(max_steps_without_food=3)
    g.apple = (0, 0)
    results = [g.step(LEFT if i % 2 else RIGHT)[2:] for i in range(3)]  # zigzag, no wall in reach
    assert results == [(False, False), (False, False), (False, True)]


def test_state_and_actions():
    g = SnakeGame()
    g.apple = (15, 3)  # right of and above the head (10, 10)
    assert g.get_state() == [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    assert to_action((1, 0), (0, 1)) == RIGHT
    assert to_action((1, 0), (0, -1)) == LEFT
    assert to_action((1, 0), (-1, 0)) == STRAIGHT  # U-turn ignored


def test_target_does_not_bootstrap_on_terminal():
    agent = Agent(Config(train_start=1, batch_size=1))
    for p in agent.target.parameters():
        torch.nn.init.constant_(p, 100.0)  # huge next-state Q-values
    agent.memory.append(([0] * 11, 0, -10, [1] * 11, True))
    q_before = agent.net(torch.zeros(11))[0].item()
    loss = agent.train_step()
    assert abs(loss - (q_before + 10) ** 2) < 1e-3  # target was exactly -10, no gamma * next_q


def test_training_runs_and_checkpoint_loads():
    random.seed(0)
    path = os.path.join(tempfile.mkdtemp(), "m.pth")
    agent = train(episodes=30, save_path=path, seed=0, config=Config(train_start=64), log=False)
    assert agent.updates > 0 and agent.epsilon < 1.0
    loaded = Agent.load(path)
    s = [0] * 11
    assert loaded.act(s, explore=False) in (0, 1, 2)
    assert len(loaded.memory) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
