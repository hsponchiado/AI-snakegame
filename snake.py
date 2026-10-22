import os
import random

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

CELLS = 20
TILE = 35
SIZE = CELLS * TILE

# Relative actions: the agent says "keep going / turn" instead of "go north".
# This gives 3 choices instead of 4, and none of them is an illegal U-turn.
STRAIGHT, RIGHT, LEFT = 0, 1, 2

# Rewards
FOOD_REWARD = 10
DEATH_REWARD = -10
MOVE_REWARD = 0


def turn_right(d):
    # Screen coordinates: y grows downward, so right of (1, 0) is (0, 1).
    return (-d[1], d[0])


def turn_left(d):
    return (d[1], -d[0])


class SnakeGame:
    """Snake rules only. No keyboard, no clock. Drawing happens only if render=True."""

    def __init__(self, render=False, max_steps_without_food=100):
        # Episode is truncated after this many moves without eating (None = no limit).
        self.max_steps_without_food = max_steps_without_food
        self.window = None
        if render:
            pygame.init()
            self.window = pygame.display.set_mode((SIZE, SIZE))
            pygame.display.set_caption("AI Trained Snake Game")
            self.font = pygame.font.Font(None, 40)
        self.reset()

    def reset(self):
        # snake is a list of (col, row) cells, head first
        self.snake = [(10, 10), (9, 10), (8, 10)]
        self.direction = (1, 0)
        self.score = 0
        self.steps_since_food = 0
        self.game_over = False
        self.won = False
        self.apple = self._new_apple()
        return self.get_state()

    def _new_apple(self):
        taken = set(self.snake)
        free = [(c, r) for c in range(CELLS) for r in range(CELLS) if (c, r) not in taken]
        return random.choice(free) if free else None

    def _hits(self, cell):
        """Would the head die by entering this cell on the next move?"""
        c, r = cell
        if not (0 <= c < CELLS and 0 <= r < CELLS):
            return True
        if cell == self.apple:
            return cell in self.snake  # eating: tail stays (apple is never on the snake anyway)
        return cell in self.snake[:-1]  # normal move: the tail cell gets vacated, so it's safe

    def step(self, action):
        """Apply one relative action. Returns (state, reward, terminated, truncated).

        terminated = the game really ended (death or win): no future rewards exist.
        truncated  = we stopped it artificially (too long without food): the game
                     could have continued, so training should still bootstrap.
        """
        if action == RIGHT:
            self.direction = turn_right(self.direction)
        elif action == LEFT:
            self.direction = turn_left(self.direction)

        head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.steps_since_food += 1

        if self._hits(head):
            self.game_over = True
            return self.get_state(), DEATH_REWARD, True, False

        if head == self.apple:
            self.snake = [head] + self.snake  # keep the tail, so the snake grows by one
            self.score += 1
            self.steps_since_food = 0
            self.apple = self._new_apple()
            if self.apple is None:  # no free cell left: board filled
                self.game_over = self.won = True
                return self.get_state(), FOOD_REWARD, True, False
            return self.get_state(), FOOD_REWARD, False, False

        self.snake = [head] + self.snake[:-1]
        limit = self.max_steps_without_food
        truncated = limit is not None and self.steps_since_food >= limit
        return self.get_state(), MOVE_REWARD, False, truncated

    def get_state(self):
        """11 numbers (0 or 1) describing what the snake 'sees'."""
        hx, hy = self.snake[0]
        d = self.direction
        ahead = lambda dd: (hx + dd[0], hy + dd[1])
        ax, ay = self.apple if self.apple else (hx, hy)
        features = [
            # danger one cell away, relative to where the snake faces
            self._hits(ahead(d)),
            self._hits(ahead(turn_right(d))),
            self._hits(ahead(turn_left(d))),
            # current direction (one-hot)
            d == (-1, 0), d == (1, 0), d == (0, -1), d == (0, 1),
            # where the food is compared to the head
            ax < hx, ax > hx, ay < hy, ay > hy,
        ]
        return [int(f) for f in features]

    def render(self):
        if self.window is None:
            return
        w = self.window
        w.fill("#6CBB3C")
        for x in range(0, SIZE, TILE):
            pygame.draw.line(w, "#5AA832", (x, 0), (x, SIZE))
            pygame.draw.line(w, "#5AA832", (0, x), (SIZE, x))
        if self.apple:
            pygame.draw.rect(w, "red", (self.apple[0] * TILE, self.apple[1] * TILE, TILE, TILE))
        for col, row in self.snake:
            pygame.draw.rect(w, (5, 240, 150), (col * TILE, row * TILE, TILE, TILE))
        # eyes: two dots on the head, facing the way it's moving
        dx, dy = self.snake[0][0] - self.snake[1][0], self.snake[0][1] - self.snake[1][1]
        cx, cy = self.snake[0][0] * TILE + TILE // 2, self.snake[0][1] * TILE + TILE // 2
        q = TILE // 4
        for side in (1, -1):
            pygame.draw.circle(w, "black", (cx + dx * q - dy * q * side, cy + dy * q + dx * q * side), 3)
        w.blit(self.font.render(f"Score: {self.score}", True, "white"), (10, 10))
        if self.game_over:
            shade = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 150))
            w.blit(shade, (0, 0))
            title = "YOU WIN" if self.won else "GAME OVER"
            for i, line in enumerate([title, f"Score: {self.score}", "Press Space to play again"]):
                text = self.font.render(line, True, "white")
                w.blit(text, text.get_rect(center=(SIZE // 2, SIZE // 2 - 40 + i * 40)))
        pygame.display.update()


def to_action(direction, wanted):
    """Convert an absolute direction (from a key press) into a relative action."""
    if wanted == turn_right(direction):
        return RIGHT
    if wanted == turn_left(direction):
        return LEFT
    return STRAIGHT  # same direction, or a U-turn (ignored)


def play_human():
    keys = {
        pygame.K_w: (0, -1), pygame.K_UP: (0, -1),
        pygame.K_s: (0, 1), pygame.K_DOWN: (0, 1),
        pygame.K_a: (-1, 0), pygame.K_LEFT: (-1, 0),
        pygame.K_d: (1, 0), pygame.K_RIGHT: (1, 0),
    }
    game = SnakeGame(render=True, max_steps_without_food=None)
    clock = pygame.time.Clock()
    while True:
        wanted = game.direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if game.game_over and event.key == pygame.K_SPACE:
                    game.reset()
                    wanted = game.direction
                elif event.key in keys:
                    wanted = keys[event.key]
        if not game.game_over:
            game.step(to_action(game.direction, wanted))
        game.render()
        clock.tick(8)  # snake moves 8 cells per second


if __name__ == "__main__":
    play_human()
