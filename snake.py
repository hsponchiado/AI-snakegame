import random
import numpy
import torch
import pygame
from sys import exit

cells = 20
tile_size = 35
game_width = cells * tile_size
game_height = cells * tile_size

pygame.init()
window = pygame.display.set_mode((game_width, game_height))
pygame.display.set_caption("AI Trained Snake Game")
clock = pygame.time.Clock()

# snake is a list of (col, row) cells, head first
start_snake = [(10, 10), (9, 10), (8, 10)]
snake = start_snake
direction = (1, 0)
score = 0
game_over = False
font = pygame.font.Font(None, 40)

def new_apple():
    free = [(c, r) for c in range(cells) for r in range(cells) if (c, r) not in snake]
    return random.choice(free)

apple = new_apple()

keys = {
    pygame.K_w: (0, -1), pygame.K_UP: (0, -1),
    pygame.K_s: (0, 1), pygame.K_DOWN: (0, 1),
    pygame.K_a: (-1, 0), pygame.K_LEFT: (-1, 0),
    pygame.K_d: (1, 0), pygame.K_RIGHT: (1, 0),
}

def draw():
    window.fill("#6CBB3C")
    for x in range(0, game_width, tile_size):
        pygame.draw.line(window, "#5AA832", (x, 0), (x, game_height))
    for y in range(0, game_height, tile_size):
        pygame.draw.line(window, "#5AA832", (0, y), (game_width, y))
    pygame.draw.rect(window, "red", (apple[0] * tile_size, apple[1] * tile_size, tile_size, tile_size))
    for col, row in snake:
        pygame.draw.rect(window, (5,240,150), (col * tile_size, row * tile_size, tile_size, tile_size))
    # eyes: two dots on the head, facing the way it's moving
    dx, dy = snake[0][0] - snake[1][0], snake[0][1] - snake[1][1]
    cx, cy = snake[0][0] * tile_size + tile_size // 2, snake[0][1] * tile_size + tile_size // 2
    q = tile_size // 4
    for side in (1, -1):
        pygame.draw.circle(window, "black", (cx + dx * q - dy * q * side, cy + dy * q + dx * q * side), 3)
    window.blit(font.render(f"Score: {score}", True, "white"), (10, 10))
    if game_over:
        shade = pygame.Surface((game_width, game_height), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 150))
        window.blit(shade, (0, 0))
        for i, line in enumerate(["GAME OVER", f"Score: {score}", "Press Space to play again"]):
            text = font.render(line, True, "white")
            window.blit(text, text.get_rect(center=(game_width // 2, game_height // 2 - 40 + i * 40)))

while True:
    last = direction
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if game_over:
                if event.key == pygame.K_SPACE:
                    snake = start_snake
                    direction = last = (1, 0)
                    apple = new_apple()
                    score = 0
                    game_over = False
            elif event.key in keys:
                new = keys[event.key]
                # can't turn straight back into the body
                if (new[0] + last[0], new[1] + last[1]) != (0, 0):
                    direction = new
    if not game_over:
        head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        if not (0 <= head[0] < cells and 0 <= head[1] < cells):  # hit a wall
            game_over = True
        elif head == apple:
            snake = [head] + snake  # keep the tail, so the snake grows by one
            apple = new_apple()
            score += 1
        else:
            snake = [head] + snake[:-1]
        if head in snake[1:]:
            game_over = True
    draw()
    pygame.display.update()
    clock.tick(8) # snake moves 10 cells per second
