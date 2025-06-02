import pygame
import random
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import time

WHITE = (255, 255, 255)
BLUE = (0,0,255)
RED = (255,0,0)
BLACK = (0, 0, 0)
GREEN = (0,255,0)

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 7

WINNING_SCORE = 50


class Paddle:
    VEL = 4

    def __init__(self, x, y, width, height, COLOR):
        self.COLOR = COLOR
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        pygame.draw.rect(
            win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y


class Ball:
    MAX_VEL = 5
    COLOR = GREEN

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = self.MAX_VEL
        self.y_vel = random.randint(-5,5)

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (self.x, self.y), self.radius)

    def move(self):
        if abs(self.y_vel) < 1:
            self.y_vel = random.choice([-1, 1]) * random.randint(2, 5)
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.y_vel = random.randint(-5,5)
        self.x_vel *= -1


class PongGame(Env):
    WIDTH, HEIGHT = 700, 500
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")

    FPS = 120

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        pygame.init()
        self.SCORE_FONT = pygame.font.SysFont("comicsans", 30)

        self.observation_space = Box(low=0, high=self.WIDTH, shape=(6,), dtype=np.uint32)
        self.action_space = Discrete(2)

        self.left_paddle = Paddle(10, self.HEIGHT//2 - PADDLE_HEIGHT //
                         2, PADDLE_WIDTH, PADDLE_HEIGHT, BLUE)
        self.right_paddle = Paddle(self.WIDTH - 10 - PADDLE_WIDTH, self.HEIGHT //
                            2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT, RED)
        self.ball = Ball(self.WIDTH // 2, self.HEIGHT // 2, BALL_RADIUS)

        self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.left_score = 0
        self.right_score = 0
        self.hits=0

        obs = self.get_observation()
        self.score = (self.left_score, self.right_score)
        info = {'left_score': self.left_score, 'right_score': self.right_score}

        
        self.ball.reset()
        self.left_paddle.reset()
        self.right_paddle.reset()

        if self.render_mode == 'human':
            self.render(self.WIN, [self.left_paddle, self.right_paddle], self.ball, self.left_score, self.right_score)
        
        return obs, info
        
    def step(self, action_L, action_R):
        
        self.handle_paddle_movement(self.left_paddle, action_L)
        self.handle_paddle_movement(self.right_paddle, action_R)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                break
        
        self.ball.move()
        reward_1 = 0
        reward_2=0
        reward_1, reward_2= self.handle_collision(self.ball, self.left_paddle, self.right_paddle)

        done_1, done_2 = self.get_done()
        done = False
        if done_1:
            reward_1 = -10
            reward_2 = 15 
            self.right_score+=1
            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            # done= True
        elif done_2:
            reward_1 = 15
            reward_2 = -10
            self.left_score+=1
            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            # done= True

        if self.left_score >= WINNING_SCORE or self.right_score >= WINNING_SCORE:
            self.reset()

        obs = self.get_observation()
        truncated =  False

        self.score = (self.left_score, self.right_score)
        info = {'left_score': self.left_score, 'right_score': self.right_score}

        if self.render_mode == 'human':
            self.render(self.WIN, [self.left_paddle, self.right_paddle], self.ball, self.left_score, self.right_score)

        return obs, [reward_1, reward_2], done, truncated, info
    
    def get_observation(self):
        distance_ball_p1= self.ball.x - self.left_paddle.x
        distance_ball_p2= self.ball.x - self.right_paddle.x
        state = [self.left_paddle.y, self.right_paddle.y, self.ball.y, abs(self.ball.y_vel), distance_ball_p1, distance_ball_p2]
        return np.array(state, dtype=np.int16)

    def get_done(self):
        if self.ball.x < 0:
            return True, False
        elif self.ball.x > self.WIDTH:
            return False, True
        return False, False

    def close(self):
        pygame.quit()

    def render(self, win, paddles, ball, left_score, right_score):
        if self.render_mode is None:
            return
        
        win.fill(BLACK)

        left_score_text = self.SCORE_FONT.render(f"{left_score}", 1, WHITE)
        right_score_text = self.SCORE_FONT.render(f"{right_score}", 1, WHITE)
        hits_text = self.SCORE_FONT.render(f"hits: {self.hits}",1, WHITE)
        win.blit(left_score_text, (self.WIDTH//4 - left_score_text.get_width()//2, 20))
        win.blit(right_score_text, (self.WIDTH * (3/4) -right_score_text.get_width()//2, 20))


        win.blit(hits_text, (self.WIDTH//4 - left_score_text.get_width()//2, 70))

        for paddle in paddles:
            paddle.draw(win)

        for i in range(10, self.HEIGHT, self.HEIGHT//40):
            if i % 2 == 1:
                continue
            pygame.draw.rect(win, WHITE, (self.WIDTH//2 - 2, i, 4, self.HEIGHT//60))

        ball.draw(win)
        pygame.display.update()
        self.clock.tick(self.FPS) # run this only while playing game


    def handle_collision(self, ball, left_paddle, right_paddle):
        if ball.y + ball.radius >= self.HEIGHT:
            ball.y_vel *= -1
            self.hits+=1
        elif ball.y - ball.radius <= 0:
            ball.y_vel *= -1
            self.hits+=1

        if ball.x_vel < 0:
            if ball.y >= left_paddle.y and ball.y <= left_paddle.y + left_paddle.height:
                if ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                    ball.x_vel *= -1

                    middle_y = left_paddle.y + left_paddle.height / 2
                    difference_in_y = middle_y - ball.y
                    reduction_factor = (left_paddle.height / 2) / ball.MAX_VEL
                    y_vel = difference_in_y / reduction_factor
                    ball.y_vel = -1 * y_vel
                    return 15,0

        else:
            if ball.y >= right_paddle.y and ball.y <= right_paddle.y + right_paddle.height:
                if ball.x + ball.radius >= right_paddle.x:
                    ball.x_vel *= -1

                    middle_y = right_paddle.y + right_paddle.height / 2
                    difference_in_y = middle_y - ball.y
                    reduction_factor = (right_paddle.height / 2) / ball.MAX_VEL
                    y_vel = difference_in_y / reduction_factor
                    ball.y_vel = -1 * y_vel
                    return 0,15
        return 0,0
    
    def handle_paddle_movement(self, paddle, action):
        if action ==0 and paddle.y - paddle.VEL >= 0:
            paddle.move(up=True)
        if action ==1 and paddle.y + paddle.VEL + paddle.height <= self.HEIGHT:
            paddle.move(up=False)

    def handle_paddle_movement_manual(self, paddle, keys):
        if (keys[pygame.K_w] or keys[pygame.K_UP]) and paddle.y - paddle.VEL >= 0:
            paddle.move(up=True)
        if (keys[pygame.K_s] or keys[pygame.K_DOWN]) and paddle.y + paddle.VEL + paddle.height <= self.HEIGHT:
            paddle.move(up=False)
