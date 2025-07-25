import argparse
from dataclasses import dataclass
from typing import Optional, Protocol, Iterable, final
import typing
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import pygame
import sys
import enum

class EventTag(int, enum.Enum):
    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    QUIT = enum.auto()


class Action(int, enum.Enum):
    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()

    @staticmethod
    def from_event_tag(event_tag: EventTag) -> Optional["Action"]:
        tag = event_tag.value
        if EventTag.UP.value <= tag <= EventTag.RIGHT.value:
            return Action(tag)
            
        return None

@dataclass
class Event[T]:
    tag: EventTag
    payload: T
    

class EventGetter[T](Protocol):
    def get(self) -> Iterable[Event[T]]: ...

@final
class PygameEventGetter(EventGetter[Event[None]]):
    def get(self) -> Iterable[Event[None]]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                yield Event(EventTag.QUIT, None)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    yield Event(EventTag.UP, None)
                elif event.key == pygame.K_DOWN:
                    yield Event(EventTag.DOWN, None)
                elif event.key == pygame.K_LEFT:
                    yield Event(EventTag.LEFT, None)
                elif event.key == pygame.K_RIGHT:
                    yield Event(EventTag.RIGHT, None)
                elif event.key == pygame.K_ESCAPE:
                    yield Event(EventTag.QUIT, None)


@final
class RandomEventGetter(EventGetter[Event[None]]):
    def get(self) -> Iterable[Event[None]]:
        yield from [
            Event(EventTag(np.random.randint(1, max(EventTag))), None)
            for _ in range(3)
        ]


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, grid_size=5):
        super().__init__()

        self.grid_size = grid_size
        self.window_size = 400  # pixels
        self.cell_size = self.window_size // self.grid_size

        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(6,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)  # up, down, left, right

        self.render_mode = render_mode
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Grid World")

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._player_pos = np.array([0, 0], dtype=np.int32)
        self._goal_pos = self._rand_pos(self._player_pos)
        self._danger_pos = self._rand_pos(self._player_pos, self._goal_pos)

        return self._get_obs(), {}

    def _is_in_exclude(self, needle: NDArray[np.int32], *haystack: NDArray[np.int32]) -> bool:
        for hay in haystack:
            if np.array_equal(needle, hay):
                return True
        return False

    def _rand_pos(self, *exclude: NDArray[np.int32]) -> NDArray[np.int32]:
        retry = 100
        while retry > 0:
            pos = np.array(
                [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size),
                ],
                dtype=np.int32)
            if not self._is_in_exclude(pos, *exclude):
                return pos

            retry -= 1
        raise RuntimeError("_rand_pos aborted after 100 retries")

    def step(self, action: Action):
        if action == Action.UP and self._player_pos[1] > 0:  # up
            self._player_pos[1] -= 1
        elif action == Action.DOWN and self._player_pos[1] < self.grid_size - 1:  # down
            self._player_pos[1] += 1
        elif action == Action.LEFT and self._player_pos[0] > 0:  # left
            self._player_pos[0] -= 1
        elif action == Action.RIGHT and self._player_pos[0] < self.grid_size - 1:  # right
            self._player_pos[0] += 1

        win = np.array_equal(self._player_pos, self._goal_pos)
        lose = np.array_equal(self._player_pos, self._danger_pos)
        terminated = win or lose
        if terminated:
            reward = 1 if win else -1
        else:
            reward = -0.01
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return np.concatenate([
            self._player_pos,
            self._goal_pos,
            self._danger_pos,
        ])

    def render(self):
        if self.render_mode != "human":
            return

        assert(self.window is not None)
        self.window.fill((255, 255, 255))


        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

        # Draw goal
        goal_rect = pygame.Rect(
            self._goal_pos[0] * self.cell_size,
            self._goal_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.window, (0, 255, 0), goal_rect)

        # Draw danger
        danger_rect = pygame.Rect(
            self._danger_pos[0] * self.cell_size,
            self._danger_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.window, (255, 0, 0), danger_rect)

        # Draw player
        player_rect = pygame.Rect(
            self._player_pos[0] * self.cell_size,
            self._player_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.window, (0, 0, 255), player_rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


# Manual test loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", choices=["pygame", "random"], default="random")
    args = parser.parse_args()

    pygame.init()

    env = GridWorldEnv(render_mode="human")
    obs, info = env.reset()
    if args.input == "pygame":
        event_getter = PygameEventGetter()
    else:
        event_getter = RandomEventGetter()
    total_reward = 0

    running = True
    while running:
        for event in event_getter.get():
            if event.tag == EventTag.QUIT:
                running = False
                break

            action = Action.from_event_tag(event.tag)

            if action is not None:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated:
                    if reward > 0:
                        print(f"You win!  {total_reward:.2f}")
                    else:
                        print(f"Game over! {total_reward:.2f}")
                    obs, info = env.reset()

        env.render()

    env.close()
    sys.exit()


if __name__ == "__main__":
    main()
