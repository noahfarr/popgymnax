from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces


def is_valid_placement(board, row, col, direction, ship_size):
    """Check if a placement is valid without modifying the board."""
    board_size = board.shape[0]
    offsets = jnp.arange(ship_size)
    deltas = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    drow, dcol = deltas[direction]
    rows = row + drow * offsets
    cols = col + dcol * offsets
    in_bounds = jnp.all(
        (rows >= 0) & (rows < board_size) & (cols >= 0) & (cols < board_size)
    )
    safe_rows = jnp.clip(rows, 0, board_size - 1)
    safe_cols = jnp.clip(cols, 0, board_size - 1)
    empty = jnp.all(board[safe_rows, safe_cols] == 0)
    return jnp.logical_and(in_bounds, empty)


vectorized_validity_check = jax.vmap(
    jax.vmap(
        jax.vmap(is_valid_placement, in_axes=(None, 0, None, None, None)),
        in_axes=(None, None, 0, None, None),
    ),
    in_axes=(None, None, None, 0, None),
)


def place_ship_on_board(board, row, col, direction, ship_size):
    """Place a ship on the board at the given position and direction."""
    board_size = board.shape[0]
    offsets = jnp.arange(ship_size)
    deltas = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    drow, dcol = deltas[direction]
    rows = jnp.clip(row + drow * offsets, 0, board_size - 1)
    cols = jnp.clip(col + dcol * offsets, 0, board_size - 1)
    return board.at[rows, cols].set(1.0)


def place_random_ship_on_board(rng, board, ship_size):
    size = board.shape[0]
    dirs = jnp.arange(4)
    rows = jnp.arange(size)
    cols = jnp.arange(size)
    valid_spots = vectorized_validity_check(board, rows, cols, dirs, ship_size)
    total_num_spots = np.prod(valid_spots.shape)
    rand_valid = jax.random.choice(
        rng, jnp.arange(total_num_spots), shape=(1,), p=valid_spots.flatten()
    )[0]
    direction, col, row = (
        rand_valid // (size * size),
        (rand_valid % (size * size)) // size,
        (rand_valid % (size * size)) % size,
    )
    board = place_ship_on_board(board, row, col, direction, ship_size)
    return board


def generate_random_board(rng, board_size, ship_sizes):
    board = jnp.zeros((board_size, board_size))
    for ship_size in ship_sizes:
        rng, _rng = jax.random.split(rng)
        board = place_random_ship_on_board(_rng, board, ship_size)
    return board


@struct.dataclass
class EnvState:
    timestep: int
    board: jnp.ndarray
    guesses: jnp.ndarray
    hits: int


@struct.dataclass
class EnvParams:
    pass


class Battleship(environment.Environment):
    obs_requires_prev_action = True

    def __init__(self, board_size=8):
        super().__init__()
        self.board_size = board_size
        self.ship_sizes = [2, 3, 3, 4]
        self.max_episode_length = self.board_size * self.board_size
        self.needed_hits = sum(self.ship_sizes)
        self.reward_hit = 1.0 / self.needed_hits
        self.reward_miss = -1.0 / (self.max_episode_length - self.needed_hits)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        action_x, action_y = action // self.board_size, action % self.board_size
        is_ship = state.board[action_x, action_y] == 1
        guessed_before = state.guesses[action_x, action_y] == 1
        hit = jnp.logical_and(is_ship, jnp.logical_not(guessed_before))
        new_guesses = state.guesses.at[action_x, action_y].set(1)
        new_timestep = state.timestep + 1
        new_hits = state.hits + hit

        terminated = jnp.logical_or(
            new_timestep >= self.max_episode_length,
            new_hits >= self.needed_hits,
        )

        obs = jnp.array([hit.astype(jnp.float32)])
        reward = jnp.where(hit, self.reward_hit, self.reward_miss)

        new_state = EnvState(
            timestep=new_timestep,
            board=state.board,
            guesses=new_guesses,
            hits=new_hits,
        )

        return obs, new_state, reward, terminated, {}

    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        board = generate_random_board(key, self.board_size, self.ship_sizes)
        guesses = jnp.zeros((self.board_size, self.board_size))

        state = EnvState(
            timestep=0,
            board=board,
            guesses=guesses,
            hits=0,
        )
        obs = jnp.array([0.0])

        return obs, state

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        # TODO: Multi-Discrete?
        return spaces.Discrete(self.board_size * self.board_size)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((1,)), jnp.ones((1,)), (1,), dtype=jnp.float32)


class BattleshipEasy(Battleship):
    def __init__(self):
        super().__init__(board_size=8)


class BattleshipMedium(Battleship):
    def __init__(self):
        super().__init__(board_size=10)


class BattleshipHard(Battleship):
    def __init__(self):
        super().__init__(board_size=12)
