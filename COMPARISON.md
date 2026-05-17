# popgymnax ↔ popgym Equivalence Report (post-fix)

This document is the **verification pass** after applying fixes for all bugs catalogued
in the initial comparison. Each environment in **popgymnax** (the JAX port, forked from
[EdanToledo/popjym](https://github.com/EdanToledo/popjym)) was re-compared against the
reference **popgym** (numpy/gymnasium) at
[proroklab/popgym](https://github.com/proroklab/popgym).

**Method**: 11 parallel agents independently re-checked their respective env family,
verifying each claimed fix and looking for any remaining or new discrepancies.

**Convention** (unchanged from initial pass):
- Encoding differences that preserve information (one-hot vs index, `MultiDiscrete` vs
  flat `Discrete`, scalar vs `Box((1,))`) are **not** flagged as bugs.
- popgym's `obs_requires_prev_action = True` contract is honored centrally: envs
  declaring this attribute are auto-wrapped with `AliasPrevActionV2` by
  `popgymnax.make()`, which concatenates the previous action one-hot to the observation.
  A warning prints on wrap.
- The `terminated`/`truncated` split → single `done` collapse is intentional (gymnax
  convention) and not re-flagged.

---

## Summary table

| Env family | Status (post-fix) | Notes |
|---|---|---|
| Autoencode | ✅ equivalent | Fixed (initial fix was off-by-one; second-pass fix completes it) |
| Battleship | ✅ equivalent | Placement now uniform over 4 directions; prev_action wrapped |
| Concentration | ✅ equivalent | episode_length int; prev_action wrapped |
| CountRecall | ✅ equivalent | All 5 bugs (action_space, timing, reset, scale, length) verified fixed |
| HigherLower | ✅ equivalent | Off-by-one fixed |
| MineSweeper | ✅ equivalent | obs size, win condition, mine preservation, off-by-one, prev_action all verified |
| MultiArmedBandit | ✅ equivalent | prev_action wrapped; info dict exposes bandits |
| RepeatFirst | ✅ equivalent | Off-by-one fixed |
| RepeatPrevious | ✅ equivalent | Lag shifted to k-1; off-by-one fixed |
| CartPole | ✅ equivalent | Reward simplified; conditional clipping |
| Pendulum | ✅ equivalent | Noisy obs clipped to bounds |

**Result**: 11/11 env families are now semantically equivalent to popgym across all 3
difficulty variants (where applicable). Only "differences" remaining are info-preserving
encoding choices (one-hot vs index, Box vs Discrete) which the user explicitly accepted.

---

## Orphan environments (carried over from initial pass)

- **`popgymnax/environments/meta_cartpole.py`** (`NoisyStatelessMetaCartPole`): no
  counterpart in `popgym/envs/`. Originally added in popjym for meta-RL experiments.
  Treated as popgymnax-original; not part of the popgym benchmark.
- **`popgym/envs/labyrinth_escape.py` / `labyrinth_explore.py`**: present in upstream
  popgym, **not ported** to popgymnax.
- **`popgym/envs/velocity_only_cartpole.py`**: third CartPole flavor (velocity exposed,
  position hidden), **not ported** to popgymnax. `popgym_cartpole.py` only covers
  `position_only_*` and `noisy_position_only_*`.

---

## Autoencode

**Status**: ✅ equivalent

### Files
- popgym (reference): `~/popgym/popgym/envs/autoencode.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_autoencode.py`

### Fixes verified
- **Bug 1 (obs inversion, CRITICAL)** — `popgym_autoencode.py:78`: `jnp.where(play,
  play_obs, watch_obs)` correctly returns the card during WATCH and zeros during PLAY,
  matching popgym's mode semantics (`autoencode.py:80-94`).
- **Bug 2 (OOB during PLAY, HIGH)** — `popgym_autoencode.py:77`: `state.cards[state.timestep
  % num_cards]` avoids OOB when `state.timestep >= num_cards` (modulo defensive, value
  unused since `play=True` selects `play_obs`).
- **Bug 3 (episode length off-by-one, MEDIUM)** — second-pass fix at
  `popgym_autoencode.py:40-41,45`:
  - `terminated = state.timestep == num_cards * 2 - 2`
  - `play = state.timestep >= num_cards - 1` (reward gate, uses pre-step state)
  - PLAY index shifted to `flip(cards)[state.timestep - (num_cards - 1)]`
  - `get_obs` is unchanged (uses post-step state; the `>= num_cards` threshold there is
    still correct since `new_state.timestep` is one larger than the reward gate's input).

  Smoke-tested: Easy=103 steps, Medium=207 steps, Hard=311 steps — all match popgym's
  `2*num_cards - 1`.

### Remaining discrepancies
_None._

### Notes
- Reward magnitude `±1/num_cards` matches.
- Reverse-order recall: agent observes cards[0..N-1] during WATCH and recalls them as
  cards[N-1], cards[N-2], …, cards[0] during PLAY — matches popgym's LIFO
  `system.pop(-1)`.

---

## Battleship

**Status**: ✅ equivalent

### Files
- popgym (reference): `~/popgym/popgym/envs/battleship.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_battleship.py`

### Fixes verified
- **Bug 1 (ship placement bias, MEDIUM)** — `popgym_battleship.py:15,41,50`: rewrote
  `is_valid_placement` and `place_ship_on_board` with
  `deltas = jnp.array([[0,1],[0,-1],[1,0],[-1,0]])` for all 4 directions; placement is
  sampled uniformly via `jax.random.choice(..., p=valid_spots.flatten())`. Distributionally
  equivalent to popgym's rejection-sampling-with-`(-1)**direction[1]`.
- **Bug 2 (prev_action contract)** — `popgym_battleship.py:89`:
  `obs_requires_prev_action = True` declared on the `Battleship` base class; central
  wrapper in `registration.py` auto-applies `AliasPrevActionV2`.

### Remaining discrepancies
_None._ (Info-preserving encoding diffs only: `MultiDiscrete` → flat `Discrete`,
`Discrete(2)` → `Box((1,))`.)

### Notes
- Reward: `+1/needed_hits` on hit, `-1/(max_steps - needed_hits)` on miss; repeated guesses
  treated as misses. Identical to popgym.
- All 3 difficulty variants (board_size 8/10/12, ship_sizes `[2,3,3,4]`) match.

---

## Concentration

**Status**: ✅ equivalent

### Files
- popgym (reference): `~/popgym/popgym/envs/concentration.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_concentration.py`

### Fixes verified
- **Bug 1 (episode_length as jnp float, LOW)** — `popgym_concentration.py:33-35`:
  wrapped in `int(...)` so `timestep == episode_length` comparison is robust.
- **Bug 2 (prev_action contract)** — `popgym_concentration.py:24`:
  `obs_requires_prev_action = True` declared on base class.
- **Bug 3 (stray comment)** — removed.

### Remaining discrepancies
_None._

### Notes
- All reward branches (match success, mismatch, already-up flip, same-index-twice) match
  popgym.
- Difficulty variants match: `(num_decks=1, num_types=2)`, `(2, 2)`, `(1, 13)`.

---

## CountRecall

**Status**: ✅ equivalent (was the worst offender — now clean)

### Files
- popgym (reference): `~/popgym/popgym/envs/count_recall.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_count_recall.py`

### Fixes verified
- **Bug 1 (action_space hardcoded `Discrete(2)`, CRITICAL)** —
  `popgym_count_recall.py:79`: `spaces.Discrete(self.max_num + 1)`. Easy/Medium/Hard
  yield 27/27/17 as expected.
- **Bug 2 (step timing, HIGH)** — `popgym_count_recall.py:41-43`: `prev_count` is read
  from `running_count[query_cards[timestep]]` BEFORE the new value is added at line 43.
  Mirrors popgym's `prev_query`/`prev_count`-then-deal-then-`counts +=` ordering.
- **Bug 3 (reset count init, MEDIUM)** — `popgym_count_recall.py:60`:
  `running_count = jnp.zeros(...).at[value_cards[0]].add(1)`. Matches popgym's
  `self.counts[self.value] += 1` after the initial deal (`count_recall.py:140`).
- **Bug 4 (reward scale, LOW)** — `popgym_count_recall.py:31`: `1.0/(num_cards - 1)`,
  matches popgym `1/max_episode_length`.
- **Bug 5 (episode length, MEDIUM)** — `popgym_count_recall.py:50`:
  `new_state.timestep == self.num_cards - 1`. Total step count matches.

### Remaining discrepancies
_None._

### Notes
- `error_clamp` correctly absent (popgym's docstring advertises it but the implementation
  never applies it — port matches actual popgym behavior, not the docstring).
- Information-equivalent obs encoding (one-hot vs `MultiDiscrete`).

---

## HigherLower

**Status**: ✅ equivalent

### Files
- popgym (reference): `~/popgym/popgym/envs/higher_lower.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_higherlower.py`

### Fixes verified
- **Bug 1 (off-by-one + OOB, HIGH)** — `popgym_higherlower.py:45`:
  `terminated = new_state.timestep == num_cards - 1`. Final step reads cards in-bounds;
  total step count `num_cards - 1` matches popgym's `len(deck) <= 1` termination.

### Remaining discrepancies
_None._

### Notes
- Reward formula, tie handling (tie=0), action semantics (`0 = higher`) all match.
- Difficulty variants match: `num_decks = 1/2/3`.

---

## MineSweeper

**Status**: ✅ equivalent

### Files
- popgym (reference): `~/popgym/popgym/envs/minesweeper.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_minesweeper.py`

### Fixes verified
- **Bug 1 (obs size undersized, CRITICAL)** — `popgym_minesweeper.py:34`:
  `self.obs_size = min(self.num_mines + 1, 10)`. Reused at `:60`, `:90`, `:99-102`.
  Matches popgym `Discrete(min(num_mines + 1, 10))`. Neighbor counts 0..min(8, num_mines)
  all fit.
- **Bug 2 (win condition unreachable, HIGH)** — `popgym_minesweeper.py:58`:
  `jnp.all(new_grid != 0)` (win when no CLEAR cells remain).
- **Bug 3 (mine cell overwritten, MEDIUM)** — `popgym_minesweeper.py:47-49`:
  `jnp.where(mine, state.mine_grid, state.mine_grid.at[action].set(2))`. Mine stays as 1
  on hit. Cell encoding (0=CLEAR, 1=MINE, 2=VIEWED) consistent across reset, step,
  mine-hit branch, and win check.
- **Bug 4 (timestep off-by-one, MEDIUM)** — `popgym_minesweeper.py:55-56`: agent gets
  full `max_episode_length` steps.
- **Bug 5 (prev_action contract, HIGH)** — `popgym_minesweeper.py:24`:
  `obs_requires_prev_action = True` declared; central wrapper applies.

### Remaining discrepancies
_None._

### Notes
- Mine placement uses `jax.random.choice(replace=False)`; distribution matches popgym.
- Neighbor convolution via `scipy.signal.convolve2d` equivalent to popgym's roll-based
  3x3 sum.

---

## MultiArmedBandit

**Status**: ✅ equivalent (was unlearnable — now learnable)

### Files
- popgym (reference): `~/popgym/popgym/envs/multiarmed_bandit.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_multiarmedbandit.py`

### Fixes verified
- **Bug 1 (missing prev_action, HIGH)** — `popgym_multiarmedbandit.py:22`:
  `obs_requires_prev_action = True` declared; central wrapper applies
  `AliasPrevActionV2`. Agent can now associate rewards with arms — task is learnable.
- **Bug 2 (empty info dict, LOW)** — `popgym_multiarmedbandit.py:46`:
  returns `{"bandits": state.payouts}`.

### Remaining discrepancies
_None._

### Notes
- All 3 difficulty variants match: `(num_bandits, episode_length)` =
  `(10,200)/(20,400)/(30,600)`.
- Bernoulli sampling logic equivalent (popgym `np_random.random()` < arm,
  popgymnax `jax.random.uniform()` < arm).

---

## RepeatFirst

**Status**: ✅ equivalent

### Files
- popgym (reference): `~/popgym/popgym/envs/repeat_first.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_repeat_first.py`

### Fixes verified
- **Bug 1 (episode length, LOW)** — `popgym_repeat_first.py:40`:
  `terminated = new_state.timestep == num_cards - 1`. Step count matches popgym
  (`num_cards - 1`).
- **Bug 2 (modular wrap leaking answer, LOW)** — `popgym_repeat_first.py:60`:
  `% num_cards` removed; with bug 1 fix, indices stay in-bounds.

### Remaining discrepancies
_None._

### Notes
- Reward scale `±1/(num_cards - 1)` matches; target card is fixed first dealt card.
- Difficulty variants match: `num_decks = 1/8/16`.

---

## RepeatPrevious

**Status**: ✅ equivalent

### Files
- popgym (reference): `~/popgym/popgym/envs/repeat_previous.py`
- popgymnax (patched): `~/popgymnax/popgymnax/environments/popgym_repeat_previous.py`

### Fixes verified
- **Bug 1 (lag k vs k-1, HIGH)** — `popgym_repeat_previous.py:41,43`: shifted both index
  and gate by one. Query is `cards[state.timestep - self.k + 1]`; gate is
  `state.timestep >= self.k - 1`. For k=4: at `state.timestep=3` the agent recalls
  `cards[0]` having seen `cards[0..3]` (lag = 3 = k-1, matching popgym).
- **Bug 2 (episode length, LOW)** — `popgym_repeat_previous.py:45`: terminate at
  `new_state.timestep == num_cards - 1`.
- **Bug 3 (terminal OOB)** — resolved by bug 2.

### Remaining discrepancies
_None._

### Notes
- Reward scale `±1/(num_cards - k)` matches.
- Difficulty variants match: `(num_decks, k)` = `(1, 4)`, `(2, 32)`, `(3, 64)`.

---

## CartPole

**Status**: ✅ equivalent

### Files
- popgym (reference): `position_only_cartpole.py`, `noisy_position_only_cartpole.py`
  (`velocity_only_cartpole.py` has no JAX port — flagged)
- popgymnax (patched): `popgym_cartpole.py`

### Fixes verified
- **Bug 1 (spurious `-1.0` post-termination, HIGH)** — `popgym_cartpole.py:73`:
  `reward = 1.0 / self.max_steps_in_episode` is unconditional; `prev_terminal` removed
  from `EnvState`; `reward_transform` removed. Matches popgym's actual behavior (constant
  `+1/max_episode_length` per step; popgym's `-1` branch was dead code under default
  gymnasium settings).
- **Bug 2 (clipping when noise_sigma=0, LOW)** — `popgym_cartpole.py:103-109`: noise +
  clipping gated by `if self.noise_sigma > 0`. Stateless variants now return raw
  `[x, theta]` without clipping, matching popgym.

### Remaining discrepancies
_None._

### Notes
- Physics constants identical to gymnasium CartPoleEnv (gravity=9.8, masscart=1.0,
  masspole=0.1, length=0.5, polemass_length=0.05, force_mag=10.0, tau=0.02).
- Forward Euler integration order matches.
- Reset distribution `U(-0.05, 0.05, shape=(4,))` matches.
- Mild RNG coupling in `reset_env` (init-state key reused for obs noise) — informational
  only, not a behavioral divergence.

---

## Pendulum

**Status**: ✅ equivalent

### Files
- popgym (reference): `position_only_pendulum.py`, `noisy_position_only_pendulum.py`
- popgymnax (patched): `popgym_pendulum.py`

### Fixes verified
- **Bug 1 (noisy obs not clipped, MEDIUM)** — `popgym_pendulum.py:115`:
  `jnp.clip(obs, -1.0, 1.0)` applied after adding noise. Matches popgym's
  `np.clip(obs + noise, obs_space.low, obs_space.high)`.

### Remaining discrepancies
_None._

### Notes
- Reward formula (raw cost and shift/scale transform) matches gymnasium PendulumEnv.
- Semi-implicit Euler integration matches.
- Physics constants identical (g=10, m=1, l=1, dt=0.05, max_speed=8, max_torque=2.0).
- Reset distribution `theta ~ U(-π, π), theta_dot ~ U(-1, 1)` matches.

---

## Cross-cutting patterns (resolved)

The systemic issues from the initial pass have been addressed:

1. **Off-by-one episode length** in card-based envs (CountRecall, HigherLower,
   RepeatFirst, RepeatPrevious, Autoencode): all fixed. JAX OOB clamping no longer
   leaks data on terminal steps.

2. **Missing `obs_requires_prev_action` handling**: a class attribute now signals the
   contract; `popgymnax.make()` auto-wraps with `AliasPrevActionV2` and prints
   ```
   [popgymnax] <env> declares obs_requires_prev_action=True; auto-wrapping with
   AliasPrevActionV2 (observation includes prev action).
   ```
   Applied to Battleship, Concentration, MineSweeper, MultiArmedBandit.

3. **`truncated` collapsed into `terminated`**: intentionally left as the gymnax
   convention; documented but not changed.

4. **Silent OOB clamping**: each card-based env's off-by-one fix eliminates the OOB
   reads that JAX was silently clamping.

5. **Dead `error_clamp` in CountRecall**: confirmed popgym never applies it either; port
   matches actual reference behavior.
