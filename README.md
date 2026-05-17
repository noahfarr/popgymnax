<p align="center">
    <a href="docs/images/popjym.png">
        <img src="docs/images/popjym.png" alt="popgymnax logo" width="30%"/>
    </a>
</p>

# popgymnax: Partially Observable Process Gym in JAX

popgymnax is a fork of [POPJym](https://github.com/EdanToledo/popjym) (itself a JAX
port of [POPGym](https://openreview.net/forum?id=chDrutUTs0K)) that fixes a set of
semantic-equivalence bugs found by comparing each environment against the original
[proroklab/popgym](https://github.com/proroklab/popgym). The full bug catalog and
per-env verification is in [COMPARISON.md](COMPARISON.md).

## What this fork changes vs. upstream POPJym

The upstream JAX port has several environments that diverge from the original POPGym
reference — sometimes silently, sometimes catastrophically (e.g. an env that was
unlearnable due to a hard-coded action space, or rewards that leak the answer to the
agent). popgymnax patches all 11 env families × 3 difficulty variants so that the
behavior matches the numpy reference. Headline fixes:

- **CountRecall** — `action_space` was hard-coded `Discrete(2)` regardless of
  difficulty; now scales with the deck (`Discrete(max_num + 1)`). Plus fixes for
  step timing, reset count init, reward scale, and episode length.
- **Autoencode** — observation mode was inverted (the answer was shown during PLAY
  and zeros during WATCH); fixed plus an OOB read and an off-by-one in episode length.
- **MineSweeper** — observation array was sized to `num_mines` rather than
  `min(num_mines + 1, 10)` so neighbor counts were silently dropped on Easy; win
  condition was unreachable; mine cells were overwritten on hit. All fixed.
- **MultiArmedBandit** — was effectively unlearnable because the agent couldn't see
  its own previous action. The env now declares `obs_requires_prev_action = True` and
  `popgymnax.make()` auto-applies `AliasPrevActionV2`.
- **CartPole** — a spurious `-1.0` reward was emitted on the post-termination step
  that the reference never emits; dropped.
- **Pendulum** — noisy observations were not clipped to `[-1, 1]`; now clipped.
- **RepeatPrevious** — effective recall lag was `k` instead of `k - 1`.
- **HigherLower, RepeatFirst, Autoencode, CountRecall, RepeatPrevious** — all card
  envs had off-by-ones that produced one extra step and an OOB-clamped terminal
  observation (often leaking the answer).
- **Battleship** — ship placement was biased toward upper-left starts; now uniform
  over all 4 directions. Declares `obs_requires_prev_action`.
- **Concentration** — `episode_length` is now a Python `int` (was a `jnp` float).
- **Battleship / Concentration / MineSweeper / MultiArmedBandit** — all four declare
  `obs_requires_prev_action = True`. `popgymnax.make()` auto-wraps these with
  `AliasPrevActionV2` and prints a one-line warning so the wrapping is visible.

Infrastructure fixes (pre-existing issues in upstream popjym that prevented a clean
install on current dependencies):

- `wrappers.py` — replaced the broken `from gymnax.wrappers.purerl import …`
  re-exports (newer gymnax doesn't expose them) with direct imports from stdlib /
  chex / gymnax.environments.
- `meta_cartpole.py` — `jax.tree_map` → `jax.tree.map` (newer JAX).
- `requirements.txt` — added `chex` (used by `meta_cartpole.py` but never declared).

## Quickstart Install

```python
pip install popgymnax
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

For e.g.
```
pip install "jax[cuda12_pip]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quickstart Usage

```python
import jax
import popgymnax
seed = jax.random.PRNGKey(0)
env, env_params = popgymnax.make(env_name)

env.reset(seed, env_params)

env.step(seed, state, action)
```

# Contributing
Please follow the coding style by using pre-commit.

```python
pip install pre-commit
pre-commit install
```

# Citing

If used in your work, please cite **a)** the original POPGym paper and **b)** the Structured State Space Models for In-Context Reinforcement Learning paper:
```
@inproceedings{
morad2023popgym,
title={{POPG}ym: Benchmarking Partially Observable Reinforcement Learning},
author={Steven Morad and Ryan Kortvelesy and Matteo Bettini and Stephan Liwicki and Amanda Prorok},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=chDrutUTs0K}
}
```
```
@article{lu2023structured,
  title={Structured State Space Models for In-Context Reinforcement Learning},
  author={Lu, Chris and Schroecker, Yannick and Gu, Albert and Parisotto, Emilio and Foerster, Jakob and Singh, Satinder and Behbahani, Feryal},
  journal={arXiv preprint arXiv:2303.03982},
  year={2023}
}
```
