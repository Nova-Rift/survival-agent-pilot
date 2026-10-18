"""Small synchronous survival environment; no external services or training."""
from dataclasses import dataclass
import copy
import numpy as np

MOVES = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])

@dataclass(frozen=True)
class Config:
    size: int = 16
    agents: int = 8
    radius: int = 2
    max_hunger: int = 20
    initial_food: int = 24
    food_per_step: int = 1
    horizon: int = 128

class SurvivalWorld:
    """Moves into initially occupied cells fail, including swaps. Contested empty
    destinations reject all contenders. Food is consumed after simultaneous moves
    and hunger decrement, before death; then food spawns on empty, food-free cells.
    Reward is one per agent-step alive AFTER the transition, zero otherwise.
    Death is absorbing. Horizon is truncation, not a biological death.
    """
    def __init__(self, seed=0, config=Config()):
        limits={'size':(2,128),'agents':(1,256),'radius':(0,16),'max_hunger':(1,10000),'initial_food':(0,16384),'food_per_step':(0,16384),'horizon':(1,10000)}
        if any(type(getattr(config,k)) is not int or not lo<=getattr(config,k)<=hi for k,(lo,hi) in limits.items()):raise ValueError('Unsupported bounded world configuration')
        self.config = config
        if config.size < 2 or not 1 <= config.agents <= config.size**2:
            raise ValueError('Invalid population/grid')
        if min(config.radius, config.initial_food, config.food_per_step) < 0 or min(config.max_hunger, config.horizon) < 1:
            raise ValueError('Invalid budget')
        self.rng = np.random.default_rng(seed)
        cells = self.rng.choice(config.size**2, config.agents, replace=False)
        self.positions = np.stack(np.unravel_index(cells, (config.size, config.size)), axis=1)
        self.alive = np.ones(config.agents, dtype=bool)
        self.hunger = np.full(config.agents, config.max_hunger, dtype=np.int64)
        self.food = np.zeros((config.size, config.size), dtype=bool)
        self.tick = 0
        self._spawn(config.initial_food)

    @property
    def done(self):
        return self.tick >= self.config.horizon or not self.alive.any()

    def _spawn(self, count):
        free = ~self.food.copy()
        free[tuple(self.positions[self.alive].T)] = False
        candidates = np.flatnonzero(free)
        if count and len(candidates):
            self.food.flat[self.rng.choice(candidates, min(count, len(candidates)), replace=False)] = True

    def observe(self):
        c = self.config
        width = 2*c.radius+1
        obs = np.zeros((c.agents, 5, width, width), dtype=np.float32)
        occupied = {tuple(p) for p in self.positions[self.alive]}
        for i in np.flatnonzero(self.alive):
            y, x = self.positions[i]
            for oy, dy in enumerate(range(-c.radius, c.radius+1)):
                for ox, dx in enumerate(range(-c.radius, c.radius+1)):
                    q = (y+dy, x+dx)
                    if 0 <= q[0] < c.size and 0 <= q[1] < c.size:
                        obs[i, 0, oy, ox] = self.food[q]
                        obs[i, 2, oy, ox] = q in occupied and q != (y, x)
                        obs[i, 3, oy, ox] = 1
            obs[i, 1, c.radius, c.radius] = 1
            obs[i, 4, :, :] = self.hunger[i]/c.max_hunger
        return obs

    def step(self, actions):
        if self.done:
            raise RuntimeError('Episode is finished')
        a = np.asarray(actions)
        if a.shape != (self.config.agents,) or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a >= 5)):
            raise ValueError('One integer move/wait action (0..4) per agent required')
        old = self.positions.copy()
        occupied = {tuple(p) for p in old[self.alive]}
        candidates = {}
        for i in np.flatnonzero(self.alive):
            q = tuple(old[i] + MOVES[a[i]])
            if a[i] and all(0 <= v < self.config.size for v in q) and q not in occupied:
                candidates.setdefault(q, []).append(i)
        for q, ids in candidates.items():
            if len(ids) == 1:
                self.positions[ids[0]] = q
        self.hunger[self.alive] -= 1
        ate = np.zeros(self.config.agents, dtype=bool)
        for i in np.flatnonzero(self.alive):
            q = tuple(self.positions[i])
            if self.food[q]:
                self.food[q] = False
                self.hunger[i] = self.config.max_hunger
                ate[i] = True
        died = self.alive & (self.hunger <= 0)
        self.alive[died] = False
        self.tick += 1
        self._spawn(self.config.food_per_step)
        return self.observe(), self.alive.astype(np.float32), self.done, {
            'ate': ate, 'died': died, 'truncated': self.tick >= self.config.horizon,
        }

    def state(self):
        return copy.deepcopy({k: getattr(self, k) for k in ['positions', 'alive', 'hunger', 'food', 'tick']} | {'rng_state': self.rng.bit_generator.state})

    def load_state(self, state):
        for key in ['positions', 'alive', 'hunger', 'food', 'tick']:
            setattr(self, key, copy.deepcopy(state[key]))
        self.rng.bit_generator.state = copy.deepcopy(state['rng_state'])

    def frame(self):
        return {'tick': self.tick, 'positions': self.positions.tolist(), 'alive': self.alive.tolist(),
                'hunger': self.hunger.tolist(), 'food': np.argwhere(self.food).tolist()}

def random_policy(observations, rng):
    return rng.integers(0, 5, len(observations))

def greedy_policy(observations, rng):
    """Only sees the same local observation as a learned policy; no world access."""
    actions = np.zeros(len(observations), dtype=np.int64)
    for i, obs in enumerate(observations):
        r = obs.shape[-1]//2
        if not obs[1, r, r]:
            continue
        legal = [a for a, (dy, dx) in enumerate(MOVES)
                 if obs[3, r+dy, r+dx] and not obs[2, r+dy, r+dx]] if r else [0]
        food = np.argwhere(obs[0])
        if len(food):
            scores = [(min(np.abs(food - (np.array([r, r])+MOVES[a])).sum(axis=1)), a) for a in legal]
            actions[i] = min(scores)[1]
        else:
            actions[i] = rng.choice(legal)
    return actions
