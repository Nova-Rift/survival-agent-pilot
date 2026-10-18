import unittest
import numpy as np
from simulator import Config, SurvivalWorld, greedy_policy, random_policy

class WorldTests(unittest.TestCase):
    def world(self, **kw):
        return SurvivalWorld(1, Config(size=4, agents=2, initial_food=0, food_per_step=0, **kw))

    def test_seeded_rollout_and_state_resume(self):
        a, b = SurvivalWorld(7), SurvivalWorld(7)
        rng = np.random.default_rng(12)
        for _ in range(10):
            actions = random_policy(a.observe(), rng)
            a.step(actions); b.step(actions)
            self.assertEqual(a.frame(), b.frame())
        b = SurvivalWorld(999); b.load_state(a.state())
        for _ in range(5):
            actions = random_policy(a.observe(), rng)
            a.step(actions); b.step(actions)
            self.assertEqual(a.frame(), b.frame())

    def test_collisions_reject_both_and_swaps(self):
        e = self.world(); e.positions[:] = [[1,0],[1,2]]
        e.step([2,4]); np.testing.assert_array_equal(e.positions, [[1,0],[1,2]])
        e.positions[:] = [[1,0],[1,1]]
        e.step([2,4]); np.testing.assert_array_equal(e.positions, [[1,0],[1,1]])

    def test_wall_wait_and_occupied_destination(self):
        e = self.world(); e.positions[:] = [[0,0],[0,1]]
        e.step([1,0]); np.testing.assert_array_equal(e.positions, [[0,0],[0,1]])
        e.step([2,2]); np.testing.assert_array_equal(e.positions, [[0,0],[0,2]])

    def test_last_moment_food_rescues_and_death_absorbs(self):
        e = self.world(max_hunger=2); e.positions[:] = [[0,0],[3,3]]
        e.hunger[:] = 1; e.food[0,1] = True
        _, reward, _, info = e.step([2,0])
        np.testing.assert_array_equal(reward, [1,0]); self.assertTrue(info['ate'][0])
        self.assertEqual(e.hunger[0],2); self.assertFalse(e.alive[1])
        dead_position = e.positions[1].copy(); e.step([0,1])
        np.testing.assert_array_equal(e.positions[1],dead_position)
        self.assertTrue(np.all(e.observe()[1] == 0))

    def test_food_competition_and_spawn_bounds(self):
        e = self.world(); e.positions[:] = [[1,0],[1,2]]; e.food[1,1] = True
        _, _, _, info = e.step([2,4]); self.assertFalse(info['ate'].any()); self.assertTrue(e.food[1,1])
        e = SurvivalWorld(4, Config(size=4,agents=8,initial_food=100,food_per_step=100))
        self.assertEqual(e.food.sum(),8)
        self.assertFalse(e.food[tuple(e.positions.T)].any())

    def test_local_observation_has_no_remote_information(self):
        e = SurvivalWorld(1,Config(size=8,agents=2,radius=1,initial_food=0,food_per_step=0))
        e.positions[:] = [[1,1],[7,7]]; before=e.observe()[0].copy()
        e.food[6,6] = True; e.positions[1] = [6,7]
        np.testing.assert_array_equal(before,e.observe()[0])
        e.food[1,2] = True
        self.assertEqual(greedy_policy(e.observe(), np.random.default_rng(0))[0],2)

    def test_horizon_is_truncation_and_invalid_actions_rejected(self):
        e=self.world(horizon=1); _,_,done,info=e.step([0,0])
        self.assertTrue(done); self.assertTrue(info['truncated']); self.assertTrue(e.alive.all())
        with self.assertRaises(RuntimeError):e.step([0,0])
        for a in [[5,0],[1.2,0],[0]]:
            with self.assertRaises(ValueError):self.world().step(a)

    def test_population_invariants_over_seeded_episodes(self):
        for seed in range(4):
            e=SurvivalWorld(seed); rng=np.random.default_rng(seed)
            while not e.done:
                e.step(random_policy(e.observe(),rng))
                live=e.positions[e.alive]
                self.assertEqual(len(set(map(tuple,live))),len(live))
                self.assertTrue(np.all((live>=0)&(live<e.config.size)))
                self.assertTrue(np.all(e.hunger[e.alive]>0))
                self.assertFalse(e.food[tuple(live.T)].any())

if __name__ == '__main__': unittest.main()
