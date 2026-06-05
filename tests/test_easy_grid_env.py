import unittest

import numpy as np


class EasyGridEnvTests(unittest.TestCase):
    def test_easy_task_registration_and_grid_observation_contract(self):
        from nesylink.env import make_env

        env = make_env(task_id="task_1_easy")
        try:
            obs, info = env.reset(seed=0)
            self.assertEqual(env.unwrapped.control_mode, "grid")
            self.assertEqual(env.unwrapped.observation_mode, "grid")
            self.assertNotIn("player_position_px", obs)
            self.assertNotIn("monsters_position_px", obs)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertEqual(obs["grid"].shape, (8, 10))
            self.assertEqual(obs["inventory_ids"].shape, (2,))
            self.assertNotIn("position_px", info["agent"])
            self.assertEqual(info["control"]["control_mode"], "grid")
            self.assertEqual(info["control"]["observation_mode"], "grid")
            self.assertEqual(info["control"]["tile_size"], 16)
        finally:
            env.close()

    def test_gymnasium_can_create_easy_task(self):
        import gymnasium as gym
        import nesylink

        nesylink.register_gym_envs()
        env = gym.make("task_1_easy")
        try:
            obs, info = env.reset(seed=0)
            self.assertIn("grid", obs)
            self.assertEqual(info["env"]["map_id"], "task_1")
            self.assertEqual(info["control"]["control_mode"], "grid")
        finally:
            env.close()

    def test_existing_task_keeps_full_pixel_observation(self):
        from nesylink.env import make_env

        env = make_env(task_id="task_1")
        try:
            obs, info = env.reset(seed=0)
            self.assertIn("player_position_px", obs)
            self.assertIn("monsters_position_px", obs)
            self.assertIn("position_px", info["agent"])
            self.assertIn("movement_pixels", info["control"])
        finally:
            env.close()

    def test_grid_move_advances_exactly_one_tile(self):
        from nesylink.core.constants import ACTION_RIGHT
        from nesylink.env import make_env

        env = make_env(task_id="task_1_easy")
        try:
            obs, _ = env.reset(seed=0)
            start = obs["player_tile"].copy()
            obs, _, _, _, _ = env.step(ACTION_RIGHT)
            expected = start + np.asarray([1, 0], dtype=np.int32)
            np.testing.assert_array_equal(obs["player_tile"], expected)
        finally:
            env.close()

    def test_blocked_grid_move_leaves_tile_unchanged(self):
        from nesylink.core.constants import ACTION_LEFT
        from nesylink.env import make_env

        env = make_env(task_id="task_1_easy")
        try:
            env.reset(seed=0)
            env.unwrapped.player.position_px = (0.0, 3.0 * 16.0)
            before = np.asarray(env.unwrapped._player_tile(), dtype=np.int32)
            obs, _, _, _, info = env.step(ACTION_LEFT)
            np.testing.assert_array_equal(obs["player_tile"], before)
            self.assertTrue(info["events"]["flags"]["action_blocked"])
        finally:
            env.close()

    def test_chaser_moves_one_tile_toward_player_and_info_matches_obs(self):
        from nesylink.core.constants import ACTION_NOOP
        from nesylink.env import make_env

        env = make_env(task_id="task_2_easy")
        try:
            obs, _ = env.reset(seed=0)
            before = obs["monsters_tile"][0].copy()
            obs, _, _, _, info = env.step(ACTION_NOOP)
            after = obs["monsters_tile"][0]
            self.assertEqual(int(np.abs(after - before).sum()), 1)
            before_distance = int(np.abs(before - obs["player_tile"]).sum())
            after_distance = int(np.abs(after - obs["player_tile"]).sum())
            self.assertLess(after_distance, before_distance)
            self.assertEqual(info["entities"]["monsters_tile"][0], after.tolist())
        finally:
            env.close()

    def test_patroller_period_controls_grid_movement(self):
        from nesylink.core.constants import ACTION_NOOP
        from nesylink.core.monsters import MonsterState
        from nesylink.core.state import tile_to_top_left_px
        from nesylink.env import make_env

        env = make_env(
            task_id="task_2_easy",
            max_monsters=1,
            monster_move_periods={"patroller": 2},
        )
        try:
            obs, _ = env.reset(seed=0)
            room = env.unwrapped.room
            room.monsters.clear()
            monster = MonsterState(
                monster_id="patroller",
                monster_type="patroller",
                position_px=tile_to_top_left_px((2, 2)),
            )
            monster.patrol_points_px = [
                tile_to_top_left_px((2, 2)),
                tile_to_top_left_px((4, 2)),
            ]
            room.monsters["patroller"] = monster
            obs = env.unwrapped._get_obs()
            start = obs["monsters_tile"][0].copy()
            obs, _, _, _, _ = env.step(ACTION_NOOP)
            np.testing.assert_array_equal(obs["monsters_tile"][0], start)
            obs, _, _, _, _ = env.step(ACTION_NOOP)
            self.assertNotEqual(obs["monsters_tile"][0].tolist(), start.tolist())
        finally:
            env.close()

    def test_monsters_do_not_move_into_occupied_tiles(self):
        from nesylink.core.constants import ACTION_NOOP
        from nesylink.core.monsters import MonsterState
        from nesylink.core.state import tile_to_top_left_px
        from nesylink.env import make_env

        env = make_env(task_id="task_2_easy", max_monsters=2)
        try:
            env.reset(seed=0)
            room = env.unwrapped.room
            room.monsters.clear()
            env.unwrapped.player.position_px = tile_to_top_left_px((4, 2))
            room.monsters["m1"] = MonsterState(
                monster_id="m1",
                monster_type="chaser",
                position_px=tile_to_top_left_px((2, 2)),
            )
            room.monsters["m2"] = MonsterState(
                monster_id="m2",
                monster_type="chaser",
                position_px=tile_to_top_left_px((3, 2)),
            )
            obs, _, _, _, _ = env.step(ACTION_NOOP)
            active_tiles = [
                tuple(tile)
                for tile, active in zip(obs["monsters_tile"], obs["monsters_active_mask"])
                if bool(active)
            ]
            self.assertEqual(len(active_tiles), len(set(active_tiles)))
        finally:
            env.close()

    def test_easy_grid_seed_determinism_and_reward_without_pixel_obs(self):
        from nesylink.core.constants import ACTION_NOOP
        from nesylink.env import make_env

        first = make_env(task_id="task_2_easy")
        second = make_env(task_id="task_2_easy")
        try:
            first_obs, _ = first.reset(seed=123)
            second_obs, _ = second.reset(seed=123)
            np.testing.assert_array_equal(first_obs["grid"], second_obs["grid"])
            first_obs, first_reward, _, _, first_info = first.step(ACTION_NOOP)
            second_obs, second_reward, _, _, second_info = second.step(ACTION_NOOP)
            np.testing.assert_array_equal(first_obs["monsters_tile"], second_obs["monsters_tile"])
            self.assertEqual(first_reward, second_reward)
            self.assertNotIn("player_position_px", first_obs)
            self.assertIn("reward", first_info)
            self.assertIn("reward", second_info)
        finally:
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main()
