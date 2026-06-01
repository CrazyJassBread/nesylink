import unittest


class EnvEntrypointTests(unittest.TestCase):
    def test_make_env_accepts_task_id(self):
        from nesylink.env import make_env

        env = make_env(task_id="task_1")
        try:
            obs, info = env.reset(seed=0)
            self.assertIn("grid", obs)
            self.assertEqual(info["env"]["map_id"], "task_1")
            self.assertEqual(env.spec.id, "task_1")
            self.assertEqual(env.unwrapped.mission, "Collect the key and reach the exit.")
        finally:
            env.close()

    def test_explicit_arguments_override_task_defaults(self):
        from nesylink.env import make_env

        env = make_env(task_id="task_1", max_steps=1)
        try:
            env.reset(seed=0)
            _, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_gymnasium_make_can_create_registered_task(self):
        import gymnasium as gym
        import nesylink

        nesylink.register_gym_envs()
        env = gym.make("task_1")
        try:
            obs, info = env.reset(seed=0)
            self.assertIn("grid", obs)
            self.assertEqual(info["env"]["map_id"], "task_1")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
