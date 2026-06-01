import unittest


class TaskRegistryTests(unittest.TestCase):
    def test_builtin_tasks_are_registered(self):
        from nesylink.tasks import get_task, list_tasks

        task_ids = [task.task_id for task in list_tasks()]
        self.assertIn("task_1", task_ids)
        self.assertIn("task_2", task_ids)
        self.assertIn("task_3", task_ids)

        task = get_task("task_1")
        self.assertEqual(task.map_id, "task_1")
        self.assertEqual(task.reward_id, "collect_key")
        self.assertEqual(task.gym_id, "task_1")

    def test_unknown_task_id_has_clear_error(self):
        from nesylink.tasks import get_task

        with self.assertRaisesRegex(ValueError, "unknown task_id 'missing'"):
            get_task("missing")


if __name__ == "__main__":
    unittest.main()
