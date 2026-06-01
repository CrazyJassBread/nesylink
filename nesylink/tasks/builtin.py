from __future__ import annotations

from .registry import register_task
from .specs import TaskSpec


BUILTIN_TASKS = (
    TaskSpec(
        task_id="task_1",
        gym_id="task_1",
        map_id="task_1",
        reward_id="collect_key",
        max_steps=500,
        mission="Collect the key and reach the exit.",
    ),
    TaskSpec(
        task_id="task_2",
        gym_id="task_2",
        map_id="task_2",
        reward_id="kill_monster",
        max_steps=500,
        mission="Defeat the monster, collect the key, and reach the exit.",
    ),
    TaskSpec(
        task_id="task_3",
        gym_id="task_3",
        map_id="task_3",
        reward_id="collect_key",
        max_steps=500,
        mission="Travel west through the chaser room, collect the key, return, and unlock the right door.",
    ),
)


def register_builtin_tasks() -> None:
    for task in BUILTIN_TASKS:
        try:
            register_task(task)
        except ValueError as exc:
            if "duplicate" not in str(exc):
                raise
