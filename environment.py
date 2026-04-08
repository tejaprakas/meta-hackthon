import numpy as np
import random

class AlgoBrainEnv:
    def __init__(self):
        self.tasks = ["easy", "medium", "hard"]
        self.current_task = "easy"
        self.step_count = 0
        self.max_steps = 10

    def reset(self):
        self.current_task = random.choice(self.tasks)
        self.step_count = 0

        self.state = {
            "cpu_temp": [random.uniform(50, 70)],
            "battery_maH": [random.uniform(20, 80)],
            "accuracy_score": [random.uniform(0.7, 0.9)]
        }

        return self.state, {}

    def step(self, action):
        self.step_count += 1

        # Simulate changes
        if action == 1:  # RUN
            self.state["cpu_temp"][0] += random.uniform(2, 5)
            self.state["battery_maH"][0] -= random.uniform(5, 10)
            self.state["accuracy_score"][0] += 0.01
        else:  # IDLE
            self.state["cpu_temp"][0] -= random.uniform(1, 3)
            self.state["battery_maH"][0] += random.uniform(1, 3)

        # Reward logic
        reward = 0.0

        # Task-based scaling
        if self.current_task == "easy":
            reward += 1.0
        elif self.current_task == "medium":
            reward += 0.5
        else:
            reward += 0.2

        # Safety penalty
        if self.state["cpu_temp"][0] > 95:
            reward -= 1.0
            done = True
        else:
            done = False

        # End condition
        if self.step_count >= self.max_steps:
            done = True

        return self.state, float(reward), done, False, {}
