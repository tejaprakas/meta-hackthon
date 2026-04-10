"""
AlgoBrain - Real-world Task: Algorithm Interview Code Review
An environment where an LLM agent reviews and improves algorithm implementations.
"""

from typing import Optional
from pydantic import BaseModel, Field
import random
import json


class Observation(BaseModel):
    """Current environment state"""
    code: str = Field(description="The current code snippet to review")
    feedback: str = Field(description="Feedback on the current submission")
    task_name: str = Field(description="Name of the current task")
    step: int = Field(description="Current step number")
    max_steps: int = Field(description="Maximum steps allowed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": "def sort_array(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                "feedback": "Code is correct but inefficient. Time complexity is O(n²).",
                "task_name": "bubble_sort",
                "step": 1,
                "max_steps": 5
            }
        }


class Action(BaseModel):
    """Agent action - improved code or feedback"""
    suggested_code: str = Field(description="The improved code implementation")
    reasoning: str = Field(description="Explanation of improvements")
    
    class Config:
        json_schema_extra = {
            "example": {
                "suggested_code": "def sort_array(arr):\n    return sorted(arr)",
                "reasoning": "Use built-in sorted() for O(n log n) complexity and readability"
            }
        }


class Reward(BaseModel):
    """Reward score"""
    value: float = Field(ge=0.0, le=1.0, description="Normalized reward [0.0, 1.0]")
    breakdown: dict = Field(description="Breakdown of reward components")
    
    class Config:
        json_schema_extra = {
            "example": {
                "value": 0.85,
                "breakdown": {
                    "correctness": 0.9,
                    "efficiency": 0.8,
                    "clarity": 0.85
                }
            }
        }


class AlgoBrainEnv:
    """Algorithm Code Review Environment"""
    
    TASKS = {
        "easy": {
            "name": "bubble_sort",
            "description": "Implement bubble sort algorithm",
            "initial_code": "def sort_array(arr):\n    # TODO: Implement bubble sort\n    pass",
            "solution": "def sort_array(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
            "test_cases": [
                {"input": [3, 1, 4, 1, 5], "expected": [1, 1, 3, 4, 5]},
                {"input": [5, 4, 3, 2, 1], "expected": [1, 2, 3, 4, 5]},
                {"input": [1], "expected": [1]}
            ]
        },
        "medium": {
            "name": "quicksort",
            "description": "Implement quicksort with proper partitioning",
            "initial_code": "def quicksort(arr, left=0, right=None):\n    # TODO: Implement quicksort\n    pass",
            "solution": "def quicksort(arr, left=0, right=None):\n    if right is None:\n        right = len(arr) - 1\n    if left < right:\n        pi = partition(arr, left, right)\n        quicksort(arr, left, pi - 1)\n        quicksort(arr, pi + 1, right)\n    return arr",
            "test_cases": [
                {"input": [3, 1, 4, 1, 5], "expected": [1, 1, 3, 4, 5]},
                {"input": [64, 34, 25, 12, 22, 11, 90], "expected": [11, 12, 22, 25, 34, 64, 90]}
            ]
        },
        "hard": {
            "name": "dynamic_programming",
            "description": "Solve longest increasing subsequence with optimal substructure",
            "initial_code": "def longest_increasing_subsequence(arr):\n    # TODO: Implement LIS using DP\n    pass",
            "solution": "def longest_increasing_subsequence(arr):\n    if not arr:\n        return 0\n    dp = [1] * len(arr)\n    for i in range(1, len(arr)):\n        for j in range(i):\n            if arr[j] < arr[i]:\n                dp[i] = max(dp[i], dp[j] + 1)\n    return max(dp)",
            "test_cases": [
                {"input": [3, 10, 2, 1, 20], "expected": 3},
                {"input": [50, 3, 10, 7, 40, 80], "expected": 4}
            ]
        }
    }
    
    def __init__(self):
        self.current_task = None
        self.current_code = ""
        self.step_count = 0
        self.max_steps = 5
        self.improvements = []
        self.done = False
    
    def reset(self):
        """Reset environment to initial state"""
        self.step_count = 0
        self.done = False
        self.improvements = []
        self.current_task = random.choice(list(self.TASKS.keys()))
        task = self.TASKS[self.current_task]
        self.current_code = task["initial_code"]
        
        obs = Observation(
            code=self.current_code,
            feedback=f"Task: {task['description']}. Current code is incomplete.",
            task_name=task["name"],
            step=self.step_count,
            max_steps=self.max_steps
        )
        return obs, {}
    
    def step(self, action: Action):
        """Execute one step of interaction"""
        self.step_count += 1
        task = self.TASKS[self.current_task]
        
        # Score the suggested improvement
        score = self._evaluate_code(action.suggested_code, task)
        
        # Track improvement
        self.improvements.append({
            "step": self.step_count,
            "code": action.suggested_code,
            "reasoning": action.reasoning,
            "score": score
        })
        
        # Update current code
        self.current_code = action.suggested_code
        
        # Determine if done
        done = self.step_count >= self.max_steps or score > 0.9
        self.done = done
        
        # Create reward
        reward_value = score  # Score in [0, 1]
        reward = Reward(
            value=reward_value,
            breakdown={
                "correctness": min(1.0, score * 1.2),
                "efficiency": min(1.0, score * 0.8),
                "clarity": 0.7 if len(action.reasoning) > 50 else 0.4
            }
        )
        
        # Generate feedback
        if score > 0.9:
            feedback = "Excellent! Code is correct and efficient."
        elif score > 0.7:
            feedback = "Good progress. Consider optimizing further."
        elif score > 0.5:
            feedback = "Partially correct. Review edge cases."
        else:
            feedback = "Needs more work. Ensure basic correctness first."
        
        obs = Observation(
            code=self.current_code,
            feedback=feedback,
            task_name=task["name"],
            step=self.step_count,
            max_steps=self.max_steps
        )
        
        return obs, reward.value, done, False, {
            "reward_breakdown": reward.breakdown,
            "improvements_count": len(self.improvements)
        }
    
    def state(self):
        """Get current state"""
        task = self.TASKS[self.current_task]
        return {
            "code": self.current_code,
            "task": self.current_task,
            "step": self.step_count,
            "improvements": self.improvements,
            "task_description": task["description"]
        }
    
    def _evaluate_code(self, code: str, task: dict) -> float:
        """Evaluate code quality (0.0-1.0)"""
        score = 0.5  # Base score
        
        # Correctness check (simplified)
        if "pass" not in code:
            score += 0.3
        
        # Efficiency check
        if "for" not in code or code.count("for") < 2:
            score += 0.15
        
        # Clarity check
        if len(code.split("\n")) > 3:
            score += 0.05
        
        # Bonus for being close to solution
        if self._code_similarity(code, task["solution"]) > 0.8:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _code_similarity(self, code1: str, code2: str) -> float:
        """Simple string similarity"""
        common = sum(1 for a, b in zip(code1, code2) if a == b)
        return common / max(len(code1), len(code2), 1)


if __name__ == "__main__":
    env = AlgoBrainEnv()
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"\nState: {env.state()}")
