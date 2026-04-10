"""
Test script for AlgoBrain v2 Environment
Verifies the environment works without needing OpenAI API
"""

from environment_v2 import AlgoBrainEnv, Action, Observation

def test_environment():
    """Test basic environment functionality"""
    print("Testing AlgoBrain v2 Environment...")
    print("=" * 60)
    
    env = AlgoBrainEnv()
    
    # Test reset
    print("\n1. Testing reset...")
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Task: {obs.task_name}")
    print(f"  Initial feedback: {obs.feedback[:50]}...")
    
    # Test observation is proper type
    print("\n2. Testing observation format...")
    assert isinstance(obs, Observation), "Observation should be Observation type"
    assert hasattr(obs, 'code'), "Observation should have 'code'"
    assert hasattr(obs, 'feedback'), "Observation should have 'feedback'"
    assert hasattr(obs, 'task_name'), "Observation should have 'task_name'"
    print("✓ Observation structure is correct")
    
    # Test action and step
    print("\n3. Testing step with action...")
    action = Action(
        suggested_code="def sort_array(arr):\n    return sorted(arr)",
        reasoning="Use built-in sorted() for O(n log n) complexity"
    )
    obs, reward, done, truncated, info = env.step(action)
    print(f"✓ Step executed successfully")
    print(f"  Reward: {reward:.2f}")
    print(f"  Done: {done}")
    print(f"  Feedback: {obs.feedback}")
    
    # Test reward range
    print("\n4. Testing reward range...")
    assert 0.0 <= reward <= 1.0, f"Reward should be in [0, 1], got {reward}"
    print(f"✓ Reward in valid range: {reward:.2f}")
    
    # Test state method
    print("\n5. Testing state method...")
    state = env.state()
    assert "code" in state, "State should contain 'code'"
    assert "task" in state, "State should contain 'task'"
    assert "improvements" in state, "State should contain 'improvements'"
    print("✓ State structure is correct")
    print(f"  Current task: {state['task']}")
    print(f"  Improvements recorded: {len(state['improvements'])}")
    
    # Test multiple steps
    print("\n6. Testing multiple steps...")
    for step_num in range(2, 5):
        action = Action(
            suggested_code=f"# Step {step_num} improvement\n{action.suggested_code}",
            reasoning=f"Improvement step {step_num}"
        )
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {step_num}: reward={reward:.2f}, done={done}")
        if done:
            print(f"  Episode ended at step {step_num}")
            break
    
    # Test all tasks
    print("\n7. Testing all task types...")
    for task_name in ["easy", "medium", "hard"]:
        env = AlgoBrainEnv()
        env.current_task = task_name
        obs, _ = env.reset()
        print(f"  ✓ {task_name}: {obs.task_name}")
    
    print("\n" + "=" * 60)
    print("✅ All environment tests passed!")
    print("=" * 60)
    
    # Print task information
    print("\nAvailable Tasks:")
    for task_name, task_info in env.TASKS.items():
        print(f"\n{task_name.upper()}:")
        print(f"  Name: {task_info['name']}")
        print(f"  Description: {task_info['description']}")
        print(f"  Test cases: {len(task_info['test_cases'])}")


if __name__ == "__main__":
    test_environment()
