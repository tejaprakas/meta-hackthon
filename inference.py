"""
AlgoBrain Inference Script - LLM Agent Interaction
Uses OpenAI API to run agents against the code review environment
"""

import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

from environment import AlgoBrainEnv, Action, Observation

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MAX_STEPS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 500

def log_start(task: str, env: str, model: str) -> None:
    """Log start of evaluation"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, 
    action: str, 
    reward: float, 
    done: bool, 
    error: Optional[str] = None
) -> None:
    """Log each step"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log end of evaluation"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_system_prompt() -> str:
    """Build system prompt for the LLM agent"""
    return textwrap.dedent("""
    You are an expert code reviewer and algorithm specialist. 
    Your task is to review algorithm implementations and suggest improvements.
    
    For each code snippet:
    1. Analyze correctness (does it solve the problem?)
    2. Check efficiency (time/space complexity)
    3. Improve clarity and readability
    
    Respond with ONLY a JSON object (no markdown, no code blocks):
    {
        "suggested_code": "...",
        "reasoning": "..."
    }
    """).strip()


def build_user_prompt(obs: Observation, previous_feedback: str = "") -> str:
    """Build user prompt with current state"""
    prev_section = f"Previous feedback: {previous_feedback}\n" if previous_feedback else ""
    
    return textwrap.dedent(f"""
    Current Task: {obs.task_name}
    Step: {obs.step} / {obs.max_steps}
    
    Current Code:
    ```python
    {obs.code}
    ```
    
    Feedback: {obs.feedback}
    {prev_section}
    
    Please suggest improvements to the code. Respond with JSON only.
    """).strip()


def get_model_response(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str
) -> Optional[Action]:
    """Query the LLM model"""
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        content = response.content[0].text.strip()
        
        # Parse JSON response
        try:
            # Try to extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            action = Action(
                suggested_code=data.get("suggested_code", ""),
                reasoning=data.get("reasoning", "")
            )
            return action
        except json.JSONDecodeError:
            return None
            
    except Exception as e:
        print(f"Error calling model: {e}", flush=True)
        return None


def run_task(task_name: str) -> tuple:
    """Run a single task evaluation"""
    env = AlgoBrainEnv()
    
    # Initialize client
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL
    )
    
    obs, info = env.reset()
    if task_name in env.TASKS:
        # Override task
        env.current_task = task_name
        task = env.TASKS[task_name]
        env.current_code = task["initial_code"]
        obs = Observation(
            code=env.current_code,
            feedback=f"Task: {task['description']}. Current code is incomplete.",
            task_name=task["name"],
            step=0,
            max_steps=MAX_STEPS
        )
    
    log_start(obs.task_name, "algobrain", MODEL_NAME)
    
    system_prompt = build_system_prompt()
    rewards = []
    success = False
    previous_feedback = ""
    
    for step in range(1, MAX_STEPS + 1):
        # Build and send prompt
        user_prompt = build_user_prompt(obs, previous_feedback)
        action = get_model_response(client, system_prompt, user_prompt)
        
        if action is None:
            error_msg = "Failed to parse model response"
            log_step(step, "error", 0.0, True, error_msg)
            break
        
        # Execute action in environment
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        previous_feedback = obs.feedback
        
        # Log step
        action_str = action.suggested_code[:50].replace("\n", " ")
        log_step(step, action_str, reward, done)
        
        if done:
            success = reward > 0.7
            break
    
    # Calculate final score
    score = sum(rewards) / len(rewards) if rewards else 0.0
    success = score > 0.6 or (len(rewards) > 0 and rewards[-1] > 0.8)
    
    log_end(success, len(rewards), score, rewards)
    
    return success, score, rewards


def main():
    """Run evaluation on all tasks"""
    if not HF_TOKEN:
        print("No HF_TOKEN provided. Set HF_TOKEN environment variable.", flush=True)
        return
    
    tasks = ["easy", "medium", "hard"]
    results = {}
    
    for task in tasks:
        try:
            success, score, rewards = run_task(task)
            results[task] = {
                "success": success,
                "score": score,
                "reward_count": len(rewards)
            }
        except Exception as e:
            print(f"Error running task {task}: {e}", flush=True)
            results[task] = {"success": False, "score": 0.0}
    
    print("\n" + "="*50, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("="*50, flush=True)
    for task, result in results.items():
        print(f"{task}: success={result['success']}, score={result['score']:.3f}", flush=True)
    print("="*50, flush=True)


if __name__ == "__main__":
    main()
