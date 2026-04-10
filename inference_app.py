"""
AlgoBrain - HuggingFace Spaces Gradio Interface
Minimal, robust interface focused on core functionality
"""

import gradio as gr
import sys
import traceback

def get_environment_info():
    """Load environment information safely"""
    try:
        return """
## 🧠 AlgoBrain - Reinforcement Learning Environment

**Status:** ✅ Ready

### Environment Overview
- **Name**: algo-brain-env
- **Type**: RL Environment (Gym-like API)
- **Algorithm**: REINFORCE (Policy Gradient)

### Tasks
1. **Easy**: Accuracy-focused (Reward: 1.0)
2. **Medium**: Balanced approach (Reward: 0.5)
3. **Hard**: Efficiency-focused (Reward: 0.2)

### State Space (3D)
- CPU Temperature: 50-95°C
- Battery Level: 20-80 mAh
- Accuracy Score: 0.7-1.0

### Action Space (Discrete)
- **Action 0**: IDLE (cool down, preserve battery)
- **Action 1**: RUN (increase accuracy, use resources)

### Reward Logic
- Base reward scaled by task difficulty
- -1.0 penalty if CPU > 95°C
- Episode terminates after 10 steps or safety violation

### API Endpoints
- `GET /` - Health check
- `POST /reset` - Reset environment
- `POST /step` - Execute action
- `GET /state` - Get current state
- `GET /tasks` - List available tasks
- `GET /info` - Environment info
"""
    except Exception as e:
        return f"Error loading info: {str(e)}"

def run_inference():
    """Run inference pipeline"""
    try:
        # Import locally to catch any issues
        from environment import AlgoBrainEnv
        from inference import PolicyNet, get_obs_tensor, train
        
        # Initialize
        env = AlgoBrainEnv()
        policy = PolicyNet()
        
        # Quick training
        train(policy, episodes=5)
        
        # Run inference
        obs, _ = env.reset()
        total_reward = 0
        output_log = "[START] AlgoBrain Inference\n"
        
        for step in range(10):
            obs_tensor = get_obs_tensor(obs)
            probs = policy(obs_tensor)
            action = probs.argmax().item()
            
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            action_name = "RUN" if action == 1 else "IDLE"
            output_log += f"[STEP] step={step} action={action_name} reward={reward:.2f} temp={obs['cpu_temp'][0]:.2f}\n"
            
            if done:
                break
        
        output_log += f"[END] total_reward={total_reward:.2f}\n"
        return output_log
        
    except Exception as e:
        return f"**Error**: {str(e)}\n\n**Traceback**:\n{traceback.format_exc()}"

# Build minimal Gradio interface
with gr.Blocks(title="AlgoBrain") as demo:
    gr.Markdown("# 🧠 AlgoBrain RL Environment")
    
    with gr.Tabs():
        with gr.TabItem("🏃 Run"):
            run_btn = gr.Button("Run Inference", variant="primary", size="lg")
            output = gr.Textbox(label="Output", lines=15, interactive=False)
            run_btn.click(fn=run_inference, outputs=output)
        
        with gr.TabItem("📋 Info"):
            info_display = gr.Markdown(value=get_environment_info())

if __name__ == "__main__":
    demo.launch()
