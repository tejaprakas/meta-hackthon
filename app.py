import gradio as gr
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from environment import AlgoBrainEnv
from inference import PolicyNet, get_obs_tensor, train as train_policy
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Global state
current_session = None
training_history = {"episodes": [], "rewards": [], "cpu_temps": []}
model_state = {"policy": None, "trained": False}

def initialize_session():
    """Initialize a new training session"""
    global current_session, training_history
    current_session = {
        "env": AlgoBrainEnv(),
        "policy": PolicyNet(),
        "episode": 0,
        "total_steps": 0,
        "session_start": datetime.now()
    }
    training_history = {"episodes": [], "rewards": [], "cpu_temps": []}
    return "✅ Session initialized successfully"

def train_model(episodes: int = 30, learning_rate: float = 0.01) -> dict:
    """Train the policy network"""
    global model_state, training_history, current_session
    
    if current_session is None:
        initialize_session()
    
    policy = current_session["policy"]
    env = current_session["env"]
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    episode_rewards = []
    episode_temps = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        temps = []
        done = False
        
        while not done:
            obs_tensor = get_obs_tensor(obs)
            probs = policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            
            obs, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
            temps.append(obs['cpu_temp'][0])
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_reward = sum(rewards)
        avg_temp = np.mean(temps)
        
        episode_rewards.append(episode_reward)
        episode_temps.append(avg_temp)
        training_history["episodes"].append(ep + 1)
        training_history["rewards"].append(episode_reward)
        training_history["cpu_temps"].append(avg_temp)
        current_session["episode"] = ep + 1
    
    model_state["policy"] = policy
    model_state["trained"] = True
    
    return {
        "status": f"✅ Training completed! {episodes} episodes trained",
        "final_reward": f"{episode_rewards[-1]:.2f}",
        "avg_reward": f"{np.mean(episode_rewards):.2f}",
        "final_temp": f"{episode_temps[-1]:.2f}°C",
        "avg_temp": f"{np.mean(episode_temps):.2f}°C"
    }

def run_inference(task: str = "easy", steps: int = 10) -> dict:
    """Run inference on the trained model"""
    global current_session, model_state
    
    if model_state["policy"] is None:
        return {"error": "⚠️ Please train the model first"}
    
    policy = model_state["policy"]
    env = AlgoBrainEnv()
    env.current_task = task
    
    obs, _ = env.reset()
    obs['task'] = task
    
    total_reward = 0
    episode_data = []
    
    for step in range(steps):
        obs_tensor = get_obs_tensor(obs)
        probs = policy(obs_tensor)
        action = torch.argmax(probs).item()
        action_name = "RUN" if action == 1 else "IDLE"
        
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        episode_data.append({
            "step": step,
            "action": action_name,
            "reward": f"{reward:.2f}",
            "cpu_temp": f"{obs['cpu_temp'][0]:.2f}°C",
            "battery": f"{obs['battery_maH'][0]:.2f}mAh",
            "accuracy": f"{obs['accuracy_score'][0]:.4f}"
        })
        
        if done:
            break
    
    return {
        "task": task,
        "total_reward": f"{total_reward:.2f}",
        "steps_completed": len(episode_data),
        "episode_data": episode_data
    }

def get_training_chart():
    """Generate training visualization"""
    global training_history
    
    if not training_history["episodes"]:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=training_history["episodes"],
        y=training_history["rewards"],
        mode='lines+markers',
        name='Episode Reward',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="Training Progress - Rewards",
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        hovermode='x unified'
    )
    
    return fig

def get_temp_chart():
    """Generate temperature visualization"""
    global training_history
    
    if not training_history["episodes"]:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=training_history["episodes"],
        y=training_history["cpu_temps"],
        mode='lines+markers',
        name='CPU Temperature',
        line=dict(color='red'),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Training Progress - CPU Temperature",
        xaxis_title="Episode",
        yaxis_title="Temperature (°C)",
        hovermode='x unified'
    )
    
    return fig

def get_model_info():
    """Get model information"""
    global model_state, current_session
    
    if model_state["policy"] is None:
        return "❌ No model trained yet"
    
    policy = model_state["policy"]
    param_count = sum(p.numel() for p in policy.parameters())
    
    info = f"""
    ## 🤖 Model Information
    
    **Status:** ✅ Trained
    
    **Total Parameters:** {param_count}
    
    **Architecture:**
    - Input Layer: 3 neurons (CPU Temp, Battery, Accuracy)
    - Hidden Layer: 16 neurons (ReLU)
    - Output Layer: 2 neurons (Softmax - Action probabilities)
    
    **Training Episodes:** {current_session["episode"] if current_session else 0}
    
    **Session Duration:** {str(datetime.now() - current_session["session_start"]) if current_session else "N/A"}
    """
    
    return info

# Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="AlgoBrain RL Interface") as demo:
    gr.Markdown("# 🧠 AlgoBrain - Reinforcement Learning Environment")
    gr.Markdown("Train an RL agent to optimize algorithm performance with resource constraints")
    
    with gr.Tabs():
        # ========== TRAINING TAB ==========
        with gr.TabItem("🎓 Training"):
            gr.Markdown("### Train the Policy Network")
            
            with gr.Row():
                episodes_input = gr.Slider(
                    minimum=5, maximum=100, value=30, step=5,
                    label="Number of Episodes"
                )
                lr_input = gr.Slider(
                    minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                    label="Learning Rate"
                )
            
            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            
            with gr.Row():
                status_output = gr.Textbox(label="Status", interactive=False)
                final_reward = gr.Textbox(label="Final Reward", interactive=False)
            
            with gr.Row():
                avg_reward = gr.Textbox(label="Average Reward", interactive=False)
                avg_temp = gr.Textbox(label="Average Temperature", interactive=False)
            
            train_btn.click(
                fn=train_model,
                inputs=[episodes_input, lr_input],
                outputs=[status_output, final_reward, avg_reward, avg_temp]
            )
        
        # ========== INFERENCE TAB ==========
        with gr.TabItem("🚀 Inference"):
            gr.Markdown("### Run Inference with Trained Model")
            
            with gr.Row():
                task_select = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Task Difficulty"
                )
                steps_input = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Number of Steps"
                )
            
            inference_btn = gr.Button("Run Inference", variant="primary", size="lg")
            
            with gr.Row():
                task_output = gr.Textbox(label="Task", interactive=False)
                reward_output = gr.Textbox(label="Total Reward", interactive=False)
                steps_output = gr.Textbox(label="Steps Completed", interactive=False)
            
            episode_table = gr.Dataframe(
                label="Episode Details",
                interactive=False,
                type="pandas"
            )
            
            inference_btn.click(
                fn=lambda task, steps: (
                    run_inference(task, steps).get("task", ""),
                    run_inference(task, steps).get("total_reward", ""),
                    run_inference(task, steps).get("steps_completed", ""),
                    pd.DataFrame(run_inference(task, steps).get("episode_data", []))
                ),
                inputs=[task_select, steps_input],
                outputs=[task_output, reward_output, steps_output, episode_table]
            )
        
        # ========== VISUALIZATION TAB ==========
        with gr.TabItem("📊 Visualization"):
            gr.Markdown("### Training Metrics")
            
            with gr.Row():
                reward_chart = gr.Plot(label="Reward Chart")
                temp_chart = gr.Plot(label="Temperature Chart")
            
            refresh_btn = gr.Button("Refresh Charts")
            
            refresh_btn.click(
                fn=lambda: (get_training_chart(), get_temp_chart()),
                outputs=[reward_chart, temp_chart]
            )
        
        # ========== MODEL INFO TAB ==========
        with gr.TabItem("ℹ️ Model Info"):
            gr.Markdown("### Model Architecture & Details")
            
            info_output = gr.Markdown()
            
            refresh_info_btn = gr.Button("Refresh Info")
            
            refresh_info_btn.click(
                fn=get_model_info,
                outputs=info_output
            )
        
        # ========== ENVIRONMENT TAB ==========
        with gr.TabItem("🌍 Environment"):
            gr.Markdown("### AlgoBrain Environment Configuration")
            
            env_info = gr.Markdown("""
            ## Tasks
            - **Easy**: Optimize accuracy with moderate resource constraints (Reward Scale: 1.0)
            - **Medium**: Balance performance and efficiency (Reward Scale: 0.5)
            - **Hard**: Maximize efficiency with minimal resources (Reward Scale: 0.2)
            
            ## State Space
            - CPU Temperature: 50-95°C (safety limit at 95°C)
            - Battery Level: 0-100 mAh
            - Accuracy Score: 0.0-1.0
            
            ## Action Space
            - **Action 0 (IDLE)**: Reduce CPU temp, slightly recharge battery
            - **Action 1 (RUN)**: Increase accuracy, consume resources (CPU heat, battery)
            
            ## Reward Structure
            - Base task reward (depends on difficulty)
            - Safety penalty if CPU > 95°C (-1.0)
            - Episode length: 10 steps max
            """)
    
    # Initialize session on load
    demo.load(fn=initialize_session)

if __name__ == "__main__":
    demo.launch(share=True)
