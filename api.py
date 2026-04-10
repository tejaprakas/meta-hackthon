"""
AlgoBrain OpenEnv REST API
Provides endpoints for automated validation
"""

from flask import Flask, jsonify, request
import json
from environment import AlgoBrainEnv
from inference import PolicyNet, get_obs_tensor, train
import torch

app = Flask(__name__)

# Global state
env = None
policy = None
trained = False

def initialize():
    """Initialize environment and policy"""
    global env, policy, trained
    env = AlgoBrainEnv()
    policy = PolicyNet()
    trained = False

initialize()

@app.route("/", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "algobrain-env",
        "version": "1.0"
    }), 200

@app.route("/reset", methods=["POST"])
def reset():
    """Reset environment and return initial observation"""
    global env, policy, trained
    try:
        # Train policy if not already trained
        if not trained:
            train(policy, episodes=5)
            trained = True
        
        obs, info = env.reset()
        
        # Convert observation to JSON-serializable format
        obs_json = {
            "cpu_temp": float(obs["cpu_temp"][0]),
            "battery_maH": float(obs["battery_maH"][0]),
            "accuracy_score": float(obs["accuracy_score"][0]),
            "task": env.current_task
        }
        
        return jsonify({
            "observation": obs_json,
            "info": info,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/step", methods=["POST"])
def step():
    """Execute one step in the environment"""
    try:
        data = request.get_json()
        action = data.get("action", 0)
        
        obs, reward, done, truncated, info = env.step(int(action))
        
        obs_json = {
            "cpu_temp": float(obs["cpu_temp"][0]),
            "battery_maH": float(obs["battery_maH"][0]),
            "accuracy_score": float(obs["accuracy_score"][0])
        }
        
        return jsonify({
            "observation": obs_json,
            "reward": float(reward),
            "done": bool(done),
            "truncated": bool(truncated),
            "info": info,
            "status": "success"
        }), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/state", methods=["GET"])
def state():
    """Get current environment state"""
    try:
        if env is None:
            return jsonify({"error": "Environment not initialized"}), 500
        
        state_json = {
            "current_task": env.current_task,
            "step_count": env.step_count,
            "max_steps": env.max_steps,
            "observation": {
                "cpu_temp": float(env.state["cpu_temp"][0]),
                "battery_maH": float(env.state["battery_maH"][0]),
                "accuracy_score": float(env.state["accuracy_score"][0])
            },
            "status": "success"
        }
        return jsonify(state_json), 200
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/tasks", methods=["GET"])
def tasks():
    """Get available tasks"""
    tasks_info = [
        {
            "name": "easy",
            "description": "Easy task - optimize accuracy with moderate constraints",
            "reward_scale": 1.0,
            "difficulty_level": 1
        },
        {
            "name": "medium",
            "description": "Medium task - balance performance and efficiency",
            "reward_scale": 0.5,
            "difficulty_level": 2
        },
        {
            "name": "hard",
            "description": "Hard task - maximize efficiency with minimal resources",
            "reward_scale": 0.2,
            "difficulty_level": 3
        }
    ]
    
    return jsonify({
        "tasks": tasks_info,
        "status": "success"
    }), 200

@app.route("/info", methods=["GET"])
def info():
    """Get environment information"""
    return jsonify({
        "name": "algo-brain-env",
        "version": "1.0",
        "description": "AlgoBrain RL Environment - Optimize algorithm performance with resource constraints",
        "observation_space": {
            "type": "box",
            "shape": [3],
            "bounds": [[50, 95], [20, 80], [0.7, 1.0]],
            "features": ["cpu_temp", "battery_maH", "accuracy_score"]
        },
        "action_space": {
            "type": "discrete",
            "n": 2,
            "actions": ["IDLE", "RUN"]
        },
        "max_episode_steps": 10,
        "status": "success"
    }), 200

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
