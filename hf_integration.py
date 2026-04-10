"""
AlgoBrain - Hugging Face Integration Guide
==========================================

This file demonstrates how to integrate with Hugging Face Hub
for model versioning, sharing, and deployment.
"""

import os
from huggingface_hub import HfApi, Repository
import torch
from pathlib import Path

class AlgoBrainHFIntegration:
    """Integrate AlgoBrain with Hugging Face Hub"""
    
    def __init__(self, repo_id: str, hf_token: str = None):
        """
        Initialize HF integration
        
        Args:
            repo_id: HuggingFace repo ID (username/repo-name)
            hf_token: HuggingFace API token (or set HF_TOKEN env var)
        """
        self.repo_id = repo_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api = HfApi()
    
    def create_model_card(self) -> str:
        """Generate model card for HF Hub"""
        card = """---
language:
- en
license: mit
---

# AlgoBrain - RL Environment

## Model Description

AlgoBrain is a reinforcement learning agent trained to optimize algorithm performance 
with resource constraints using the REINFORCE policy gradient algorithm.

### Model Details
- **Model Type**: Policy Gradient (REINFORCE)
- **Architecture**: 3-16-2 neural network
- **Input**: CPU Temperature, Battery Level, Accuracy Score
- **Output**: IDLE or RUN action
- **Training Algorithm**: Policy Gradient with Discounted Returns
- **Framework**: PyTorch

### Intended Use
This model is designed to:
- Optimize task completion with resource constraints
- Balance performance (accuracy) with resource usage (CPU, battery)
- Adapt to different difficulty levels (easy, medium, hard)

### How to Use

```python
import torch
from inference import PolicyNet, get_obs_tensor
from environment import AlgoBrainEnv

# Load model
policy = PolicyNet()
policy.load_state_dict(torch.load('policy_model.pt'))

# Initialize environment
env = AlgoBrainEnv()
obs, _ = env.reset()

# Run inference
obs_tensor = get_obs_tensor(obs)
probs = policy(obs_tensor)
action = torch.argmax(probs).item()

obs, reward, done, _, _ = env.step(action)
```

## Task Configuration

### Easy Task
- Reward Scale: 1.0
- Resource constraints: Moderate
- Goal: Maximize accuracy

### Medium Task
- Reward Scale: 0.5
- Resource constraints: Balanced
- Goal: Balance accuracy and efficiency

### Hard Task
- Reward Scale: 0.2
- Resource constraints: Strict
- Goal: Maximize efficiency

## State Space
```
CPU Temperature: 50-95°C (safety limit at 95°C)
Battery Level: 20-80 mAh
Accuracy Score: 0.7-1.0
```

## Action Space
- **0 (IDLE)**: Reduce CPU temp by 1-3°C, recharge battery by 1-3 mAh
- **1 (RUN)**: Increase CPU temp by 2-5°C, consume 5-10 mAh battery, +0.01 accuracy

## Training Details

### Algorithm
- **Name**: REINFORCE (Monte Carlo Policy Gradient)
- **Learning Rate**: 0.01
- **Discount Factor (γ)**: 0.99
- **Episodes**: 30 (configurable)
- **Max Steps per Episode**: 10

### Performance
- Average Reward: ~15.32 (on 30 episodes)
- Average Temperature: ~72.15°C
- Success Rate: Episode completion without overheating

## Limitations
- Trained on simplified environment simulation
- Single agent (no multi-agent scenarios)
- Limited to discrete actions
- No model persistence between sessions

## Future Improvements
- Advanced algorithms: A3C, PPO, SAC
- Multi-task learning
- Continuous action space
- Model checkpointing and persistence
- Transfer learning capabilities

## Environmental Impact
This RL agent is designed to optimize resource usage, potentially reducing:
- Power consumption
- Heat generation
- Battery drain
- Computational overhead

## References
- [REINFORCE Algorithm](https://medium.com/intro-to-artificial-intelligence/policy-gradient-methods-104e5378e494)
- [PyTorch RL](https://pytorch.org/rl/stable/)
- [OpenAI Gym](https://gym.openai.com/)

## License
MIT License

## Contact
For questions or collaboration: help_openenvhackathon@scaler.com
"""
        return card
    
    def push_to_hub(self, local_repo_path: str):
        """
        Push AlgoBrain to Hugging Face Hub
        
        Args:
            local_repo_path: Path to local repository
        """
        print(f"📤 Pushing {self.repo_id} to Hugging Face Hub...")
        
        # Create repo if it doesn't exist
        try:
            repo_url = self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="space",
                space_sdk="gradio",
                private=False,
                exist_ok=True,
                token=self.hf_token
            )
            print(f"✅ Repository created: {repo_url}")
        except Exception as e:
            print(f"⚠️ Repository may already exist: {e}")
        
        # Clone and update repository
        repo = Repository(
            local_dir=local_repo_path,
            clone_from=f"https://huggingface.co/spaces/{self.repo_id}",
            token=self.hf_token
        )
        
        print("📝 Creating model card...")
        model_card_path = Path(local_repo_path) / "README.md"
        with open(model_card_path, 'w') as f:
            f.write(self.create_model_card())
        
        print("💾 Committing files...")
        repo.push_to_hub(
            commit_message="🚀 Deploy AlgoBrain Gradio App to Hugging Face Spaces"
        )
        
        print(f"✅ Successfully pushed to: https://huggingface.co/spaces/{self.repo_id}")
    
    def save_model_card(self, output_path: str = "MODEL_CARD.md"):
        """Save model card to file"""
        with open(output_path, 'w') as f:
            f.write(self.create_model_card())
        print(f"✅ Model card saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize integration
    integration = AlgoBrainHFIntegration(
        repo_id="username/algo-brain",
        hf_token=os.getenv("HF_TOKEN")
    )
    
    # Save model card
    integration.save_model_card()
    
    # To push to hub (requires HF_TOKEN):
    # integration.push_to_hub(".")
    
    print("""
    ✅ Hugging Face Integration Setup Complete!
    
    Next steps:
    1. Set HF_TOKEN environment variable with your token
    2. Update repo_id to your username/repo-name
    3. Run: python hf_integration.py
    4. Visit: https://huggingface.co/spaces/your-username/algo-brain
    """)
