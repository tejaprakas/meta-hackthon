import os
import torch
import torch.nn as nn
import torch.optim as optim
from environment import AlgoBrainEnv

# ENV VARIABLES (MANDATORY)
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Policy Network
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


def get_obs_tensor(obs):
    return torch.tensor([
        obs['cpu_temp'][0],
        obs['battery_maH'][0],
        obs['accuracy_score'][0]
    ], dtype=torch.float32)


# 🔥 TRAINING (REINFORCE)
def train(policy, episodes=30):
    env = AlgoBrainEnv()
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    for ep in range(episodes):
        obs, _ = env.reset()

        log_probs = []
        rewards = []

        done = False

        while not done:
            obs_tensor = get_obs_tensor(obs)
            probs = policy(obs_tensor)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))

            obs, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

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


# 🔥 INFERENCE (EVALUATION)
def run():
    env = AlgoBrainEnv()
    policy = PolicyNet()

    # Train quickly before inference
    train(policy, episodes=30)

    obs, _ = env.reset()

    print("[START] Algo Brain RL Inference")

    total_reward = 0

    for step in range(10):
        obs_tensor = get_obs_tensor(obs)
        probs = policy(obs_tensor)
        action = torch.argmax(probs).item()

        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        print(f"[STEP] step={step} action={action} reward={reward:.2f} temp={obs['cpu_temp'][0]:.2f}")

        if done:
            break

    print(f"[END] total_reward={total_reward:.2f}")


if __name__ == "__main__":
    run()
