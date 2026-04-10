# AlgoBrain - RL Environment Web App

A fully functional web application for training and running reinforcement learning agents to optimize algorithm performance with resource constraints.

## 🚀 Features

### 1. **Training Interface** 
- Train the policy network with configurable episodes and learning rate
- Real-time feedback on training progress
- Track rewards and CPU temperatures
- REINFORCE algorithm implementation

### 2. **Inference Interface**
- Run trained models on different task difficulties (Easy, Medium, Hard)
- Visualize step-by-step decisions
- Monitor resource metrics (CPU temp, battery, accuracy)
- Detailed episode logs

### 3. **Visualization Dashboard**
- Track reward trends across episodes
- Monitor CPU temperature evolution
- Interactive Plotly charts
- Real-time metrics

### 4. **Model Information**
- Display model architecture
- Parameter count
- Training statistics
- Session information

### 5. **Environment Configuration**
- Task-specific reward scaling
- Dynamic state space
- Resource constraints and safety limits

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│         Gradio Web Interface                     │
├─────────────────────────────────────────────────┤
│  Training │ Inference │ Visualization │ Model   │
├─────────────────────────────────────────────────┤
│     PolicyNet (PyTorch RL Agent)                │
├─────────────────────────────────────────────────┤
│    AlgoBrainEnv (Gym-like Environment)          │
├─────────────────────────────────────────────────┤
│  State: CPU Temp, Battery, Accuracy Score      │
│  Actions: IDLE, RUN                            │
│  Rewards: Task-based + Safety Penalties        │
└─────────────────────────────────────────────────┘
```

## 📊 Model Details

**Neural Network Architecture:**
- Input: 3 features (CPU Temp, Battery, Accuracy)
- Hidden: 16 neurons (ReLU activation)
- Output: 2 actions (Softmax probabilities)
- Total Parameters: ~50

**Training Algorithm:** REINFORCE (Policy Gradient)

**Hyperparameters:**
- Learning Rate: 0.01 (adjustable)
- Discount Factor (γ): 0.99
- Episodes: 30 (adjustable)

## 🌍 Environment

### Tasks
- **Easy**: Optimize accuracy with moderate constraints (Reward: 1.0 base)
- **Medium**: Balance performance and efficiency (Reward: 0.5 base)
- **Hard**: Maximize efficiency with minimal resources (Reward: 0.2 base)

### State Space
```
CPU Temperature: [50, 95]°C
Battery Level: [20, 80] mAh
Accuracy Score: [0.7, 1.0]
```

### Action Space
- **Action 0 (IDLE)**: 
  - CPU Temp: -1 to -3°C
  - Battery: +1 to +3 mAh
  
- **Action 1 (RUN)**:
  - CPU Temp: +2 to +5°C
  - Battery: -5 to -10 mAh
  - Accuracy: +0.01

### Rewards
- Base reward: Task-dependent
- Safety penalty: -1.0 if CPU Temp > 95°C (episode ends)
- Episode termination: 10 steps or safety violation

## 🔧 Installation

### Local Setup
```bash
# Clone repository
git clone <repo-url>
cd algo-brain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### Docker Setup
```bash
# Build image
docker build -t algo-brain .

# Run container
docker run -p 7860:7860 algo-brain
```

### Hugging Face Spaces
1. Create new Space on Hugging Face Hub
2. Choose Gradio as SDK
3. Upload files:
   - `app.py`
   - `environment.py`
   - `inference.py`
   - `requirements.txt`
   - `README.md`
4. App auto-deploys and goes live!

## 📈 Usage

### Training
1. Navigate to **Training** tab
2. Set desired episodes (5-100)
3. Adjust learning rate if needed (0.001-0.1)
4. Click "Start Training"
5. Monitor final reward, average reward, and temperature metrics

### Running Inference
1. Navigate to **Inference** tab
2. Select task difficulty
3. Set number of steps (1-20)
4. Click "Run Inference"
5. View detailed step-by-step results in table

### Monitoring Performance
1. Go to **Visualization** tab
2. Click "Refresh Charts" to update plots
3. Analyze reward trends and temperature evolution
4. Check **Model Info** tab for architecture details

## 📊 Example Results

### Training Output
```
Status: ✅ Training completed! 30 episodes trained
Final Reward: 18.45
Average Reward: 15.32
Final Temperature: 78.50°C
Average Temperature: 72.15°C
```

### Inference Output
```
Task: medium
Total Reward: 8.32
Steps Completed: 10

| Step | Action | Reward | CPU Temp | Battery | Accuracy |
|------|--------|--------|----------|---------|----------|
| 0    | RUN    | 0.50   | 73.25°C  | 65.32mAh| 0.7124   |
| 1    | IDLE   | 0.50   | 71.10°C  | 68.45mAh| 0.7124   |
...
```

## 🔬 Technical Details

### Dependencies
- **torch**: Deep learning framework
- **gradio**: Web UI framework
- **plotly**: Interactive visualizations
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **requests**: HTTP library
- **huggingface-hub**: HF integration

### File Structure
```
algo-brain/
├── app.py                 # Main Gradio application
├── environment.py         # AlgoBrainEnv definition
├── inference.py          # PolicyNet and training logic
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── README.md            # This file
├── openenv.yaml        # Environment config
└── spaces_config.yml   # HF Spaces config
```

## 🎓 How It Works

1. **Environment Steps:**
   - Agent observes state (CPU temp, battery, accuracy)
   - Samples action from policy network
   - Environment updates state and provides reward
   - Episode ends after 10 steps or safety violation

2. **Training:**
   - Uses REINFORCE algorithm (basic policy gradient)
   - Iterates through episodes collecting trajectories
   - Computes discounted returns
   - Updates policy via gradient descent

3. **Inference:**
   - Uses greedy action selection (argmax)
   - Visualizes step-by-step decisions
   - Tracks resource utilization

## 🚀 Deployment

### Option 1: Hugging Face Spaces (Easiest)
- Push repo to HF Spaces
- Auto-deploys with Gradio
- Free tier available
- Share link publicly

### Option 2: Docker Container
- Build image: `docker build -t algo-brain .`
- Deploy on any platform (AWS, GCP, Azure, etc.)
- Reproducible environment
- Production-ready

### Option 3: Local Server
- Run `python app.py`
- Access at `http://localhost:7860`
- Development/testing only

## 📝 Notes

- Model trained from scratch for each session
- Inference requires prior training
- GPU acceleration available if CUDA installed
- Charts require episode data (train first)

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- More sophisticated RL algorithms (A3C, PPO, SAC)
- Multi-agent environments
- Curriculum learning
- Model persistence/loading
- Distributed training

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Links

- [Gradio Docs](https://gradio.app)
- [PyTorch Docs](https://pytorch.org)
- [Hugging Face Hub](https://huggingface.co)
- [Plotly Docs](https://plotly.com)

---

**Happy Training! 🧠🚀**
