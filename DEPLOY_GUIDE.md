# AlgoBrain - OpenEnv Hackathon Submission

Build complete RL environments following the OpenEnv specification for automated evaluation.

## 📋 Project Structure

```
algo-brain/
├── environment.py          # AlgoBrainEnv (Gym-like interface)
├── inference.py            # Training & evaluation with [START]/[STEP]/[END] logs
├── inference_app.py        # Gradio app for HF Spaces (entry point)
├── api.py                  # REST API endpoints for validation
├── app.py                  # Full-featured web interface (optional)
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── spaces_config.yml       # HF Spaces configuration
└── README.md              # This file
```

## 🚀 Quick Start

### Local Testing (Evaluation)
```bash
# Run inference with structured logging
python inference.py

# Output format:
# [START] Algo Brain RL Inference
# [STEP] step=0 action=1 reward=1.00 temp=73.45
# [STEP] step=1 action=0 reward=1.00 temp=71.20
# ...
# [END] total_reward=12.34
```

### HuggingFace Spaces Deployment
```bash
# The app_file in spaces_config.yml points to inference_app.py
# This provides a Gradio interface for the validator

# HF will automatically run:
# gradio run inference_app.py --listen 0.0.0.0 --port 7860
```

### Local Gradio Interface
```bash
python inference_app.py
# Visit: http://localhost:7860
```

## 📦 Files Explained

### Core Implementation
- **environment.py**: Implements `AlgoBrainEnv` with `reset()`, `step()`, `state()` methods
- **inference.py**: Implements `PolicyNet`, training logic, and inference with required logging format

### Evaluation Interfaces
- **inference_app.py**: Minimal Gradio app for HF Spaces validation (primary entry point)
- **api.py**: REST API endpoints for direct validation requests
- **app.py**: Full-featured web dashboard (optional, not used for evaluation)

### Configuration
- **openenv.yaml**: OpenEnv specification with tasks, state space, action space
- **spaces_config.yml**: Tells HF Spaces to run `inference_app.py`
- **Dockerfile**: Containerization for local/cloud deployment

## 🔧 OpenEnv Specification

### Environment Definition
```python
entry_point: environment:AlgoBrainEnv
```

### State Space
- **Type**: Box (continuous, 3D)
- **Values**: [CPU Temp, Battery, Accuracy]
- **Bounds**: [[50,95], [20,80], [0.7,1.0]]

### Action Space
- **Type**: Discrete
- **Size**: 2 (IDLE=0, RUN=1)

### Available Tasks
1. **easy** (reward_scale: 1.0)
2. **medium** (reward_scale: 0.5)
3. **hard** (reward_scale: 0.2)

## 📝 API Endpoints

The REST API (`api.py`) provides these endpoints for validation:

```bash
# Health check
GET /

# Reset environment
POST /reset

# Execute action
POST /step
# Body: {"action": 0}

# Get current state
GET /state

# List tasks
GET /tasks

# Environment info
GET /info
```

## 🧪 Validation Checklist

**Pre-Submission Requirements:**
- [x] `environment.py` at repo root
- [x] `inference.py` at repo root with [START]/[STEP]/[END] logging
- [x] `openenv.yaml` with proper schema
- [x] Dockerfile builds successfully
- [x] `requirements.txt` with all dependencies
- [x] HF Space deploys (runs `inference_app.py`)
- [x] Inference script completes in < 20 min
- [x] Proper environment variables handling

## 🔍 Troubleshooting

### If HF Space won't start:
1. Check `spaces_config.yml` points to `inference_app.py`
2. Verify all imports in `inference_app.py` work locally
3. Run: `python -c "import inference_app"` to test imports
4. Check HF Space logs for specific error

### If validation fails:
1. Run locally: `python validate.py`
2. Run inference: `python inference.py`
3. Check logs match [START], [STEP], [END] format exactly
4. Verify environment variables are accessible

### If imports fail:
1. Install dependencies: `pip install -r requirements.txt`
2. Check Python version (should be 3.10+)
3. Verify CUDA/PyTorch installation: `python -c "import torch; print(torch.__version__)"`

## 🎓 How It Works

1. **Environment Reset**: `reset()` returns initial state and info
2. **Step Execution**: `step(action)` returns (observation, reward, done, truncated, info)
3. **Policy Training**: REINFORCE algorithm optimizes action selection
4. **Inference**: Runs trained policy on environment, logs results in structured format

## 📊 Reward Structure

```
base_reward = task_difficulty_scale  # 1.0, 0.5, or 0.2
safety_penalty = -1.0 if cpu_temp > 95 else 0.0
total_reward = base_reward + safety_penalty
```

## 🚢 Deployment Options

### Option 1: HuggingFace Spaces (Recommended)
```bash
# Push to repo, HF auto-deploys
git push origin main
```

### Option 2: Docker on Local Machine
```bash
docker build -t algo-brain .
docker run -p 7860:7860 algo-brain
```

### Option 3: Direct Python Execution
```bash
# Inference evaluation
python inference.py

# Gradio interface
python inference_app.py

# REST API
python api.py
```

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- Gradio 4.0+
- Flask 2.0+ (for REST API)
- Hugging Face Hub tools

Install all:
```bash
pip install -r requirements.txt
```

## 🔐 Environment Variables

Required for full functionality:

```bash
export API_BASE_URL="https://api.example.com"
export MODEL_NAME="gpt-4"
export HF_TOKEN="hf_xxxxxxxxxxxx"
```

(Currently using stub values for OpenEnv compatibility)

## 📞 Support

For issues with:
- **OpenEnv spec**: See [openenv.yaml](openenv.yaml)
- **Environment design**: See [environment.py](environment.py)
- **RL algorithm**: See [inference.py](inference.py)
- **HF Spaces**: See [spaces_config.yml](spaces_config.yml)

---

**Status**: ✅ Ready for Submission

All validation tests passing. Environment is fully functional and deployed to HuggingFace Spaces.
