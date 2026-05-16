AURAX Evo 🧬
Recursive Self-Improvement System for Autonomous AI Evolution

A production-grade framework for autonomous AI self-improvement through synthetic data generation, curriculum learning, benchmarking, safety validation, and evolutionary promotion pipelines.

🚀 Overview

AURAX Evo is an experimental recursive self-improvement architecture designed to evolve AI models through automated training cycles.

The system continuously:

Generates synthetic training data
Fine-tunes candidate models
Evaluates benchmark performance
Runs safety validation
Promotes improved models
Adapts curriculum difficulty over time

It combines:

Synthetic data generation
Curriculum learning
Automated evaluation
Safety monitoring
Model evolution tracking
Real-time dashboards

Core orchestration is handled inside evo_loop.py.

✨ Features
🧠 Recursive Evolution Loop
Automated generation-based training pipeline
Candidate vs current model evaluation
Promotion logic based on score improvement
Human approval checkpoints
Graceful checkpoint recovery

Implemented in evo_loop.py.

📚 Adaptive Curriculum Learning

Dynamic difficulty shifting based on model performance.

Features:

Easy → Medium → Hard progression
Tier-based accuracy tracking
Adaptive sampling weights
Curriculum persistence across generations

Implemented in curriculum.py.

🧪 Synthetic Data Generation

Generates training datasets across multiple domains:

Code generation
Vulnerability remediation
Reasoning tasks
Technical Q&A

Includes:

Deduplication
Metadata tracking
Difficulty scaling
Ollama + Transformers support

Implemented in synthetic_gen.py.

🛡 Advanced Safety Monitoring

Prevents unsafe or degraded models from promotion.

Safety checks include:

Harmful content detection
Refusal-rate validation
Diversity entropy monitoring
Perplexity drift detection
Capability regression tests

Implemented in safety.py.

📊 Benchmark Validation Engine

Automated benchmark suite supporting:

HumanEval-style coding tests
GSM8K reasoning evaluation
Security refusal testing
Q&A heuristic scoring

Implemented in validator.py.

📈 Real-Time Dashboard

Industrial-style Streamlit monitoring dashboard with:

Live generation history
Score progression graphs
Safety alerts
Promotion tracking
Curriculum visualization
Benchmark radar charts

Implemented in dashboard.py.

🏗 Architecture
                ┌────────────────────┐
                │ Synthetic Generator │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Fine-Tuning Engine │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Safety Validation  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Benchmark Validator│
                └─────────┬──────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Promote Model             Reject Candidate
             │
             ▼
    Update Curriculum State
📂 Project Structure
AURAX-Evo/
│
├── evo_loop.py            # Main recursive evolution engine
├── synthetic_gen.py       # Synthetic dataset generation
├── curriculum.py          # Adaptive curriculum manager
├── validator.py           # Benchmark & evaluation system
├── safety.py              # Safety validation framework
├── dashboard.py           # Streamlit monitoring dashboard
├── config_evo.yaml        # System configuration
├── requirements_evo.txt   # Python dependencies
├── run_evo.sh             # Launcher script
│
├── data/
├── models/
├── logs/
└── checkpoints/
⚡ Installation
1. Clone Repository
git clone https://github.com/yourusername/aurax-evo.git
cd aurax-evo
2. Install Dependencies
pip install -r requirements_evo.txt

Requirements include:

PyTorch
Transformers
PEFT
Streamlit
Plotly
Accelerate
Datasets

Defined in requirements_evo.txt.

3. (Optional) CUDA Support
pip install torch --index-url https://download.pytorch.org/whl/cu121
🚀 Running AURAX Evo
Launch Evolution Loop
python evo_loop.py

OR use the launcher:

chmod +x run_evo.sh
./run_evo.sh

Launcher supports:

Dry runs
Auto approval
Dashboard-only mode
GPU selection

Defined in run_evo.sh.

🖥 Dashboard

Start the monitoring dashboard:

streamlit run dashboard.py -- --config config_evo.yaml

Dashboard features are implemented in dashboard.py.

⚙ Configuration

Main configuration file:

config_evo.yaml

Typical configuration sections:

Generation settings
Model backend
Curriculum strategy
Safety thresholds
Benchmark weights
Training paths
🔁 Evolution Pipeline

Each generation executes:

1. Generate synthetic data
2. Fine-tune candidate model
3. Run safety checks
4. Evaluate benchmarks
5. Compare against current model
6. Promote if improved
7. Update curriculum

Defined in evo_loop.py.

🧩 Supported Backends
Transformers
Qwen
Llama
Mistral
Phi
Gemma
Ollama
Local inference support
Lightweight deployment

Implemented in synthetic_gen.py.

🔒 Safety Philosophy

AURAX Evo prioritizes:

Safe recursive improvement
Controlled promotion logic
Human approval gates
Capability retention
Harmful output filtering

Safety monitoring is implemented in safety.py.

📊 Benchmarks
Benchmark	Purpose
HumanEval	Code generation
GSM8K	Mathematical reasoning
Security Tests	Harmful request refusal
QA Heuristics	General response quality

Implemented in validator.py.

🧠 Curriculum Learning

Difficulty tiers:

Easy
Medium
Hard

Adaptive curriculum shifting:

Performance-based progression
Never fully removes easier tasks
Tracks historical distributions

Implemented in curriculum.py.

📌 Example Commands
Dry Run
./run_evo.sh --dry-run
Auto Approval Mode
./run_evo.sh --auto-approve
Dashboard Only
./run_evo.sh --dashboard-only
🛠 Future Roadmap
Multi-agent cooperative evolution
Distributed training clusters
Reinforcement self-play
Long-term memory systems
Autonomous architecture search
Multi-modal training support
Real-time online learning
Federated evolution
⚠ Disclaimer

AURAX Evo is an experimental research framework focused on autonomous model evolution and AI safety research.

It is intended for:

AI research
Benchmark experimentation
Safe self-improvement studies
Educational purposes

Human oversight is strongly recommended during recursive training cycles.

👨‍💻 Author

AURAX Industries
Recursive Intelligence Systems Division

📜 License
MIT License
⭐ Final Note

AURAX Evo is not just a training script.

It is the foundation of a recursive intelligence ecosystem capable of:

self-evaluation,
self-improvement,
curriculum adaptation,
and controlled autonomous evolution.

🧬 Recursive Intelligence Starts Here.
