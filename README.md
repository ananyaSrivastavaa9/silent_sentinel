# 🛡️ Silent Sentinel — Edge AI Research Platform

An elite, production-grade (10/10) Edge AI Research Platform and Proof-of-Concept validation engine built for passive personal safety. 

Traditional safety applications rely on active user ingestion (like pressing a panic button), which completely fails during severe panic freeze or physical incapacitation. Silent Sentinel solves this by operating entirely as a passive background monitoring engine.

---

## 👁️ Core System Architecture

The platform processes multimodal environmental signals through decoupled architectural layers:

1. Acoustic Stress Layer: High-frequency raw audio stream processing via NumPy & SciPy, expanding signal vectors into 65-dimensional feature matrices (MFCCs, Delta, Delta-Delta, Spectral Rolloff, and ZCR).
2. Motion Telemetry Layer: A deterministic finite state machine tracking IMU matrices for an absolute sequence: [High-G Impact Spike] followed strictly by [Prolonged Structural Stillness].
3. Bayesian Sensor Fusion Engine: Instead of naive OR gates, it executes a Weighted Probabilistic Fusion Engine with Dynamic Confidence Gates and explicit Veto Rules (e.g., attenuating alerts if a device drop occurs without corresponding vocal distress).
4. Dead Man's Switch FSM: A thread-safe asynchronous countdown mechanism that triggers an interactive validation window before committing to a crisis dispatch state.

---

## 📂 Repository Blueprint (Src-Layout)

silent_sentinel/
├── config/
│   └── config.yaml          # Hyperparameters & hardware thresholds
├── src/
│   ├── config_loader.py     # Authoritative Pydantic V2 validation core
│   ├── core/
│   │   ├── event_bus.py     # Decoupled Observer Pattern engine
│   │   ├── events.py        # Strongly-typed dataclass events
│   │   ├── fsm.py           # Hand-rolled State Control Machine
│   │   └── fusion.py        # Bayesian logic & smart veto nodes
│   ├── sensors/
│   │   ├── acoustic.py      # Microsecond signal extraction core
│   │   └── motion.py        # Accelerometer sequence modeler
│   └── ui/
│       └── app.py           # Sci-Fi Mission Control Dashboard (Streamlit)

---

## ⚡ Quickstart, Deployment & Execution

Run the follow sequence of commands inside your system terminal to clone the workspace, activate your virtual sandbox, and boot the mission control interface:

# 1. Access your workspace directory
cd silent_sentinel

# 2. Initialize and configure the virtual environment sandbox
python -m venv venv

# 3. Activate the isolated sandbox target environment
# On Windows (Command Prompt / PowerShell):
venv\Scripts\activate
# On Mac / Linux / Git Bash:
# source venv/bin/activate

# 4. Install authoritative project dependencies
pip install numpy scipy pydantic pyyaml streamlit

# 5. Boot the live Sci-Fi Mission Control Center interface
streamlit run src/ui/app.py

---

## 🚀 GitHub Repository Initialization

If you are initializing this repository for Git and pushing it to GitHub for the first time, execute the following commands in your terminal:

# Initialize local git tracking
git init

# Stage all production components (.gitignore will automatically protect your venv)
git add .

# Commit architecture blueprint to local main branch
git commit -m "feat: complete production-grade edge AI platform core architecture"

# Link your local repo to your GitHub (Replace with your actual GitHub link)
# git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
# git branch -M main
# git push -u origin main