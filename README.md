# 🛡️ Silent Sentinel — Edge AI Passive Safety Platform
**Designed & Engineered by: Ananya Srivastava**

---

### 💡 The Big Vision: Why am I building this?
Imagine a traditional safety app. When someone is in real danger or freezes out of sheer panic, they **cannot** unlock their phone, open an app, and press a "Panic Button." Active safety fails when you need it the most.

**Silent Sentinel** is my solution to this problem. It is a smart background engine designed to monitor personal safety **passively**. 

Right now, I am developing this as an **Edge AI Research Platform** to stress-test high-level signal processing and sensor fusion logic locally on a desktop environment before deploying it natively to mobile devices (React Native + TFLite). No cloud delays, no privacy leaks—everything stays on the edge.

---

## 🚦 Current Project Status & Live Roadmap

To make it easy for researchers and developers to track my progress, here is the exact lifecycle of the project. I am updating this repository step-by-step as I build and test each module!

### ✅ Phase 1: Core Architecture & UI Dashboard (COMPLETED)
* **Authoritative Config Loader:** Built a strict runtime validation subsystem using **Pydantic V2** to parse hardware thresholds from `config.yaml`.
* **Decoupled Event Bus:** Implemented a thread-safe **Observer Pattern** so sensors can talk to the brain without messy, tangled code dependencies.
* **Premium Mission Control UI:** Developed a high-end sci-fi dark dashboard using **Streamlit** with fully functional telemetry simulation matrix.
* **Structured Logging:** Configured centralized JSON logging to track the system's internal decisions second-by-second.

### ⏳ Phase 2: Active Signal Extraction & Machine Brain (IN PROGRESS - CURRENT FOCUS)
* [ ] **Acoustic Stress Layer:** Coding microsecond-level raw audio stream feature extraction (13 MFCCs, Delta, Delta-Delta, Spectral Rolloff, ZCR, and RMS Energy) using pure NumPy/SciPy.
* [ ] **Motion Telemetry Layer:** Building a deterministic algorithmic state machine to catch a specific physical sequence: *High-G Impact Spike* followed strictly by *Prolonged Structural Stillness*.
* [ ] **Calibrated AI Model:** Training and calibrating a local Random Forest Classifier (`sentinel_brain.pkl`) for high-accuracy local threat classification.

### 🚀 Phase 3: Advanced Sensor Fusion & Deployment (FUTURE PIPELINE)
* [ ] **Bayesian Fusion Engine:** Implementing a Weighted Probabilistic Engine that uses normalized entropy to calculate an absolute threat score.
* [ ] **Smart Veto Subsystem:** Writing explicit override rules to kill false alarms (e.g., if the phone drops but the acoustic layer detects no voice distress, the alert is suppressed).
* [ ] **Dead Man's Switch FSM:** Integrating a non-blocking 10-second interactive countdown loop that prompts the user before auto-dispatching emergency logs.

---

## 📂 Repository Blueprint (Clean Src-Layout)

This codebase follows industry-standard enterprise patterns:

silent_sentinel/
├── config/
│   └── config.yaml          # Hyperparameters & sensor thresholds
├── src/
│   ├── config_loader.py     # Pydantic V2 configuration validator
│   ├── core/
│   │   ├── event_bus.py     # Decoupled messaging hub (Observer Pattern)
│   │   ├── events.py        # Strongly-typed dataclass communication events
│   │   ├── fsm.py           # Dead Man's Switch finite state controller
│   │   └── fusion.py        # Bayesian intelligence & smart veto nodes
│   ├── sensors/
│   │   ├── acoustic.py      # Microsecond audio feature extraction core
│   │   └── motion.py        # Accelerometer pattern sequence tracker
│   └── ui/
│       └── app.py           # Sci-Fi Mission Control Dashboard (Streamlit)

---

## ⚡ Quickstart, Deployment & Execution

Want to boot up my current active dashboard? Run this sequence of commands inside your system terminal:

# 1. Access the workspace directory
cd silent_sentinel

# 2. Initialize the isolated virtual sandbox environment
python -m venv venv

# 3. Activate the sandbox
# On Windows (Command Prompt / PowerShell):
venv\Scripts\activate
# On Mac / Linux:
# source venv/bin/activate

# 4. Install authoritative dependencies
pip install numpy scipy pydantic pyyaml streamlit

# 5. Boot the live Sci-Fi Mission Control interface
streamlit run src/ui/app.py

---

## 🤝 Let's Connect!
I am building this platform to bridge the gap between complex signal mathematics and real-world personal safety. If you find this project interesting, feel free to drop a star 🌟 on this repository or follow my development updates!