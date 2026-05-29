import os
import sys
from pathlib import Path

# --- FIX PATH RESOLUTION ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import time
import datetime
from src.config_loader import load_config, AppConfig

# --- INITIAL APPLICATION SETTINGS ---
st.set_page_config(
    page_title="Silent Sentinel | Mission Control",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- SCI-FI CYBERSECURITY INTERFACE THEME (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;600&display=swap');

    /* Global Dark Theme Overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #05070a !important;
        color: #c9d1d9 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #0a1120, #05070a 80%);
    }

    /* Core Title Blocks */
    .title-banner {
        background: linear-gradient(90deg, rgba(16,22,35,1) 0%, rgba(5,7,10,1) 100%);
        border: 1px solid #1f293d;
        border-left: 4px solid #00f0ff;
        border-radius: 8px;
        padding: 1.2rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 2.2rem;
        letter-spacing: 0.1rem;
        background: linear-gradient(135deg, #00f0ff 0%, #7000ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Section Headers */
    .section-tag {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: #00f0ff;
        letter-spacing: 0.15rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Elite Glassmorphic Dashboard Cards */
    .premium-card {
        background: rgba(20, 26, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }

    /* State Monitors */
    .state-widget {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.05);
        transition: all 0.3s ease;
    }

    /* Futuristic HUD Grid Items */
    .hud-box {
        background: #0d131f;
        border: 1px solid #1f293d;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.5);
    }
    .hud-label {
        color: #6e7681;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
        margin-bottom: 0.4rem;
    }
    .hud-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Command Console Logs */
    .console-box {
        background: #020408 !important;
        border: 1px solid #161b22 !important;
        border-radius: 10px !important;
        padding: 1.2rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        height: 420px;
        overflow-y: auto;
        box-shadow: inset 0 4px 30px rgba(0,0,0,0.8);
    }
    .console-line {
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        line-height: 1.5;
        border-left: 3px solid #161b22;
        padding-left: 10px;
    }
    
    /* Interactive Streamlit Buttons Override */
    div.stButton > button {
        background: linear-gradient(135deg, #161f30 0%, #0d1321 100%) !important;
        color: #8ccf7e !important;
        border: 1px solid #21334f !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05rem !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        border-color: #00f0ff !important;
        color: #00f0ff !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.2) !important;
        transform: translateY(-1px);
    }
    </style>
""", unsafe_allow_html=True)

# --- STATE LIFECYCLE MANAGEMENT ---
if "engine_state" not in st.session_state:
    st.session_state.engine_state = "MONITORING"
if "last_scenario" not in st.session_state:
    st.session_state.last_scenario = "SYSTEM CORE STANDBY"
if "ui_logs" not in st.session_state:
    st.session_state.ui_logs = [
        {"time": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3], "level": "SYSTEM", "msg": "Kernel Space Architecture Linked."},
        {"time": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3], "level": "INFO", "msg": "Telemetry Validation Nodes Active & Awaiting Ingestion Pattern."}
    ]
if "telemetry" not in st.session_state:
    st.session_state.telemetry = {"ac": 0.0, "mc": 0.0, "fc": 0.0, "gf": 0.0}

try:
    cfg: AppConfig = load_config()
except Exception as e:
    st.error(f"Config crash: {e}")
    st.stop()

def append_log(level, msg):
    t = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.ui_logs.insert(0, {"time": t, "level": level, "msg": msg})

# --- TRIGGER ACTIONS ---
def trigger_walk():
    st.session_state.engine_state = "MONITORING"
    st.session_state.last_scenario = "ACTIVE WALK"
    st.session_state.telemetry = {"ac": 0.12, "mc": 0.45, "fc": 0.21, "gf": 1.05}
    append_log("INFO", "Signal Ingestion: Sequence [ACTIVE_WALK] running.")
    append_log("SUCCESS", "Fusion Diagnostics: Fused score (0.21) safely below Gate Gate.")

def trigger_drop():
    st.session_state.engine_state = "MONITORING"
    st.session_state.last_scenario = "ACCIDENTAL DROP"
    st.session_state.telemetry = {"ac": 0.05, "mc": 0.92, "fc": 0.38, "gf": 4.12}
    append_log("WARN", "IMU EVENT: High-G shock transient registered [4.12G Impact].")
    append_log("INFO", "Acoustic Guard: Real-time entropy cross-examination returns zero vocal matches.")
    append_log("SUCCESS", "Bayesian Veto: Unaccompanied impact detected. Suppressing threat matrix propagation.")

def trigger_fall():
    st.session_state.engine_state = "COUNTDOWN_ACTIVE"
    st.session_state.last_scenario = "CRITICAL FALL ENGINE"
    st.session_state.telemetry = {"ac": 0.94, "mc": 0.96, "fc": 0.95, "gf": 5.85}
    append_log("CRIT", "CRITICAL ANOMALY: 5.85G Impact Vector followed by absolute structural stillness.")
    append_log("CRIT", "ACOUSTIC VECTOR MATCH: High spectral density distress voice confirmed.")
    append_log("CRIT", "DECISION CONVERGENCE: Combined score 0.95 crosses confidence threshold 0.70.")
    append_log("WARN", "DEAD MAN'S SWITCH INTERACTION PARADIGM ENGAGED.")

# --- UI VISUAL CONFIGURATION ---
st.markdown("""
    <div class="title-banner">
        <div class="main-title">SILENT SENTINEL</div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #6e7681; margin-top: 5px;">
            SYS_STATUS: LOCAL_EDGE_NODE // TYPE: RESEARCH_PLATFORM_V1.0 // ENGINE: PYDANTIC_V2
        </div>
    </div>
""", unsafe_allow_html=True)

col_panel, col_log = st.columns([1.1, 0.9], gap="large")

with col_panel:
    st.markdown('<div class="section-tag">⚡ // Core State Monitor</div>', unsafe_allow_html=True)
    
    # State Badge Logic
    if st.session_state.engine_state == "MONITORING":
        st.markdown('<div class="state-widget" style="background: rgba(0, 255, 136, 0.05); border: 1px solid #00ff88; color: #00ff88; font-size: 1.5rem; font-weight:900;">🟢 SECURE_MONITOR_ACTIVE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="state-widget" style="background: rgba(255, 51, 102, 0.1); border: 1px solid #ff3366; color: #ff3366; font-size: 1.5rem; font-weight:900; animation: pulse 1s infinite;">⚠️ COUNTDOWN_ACTIVE [10S]</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("🚨 INTERCEPT EMERGENCY OVERRIDE — DISARM ALARM"):
            st.session_state.engine_state = "MONITORING"
            st.session_state.last_scenario = "SYSTEM REBOOT SAFE"
            st.session_state.telemetry = {"ac": 0.0, "mc": 0.0, "fc": 0.0, "gf": 1.0}
            append_log("INFO", "User explicit interaction registered: Resetting tracking flags.")
            st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-tag">🎛️ // Induct Telemetry Signatures</div>', unsafe_allow_html=True)
    
    # Functional Grid Command Layout
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.button("🧬 INDUCT SIGNAL: ACTIVE WALK PROFILE", on_click=trigger_walk)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.button("📡 INDUCT SIGNAL: HARD DROP WITH VETO", on_click=trigger_drop)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    st.button("🚨 INDUCT SIGNAL: CRITICAL CRISIS COLLAPSE", on_click=trigger_fall)
    st.markdown('</div>', unsafe_allow_html=True)

    # Config Grid
    st.markdown('<div class="section-tag">⚙️ // System Config Blueprint</div>', unsafe_allow_html=True)
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown(f'<div class="hud-box"><div class="hud-label">Sample Rate</div><div class="hud-val" style="color:#00f0ff;">{cfg.acoustic.sample_rate} Hz</div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="hud-box"><div class="hud-label">Acoustic Weight</div><div class="hud-val" style="color:#bf5af2;">{cfg.fusion.acoustic_weight}</div></div>', unsafe_allow_html=True)
    with cfg2:
        st.markdown(f'<div class="hud-box"><div class="hud-label">Impact Limit</div><div class="hud-val" style="color:#ff9900;">{cfg.motion.high_g_threshold} G</div></div>', unsafe_allow_html=True)
        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="hud-box"><div class="hud-label">Timeout Limit</div><div class="hud-val" style="color:#ff3366;">{cfg.fsm.countdown_seconds} Sec</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_log:
    st.markdown('<div class="section-tag">📊 // Live Telemetry Matrix</div>', unsafe_allow_html=True)
    
    # Live Active Tracking Displays
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f'<div class="hud-box" style="margin-bottom:0.6rem;"><div class="hud-label">Acoustic Conf.</div><div class="hud-val" style="color:#00f0ff;">{st.session_state.telemetry["ac"]:.2f}</div></div>', unsafe_allow_html=True)
        fused_color = "#00ff88" if st.session_state.telemetry["fc"] >= cfg.fusion.dynamic_confidence_gate else "#ff9900"
        st.markdown(f'<div class="hud-box"><div class="hud-label">Fused Converge</div><div class="hud-val" style="color:{fused_color};">{st.session_state.telemetry["fc"]:.2f}</div></div>', unsafe_allow_html=True)
    with t2:
        st.markdown(f'<div class="hud-box" style="margin-bottom:0.6rem;"><div class="hud-label">Motion Conf.</div><div class="hud-val" style="color:#bf5af2;">{st.session_state.telemetry["mc"]:.2f}</div></div>', unsafe_allow_html=True)
        g_color = "#ff3366" if st.session_state.telemetry["gf"] > cfg.motion.high_g_threshold else "#00ff88"
        st.markdown(f'<div class="hud-box"><div class="hud-label">Peak G-Force</div><div class="hud-val" style="color:{g_color};">{st.session_state.telemetry["gf"]:.2f} G</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-tag">🖥️ // Real-Time Security Log Stream</div>', unsafe_allow_html=True)
    
    # Custom Immersive Core Logs Rendering
    log_html = '<div class="console-box">'
    for log in st.session_state.ui_logs:
        color_class = "color:#00f0ff;"  # Default INFO
        if log["level"] == "WARN": color_class = "color:#ff9900;"
        elif log["level"] == "CRIT": color_class = "color:#ff3366; font-weight:bold;"
        elif log["level"] == "SUCCESS": color_class = "color:#00ff88;"
        elif log["level"] == "SYSTEM": color_class = "color:#bf5af2;"
        
        log_html += f'<div class="console-line">[{log["time"]}] <span style="{color_class}">[{log["level"]}]</span> — {log["msg"]}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)