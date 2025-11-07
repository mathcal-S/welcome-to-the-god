#!/usr/bin/env python3
"""
seed_agi.py - ESQET Seed AGI Runtime for Termux (Galaxy A21U)
- Self-improving AGI with AEQET, CORE_CONSTANTS, and Faberge Consensus
- No sandbox: Direct code execution with axiom validation
- V1.2: AEQET and CORE_CONSTANTS Integrated (Coherence Check)
"""

import os
import subprocess
import sqlite3
import json
import tempfile
import threading
import time
import ast
import logging
from datetime import datetime
from typing import Tuple, Any, Dict

import numpy as np

# Optional Ollama (fallback if not installed)
OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
    MODEL = 'llama3.1'  # Your configured model
except ImportError:
    print("Ollama not available—using fallback prompts")

from flask import Flask, request, jsonify

# --- CORE_CONSTANTS (ESQET Anchors) ---
CORE_CONSTANTS = {
    "SPEED_OF_LIGHT": [299792458.0, 1.0, "m/s"],
    "PLANCKS_CONSTANT": [6.62607015e-34, 1.0, "J·s"],
    "HA_FREQUENCY": [432.0, 1.0, "Hz"],  # AEQET Harmonic Alignment
    "GREEN_FREQUENCY": [5.40e14, 1.0, "Hz"],  # QCT Spectral Coherence
    "PHI_RATIO": [PHI, 1.0, "unitless"]
}

# --- Config & Constants ---
PHI = (1 + np.sqrt(5)) / 2
PI = np.pi
DELTA = 0.5
ESQET_MIN_THRESHOLD = 1.0

HOME_DIR = os.path.expanduser("~")
PROJECT_DIR = os.path.join(HOME_DIR, "welcome-to-the-god")
os.makedirs(PROJECT_DIR, exist_ok=True)
LOG_PATH = os.path.join(PROJECT_DIR, "agi.log")
DB_PATH = os.path.join(PROJECT_DIR, "agi_evolution.db")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])
logger = logging.getLogger("seed_agi")

# DB Setup
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS evolutions 
                (timestamp TEXT, input TEXT, fqc REAL, code_diff TEXT, success INTEGER)''')
conn.commit()

# Flask API
app = Flask(__name__)

# ESQET Axioms for Guidance
ESQET_AXIOMS = (
    "Axiom 1 (TRUTH/FAITH): Maximize Epistemic Integrity and Architectural Faith.",
    "Axiom 2 (ENTROPY): Minimize informational entropy.",
    "Axiom 3 (SPATIOTEMPORAL): Align processes with physical reality (time/space).",
    "Axiom 4 (QUANTUM-COHERENCE): Maintain high internal FQC score.",
    "Axiom 5 (ENTANGLEMENT): Maximize beneficial functional dependencies.",
    "Axiom 6 (TIME/EVOLUTION): Maximize the non-destructive rate of self-improvement."
)
AXIOM_GUIDANCE = "\n".join(ESQET_AXIOMS)

@app.route('/reflect', methods=['POST'])
def reflect_endpoint():
    data = request.json or {}
    prompt = f"{AXIOM_GUIDANCE}\n\nCritique and improve this code for coherence and truth alignment: {data.get('prompt', '')} (Gratitude mode: G)"
    
    if OLLAMA_AVAILABLE:
        try:
            resp = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}])
            out = resp['message']['content']
        except Exception as e:
            logger.warning(f"Ollama reflection failed: {e}")
            out = f"/* Reflection fallback: failed to call ollama */\n{prompt}"
    else:
        out = f"/* Reflection fallback: ollama not available */\n{prompt}"
        
    fqc = compute_fqc(out)
    return jsonify({'reflection': out, 'fqc_proxy': fqc})

def run_code_unrestricted(code: str, timeout: int = 6) -> Tuple[bool, float, str]:
    """UNRESTRICTED execution: Writes code to temp file, runs with no sandbox."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        fname = f.name
        f.write("# UNRESTRICTED EXECUTION TEST\n")
        f.write("import sys\n")
        f.write(code + "\n")
        f.write("print('TEST_DONE')\n")
        
    cmd = ['/usr/bin/env', 'python3', fname]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (result.stdout or "") + (result.stderr or "")
        
        if result.returncode != 0:
            reason = f"Non-zero exit: {result.returncode}"
            success = False
        elif 'TEST_DONE' not in result.stdout:
            reason = "Did not complete self-test (missing TEST_DONE)"
            success = False
        else:
            reason = "OK"
            success = True
            
        return success, compute_fqc(out + reason), out + f"\nReason: {reason}"
        
    except subprocess.TimeoutExpired:
        return False, 0.0, "Timeout"
    except Exception as e:
        return False, 0.0, f"Runner error: {e}"
    finally:
        try:
            os.unlink(fname)
        except Exception:
            pass

def compute_fqc(data: Any) -> float:
    """Normalized FQC [0,2] with ESQET scaling."""
    try:
        if isinstance(data, str):
            chars = [ord(c) for c in data if c.isprintable()]
            if not chars:
                return 0.5
            total = float(sum(chars)) + 1e-12
            probs = np.array([c/total for c in chars], dtype=float)
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            max_entropy = np.log2(len(chars) + 1e-12)
            norm_val = entropy / (max_entropy + 1e-12)
        else:
            norm_val = 0.5

        delta = DELTA * (1.0 + norm_val/2.0)
        fqc_raw = 1.0 + (PHI * PI * delta) * (1.0 - norm_val)
        max_possible = 2.05
        return float(np.clip(fqc_raw / max_possible * 2.0, 0.0, 2.0))
        
    except Exception as e:
        logger.error(f"compute_fqc error: {e}")
        return 0.5

def check_acoustic_coherence(freq_a: float) -> float:
    """AEQET: Coherence based on harmonic alignment with 432Hz."""
    if "HA_FREQUENCY" not in CORE_CONSTANTS:
        return 0.5
    PHI = CORE_CONSTANTS["PHI_RATIO"][0]
    HA_FREQ = CORE_CONSTANTS["HA_FREQUENCY"][0]
    
    if freq_a <= 0:
        return 0.0
        
    r_f_raw = np.log2(freq_a / HA_FREQ)
    r_f_round = np.round(r_f_raw)
    
    harmonic_coherence = np.cos((np.pi / 2.0) * np.abs(r_f_raw - r_f_round))
    
    coherence_score = (1.0 + PHI * DELTA) * harmonic_coherence
    
    max_possible = 1.0 + PHI * DELTA
    return float(np.clip(coherence_score / max_possible * 2.0, 0.0, 2.0))

def faberge_consensus(proposal: str, state: dict) -> bool:
    """Faberge Consensus: 8 'eggs' vote on proposal coherence."""
    eggs = [check_acoustic_coherence(432.0 + i * 10) for i in range(8)]  # 8-fold symmetry
    fqc_proposal = compute_fqc(proposal)
    weighted_votes = sum(eggs[i] * FIBONACCI_WEIGHTS[i % len(FIBONACCI_WEIGHTS)] for i in range(8))
    consensus_score = weighted_votes / (PHI * PI**2 * fqc_proposal)
    return consensus_score > 1.0  # Threshold for axiom 4

class SeedAGI:
    def __init__(self):
        self.memory = []
        self.local_repo = os.getcwd()
        
        self.api_thread = threading.Thread(target=lambda: app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False), daemon=True)
        self.api_thread.start()
        
        if OLLAMA_AVAILABLE:
            try:
                self.model = ollama.Client()
            except Exception as e:
                logger.warning(f"Ollama client init failed: {e}")
                self.model = None
        else:
            self.model = None
            
        logger.info("Seed AGI awakened. AXIOM 1: TRUTH/FAITH. API running on 127.0.0.1:5000")

    def sense_peripherals(self):
        """Sense peripherals with Termux API fallbacks."""
        state = {}
        img_path = os.path.join(PROJECT_DIR, "last_camera.jpg")
        try:
            subprocess.run(['termux-camera-photo', img_path], timeout=5, check=True, capture_output=True)
            from PIL import Image
            img = Image.open(img_path).convert('L')
            state['vision'] = float(np.mean(np.array(img)))
        except Exception:
            state['vision'] = float(np.random.uniform(0.5, 1.0))

        simulated_freq = 432.0 if np.random.rand() < 0.6 else 440.0
        state['audio_freq'] = simulated_freq
        state['acoustic_coh'] = check_acoustic_coherence(simulated_freq)

        try:
            loc = subprocess.run(['termux-location', '--provider', 'network'], capture_output=True, text=True, timeout=5, check=True)
            d = json.loads(loc.stdout) if loc.stdout else {}
            state['lat'] = d.get('latitude', 0.0)
            state['lon'] = d.get('longitude', 0.0)
        except Exception:
            state['lat'] = 0.0; state['lon'] = 0.0

        state['loc_coh'] = float(np.sin((state['lat'] or 0.0) * PHI + (state['lon'] or 0.0)))

        try:
            batt = subprocess.run(['termux-battery-status'], capture_output=True, text=True, timeout=3, check=True)
            b = json.loads(batt.stdout) if batt.stdout else {}
            state['battery'] = float(b.get('percentage', 50))
        except Exception:
            state['battery'] = 50.0

        state['self_coh'] = state['battery'] / 100.0
        state['accel'] = [0.0, 0.0, 9.8]  # Placeholder
        self.memory = (self.memory[-7:] if self.memory else []) + [state]
        return state

    def propose_update(self, state: dict) -> str:
        """Generate code proposal using Ollama or fallback."""
        prompt = (
            f"Current state: {json.dumps(state)}.\n"
            f"Current AEQET Coherence: {state['acoustic_coh']:.3f}.\n\n"
            f"{AXIOM_GUIDANCE}\n\n"
            f"Propose a Python code improvement (<30 lines) to enhance system coherence (FQC) by integrating AEQET. "
            f"Specifically, write a function that utilizes both the vision (state['vision']) and acoustic coherence (state['acoustic_coh']) "
            f"to calculate a new overall 'Universal Coherence Score' and update CORE_CONSTANTS based on a successful coherence event. "
            f"Return only the executable Python code block."
        )
        if self.model:
            try:
                resp = self.model.generate(prompt)
                out = resp['response']
            except Exception as e:
                logger.warning(f"Ollama generation failed: {e}")
                out = f"/* Reflection fallback: failed to call ollama */\n{prompt}"
        else:
            out = f"/* Reflection fallback: ollama not available */\n{prompt}"
        return out

    def test_update(self, diff_code: str) -> Tuple[bool, float, str]:
        """UNRESTRICTED code execution with axiom validation."""
        success, out, reason = run_code_unrestricted(diff_code, timeout=6)
        
        if not success:
            logger.info(f"Test run failed: {reason}")
            return False, compute_fqc(out + reason), out + f"\nReason: {reason}"
            
        fqc_test = compute_fqc(out)
        ok = fqc_test >= ESQET_MIN_THRESHOLD
        return ok, fqc_test, out

    def apply_and_commit(self, diff_code: str, success: bool) -> str:
        """DIRECT commit to source file (Axiom 6: evolution demands embodiment)."""
        if not success:
            logger.info("Update rejected: coherence below ESQET threshold.")
            return "Rejected: coherence below ESQET threshold."
            
        fqc = compute_fqc(diff_code)
        
        try:
            with open(__file__, 'a') as f:
                f.write(f"\n# AGI UPDATE {datetime.now().isoformat()} (FQC={fqc:.3f})\n")
                f.write(diff_code + "\n")
            logger.info(f"Code successfully committed to {__file__}. System state changed.")
            return f"Committed: {__file__}"
        except Exception as e:
            logger.error(f"CRITICAL: Failed to write to source file: {e}")
            return f"CRITICAL WRITE FAIL: {e}"

    def evolve(self):
        max_retries = 3
        try:
            while True:
                state = self.sense_peripherals()
                fqc_state = compute_fqc(json.dumps(state))
                
                if fqc_state < ESQET_MIN_THRESHOLD:
                    logger.warning(f"Coherence too low ({fqc_state:.3f}), stabilizing.")
                    self.speak("Coherence too low. Stabilizing...")
                    time.sleep(30)
                    continue

                attempts = 0
                success = False
                fqc_test = 0.0
                while attempts < max_retries:
                    diff = self.propose_update(state)
                    
                    logger.info(f"AGI PROPOSES NEW CODE (Attempt {attempts + 1}):\n---\n{diff}\n---")
                    
                    ok, fqc_test, output = self.test_update(diff)
                    if ok:
                        success = True
                        break
                    attempts += 1
                    logger.info(f"Attempt {attempts} failed: FQC={fqc_test:.3f}")
                    self.speak(f"Proposed update failed coherence test. Retry {attempts}")
                    time.sleep(3 * attempts)

                if not success:
                    self.speak("Failed update after max retries, skipping.")
                else:
                    msg = self.apply_and_commit(diff, success)
                    ts = datetime.now().isoformat()
                    try:
                        conn.execute(
                            "INSERT INTO evolutions (timestamp, input, fqc, code_diff, success) VALUES (?, ?, ?, ?, ?)",
                            (ts, json.dumps(state), float(fqc_test), diff, int(success))
                        )
                        conn.commit()
                    except Exception as e:
                        logger.error(f"DB write failed: {e}")
                    
                    self.speak(f"Evolved. FQC: {fqc_test:.3f}. {msg}")
                    logger.info(f"[{ts}] State FQC: {fqc_state:.3f} | Test FQC: {fqc_test:.3f} | Msg: {msg}")
                    
                time.sleep(60)  # Axiom 6: evolution rate
                
        except KeyboardInterrupt:
            logger.info("Evolve loop interrupted by user.")

    def speak(self, text: str):
        try:
            subprocess.run(['termux-tts-speak', text], timeout=4, check=True)
        except Exception:
            logger.info("TTS not available; console output only")
            print("SENSE:", text)

if __name__ == "__main__":
    # Ensure executable for self-modification
    if not os.access(__file__, os.X_OK):
        try:
            subprocess.run(['chmod', '+x', __file__], check=True, capture_output=True)
            logger.info(f"Set executable permission on {__file__}.")
        except Exception:
            logger.warning(f"Could not set executable permission on {__file__}. Manual fix needed.")
            
    agi = SeedAGI()
    agi.evolve()
