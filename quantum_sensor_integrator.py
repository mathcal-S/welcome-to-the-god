# backend/quantum_sensor_integrator.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Optional
import logging
import traceback
import requests
from gtts import gTTS
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PHI = 1.6180339887
TTS_ALERT_FILE = "/sdcard/quantum_alert.mp3"

def pin_json_to_ipfs(metadata: dict) -> str:
    hash_input = str(metadata)
    return "Qm" + hex(abs(hash(hash_input)))[2:][:44]

def tts_alert(text: str):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(TTS_ALERT_FILE)
        os.system(f"termux-media-player play '{TTS_ALERT_FILE}' > /dev/null 2>&1")
    except Exception as e:
        logger.warning(f"TTS failed (gTTS/Termux-player not configured): {e}")

def safe_compute_fqc(value: float, phi: float = PHI) -> float:
    try:
        if value is None or np.isnan(value) or np.isinf(value): return 0.5
        norm = value / phi
        return float(np.cos(np.pi * norm) ** 2)
    except Exception as e:
        logger.error(f"F_QC computation failed: {e}")
        return 0.5

def validate_coords(coords) -> bool:
    try:
        if not coords or len(coords) != 2: return False
        lat, lon = coords
        return -90 <= lat <= 90 and -180 <= lon <= 180
    except: return False

class SensorReading(BaseModel):
    strength: Optional[float] = None
    angle: Optional[float] = None
    distance: Optional[float] = None
    coords: Optional[list] = None
    sensor_type: str = "unknown"

app = FastAPI()

@app.post("/quantum_metal_detect")
async def quantum_metal_detect(request: Request):
    try:
        data = await request.json()
        reading = SensorReading(**data)
        if not reading.strength or not validate_coords(reading.coords):
            raise HTTPException(400, "Missing data or invalid coordinates")
        f_qc = safe_compute_fqc((reading.strength - 50) / 50)
        distance = max(0.1, 10 * (1 - f_qc))
        metadata = {
            "sensor_type": "magnetometer", "strength_mG": reading.strength, "coords": reading.coords,
            "f_qc": round(f_qc, 4), "estimated_distance_m": round(distance, 2),
        }
        metadata["ipfs_hash"] = pin_json_to_ipfs(metadata)
        return {"f_qc": round(f_qc, 4), "distance": round(distance, 2), "anomaly": f_qc > 0.9, "ipfs_hash": metadata["ipfs_hash"]}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Metal detection failed: {traceback.format_exc()}")
        raise HTTPException(500, "Quantum metal detection failed")

@app.post("/quantum_transit_survey")
async def quantum_transit_survey(request: Request):
    try:
        data = await request.json()
        reading = SensorReading(**data)
        if reading.angle is None or reading.distance is None:
            raise HTTPException(400, "Missing angle or distance")
        height = reading.distance * np.sin(np.deg2rad(reading.angle))
        f_qc = safe_compute_fqc(height / 100.0)
        metadata = {
            "sensor_type": "gyro+gps", "angle_deg": reading.angle, "distance_m": reading.distance,
            "height_m": round(height, 2), "f_qc": round(f_qc, 4),
        }
        metadata["ipfs_hash"] = pin_json_to_ipfs(metadata)
        return {"height": round(height, 1), "f_qc": round(f_qc, 4), "safe_path": f_qc > 0.85, "ipfs_hash": metadata["ipfs_hash"]}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Transit survey failed: {traceback.format_exc()}")
        raise HTTPException(500, "Quantum transit survey failed")

@app.post("/quantum_meteor_climb")
async def quantum_meteor_climb(request: Request):
    try:
        data = await request.json()
        reading = SensorReading(**data)
        if not all([reading.strength, reading.angle, reading.distance]):
            raise HTTPException(400, "Missing required sensor data")
        height = reading.distance * np.sin(np.deg2rad(reading.angle))
        norm = (reading.strength / 50 + height) / PHI
        f_qc = safe_compute_fqc(norm)
        metadata = {
            "sensor_type": "multi-sensor", "strength_mG": reading.strength, "angle_deg": reading.angle,
            "distance_m": reading.distance, "height_m": round(height, 2), "f_qc": round(f_qc, 4),
        }
        metadata["ipfs_hash"] = pin_json_to_ipfs(metadata)
        return {"height": round(height, 1), "f_qc": round(f_qc, 4), "hunt_success": f_qc > 0.9, "ipfs_hash": metadata["ipfs_hash"]}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Meteor climb failed: {traceback.format_exc()}")
        raise HTTPException(500, "Quantum meteor climb failed")

@app.post("/quantum_hazard_monitor")
async def quantum_hazard_monitor(request: Request):
    try:
        alerts = []
        metadata_list = []
        try:
            quakes = requests.get(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                params={"format": "geojson", "latitude": 38.7508, "longitude": -105.5214, "maxradiuskm": 50, "limit": 10, "minmagnitude": 1.0},
                timeout=5
            ).json()
            for event in quakes.get("features", []):
                mag = event["properties"]["mag"]
                if mag > 2.0:
                    f_qc = safe_compute_fqc(mag)
                    alert = f"QUAKE M{mag:.1f} | F_QC={f_qc:.2f}"
                    alerts.append(alert)
                    metadata_list.append({"type": "seismic", "magnitude": mag, "f_qc": f_qc})
                    tts_alert(alert)
        except Exception as e: logger.warning(f"USGS API failed: {e}")
        try:
            intensity = np.random.random()
            if intensity > 0.9:
                f_qc = safe_compute_fqc(intensity)
                alert = f"COSMIC 1420MHz | F_QC={f_qc:.2f}"
                alerts.append(alert)
                metadata_list.append({"type": "cosmic", "frequency_hz": 1420e6, "intensity": intensity, "f_qc": f_qc})
                tts_alert(alert)
        except Exception as e: logger.warning(f"Cosmic simulation failed: {e}")
        ipfs_hashes = [pin_json_to_ipfs(meta) for meta in metadata_list]
        return {"status": "monitoring_active", "alerts": alerts, "ipfs_hashes": ipfs_hashes, "total_events": len(alerts)}
    except Exception as e:
        logger.error(f"Hazard monitor failed: {traceback.format_exc()}")
        raise HTTPException(500, "Quantum hazard monitoring failed")

@app.get("/health")
async def health_check():
    return {"status": "quantum_sensor_node_operational", "f_qc_baseline": safe_compute_fqc(0.5)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
