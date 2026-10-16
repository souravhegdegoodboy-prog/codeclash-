from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random
import datetime

app = Flask(__name__, static_folder='.')
CORS(app)

# Central Campus State
campus_state = {
    "classroom_101": {
        "room_id": "Classroom 101",
        "capacity": 60,
        "current_occupancy": 0,
        "energy_consumption_kw": 2.4,
        "ac_status": "ON",
        "last_updated": "Waiting for sensor..."
    },
    "lab_3": {
        "room_id": "AI Lab 3",
        "water_flow_lpm": 0.5,
        "leak_detected": False,
        "bin_fill_level_pct": 45
    }
}

@app.route('/')
def serve_dashboard():
    return send_from_directory('.', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_campus_status():
    # Simulate dynamic energy adjustments based on live load
    occ = campus_state["classroom_101"]["current_occupancy"]
    campus_state["classroom_101"]["energy_consumption_kw"] = round(1.5 + (occ * 0.03), 2)
    
    return jsonify({
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": campus_state
    })

@app.route('/api/update-occupancy', methods=['POST'])
def update_occupancy():
    data = request.json
    occupancy = data.get("occupancy", 0)
    campus_state["classroom_101"]["current_occupancy"] = occupancy
    campus_state["classroom_101"]["last_updated"] = datetime.datetime.now().strftime("%H:%M:%S")
    return jsonify({"status": "updated", "occupancy": occupancy})

@app.route('/api/predict', methods=['GET'])
def get_ai_prediction():
    current_occ = campus_state["classroom_101"]["current_occupancy"]
    
    # GenAI behavioral insight simulation
    if current_occ > 5:
        insight = f"High occupancy ({current_occ} students) detected in Classroom 101. HVAC workload is elevated; recommend optimizing temperature setpoint by +2°C to conserve energy."
        trend = "Peak load expected for the next 45 minutes."
    else:
        insight = "Low occupancy window detected. Consider dimming smart LED arrays and throttling secondary ventilation zones in Classroom 101."
        trend = "Stable low-usage energy conservation mode."

    return jsonify({
        "status": "success",
        "ai_insights": insight,
        "trend": trend
    })

if __name__ == '__main__':
    print("[INFO] Starting EcoSmart Campus Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)