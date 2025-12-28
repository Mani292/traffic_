from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import numpy as np
from datetime import datetime
import math

app = FastAPI(title="Bengaluru Traffic Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= LOAD LSTM MODEL =================
model = load_model("lstm_model.h5")

def lstm_predict(volume, speed, hour):
    sample = np.array([[volume, speed, hour]] * 10).reshape(1,10,3)
    pred = model.predict(sample)[0]
    return ["Low","Medium","High"][np.argmax(pred)]

# ================= DATA MODELS =================
class Location(BaseModel):
    lat: float
    lng: float
    name: str

class RouteRequest(BaseModel):
    origin: Location
    destination: Location
    time_of_day: str = "now"

# ================= UTILS =================
def haversine(a,b,c,d):
    R=6371
    a,b,c,d=map(math.radians,[a,b,c,d])
    return R*2*math.asin(math.sqrt(
        math.sin((c-a)/2)**2+math.cos(a)*math.cos(c)*math.sin((d-b)/2)**2
    ))

# ================= API =================
@app.get("/health")
def health():
    return {"status":"healthy"}

@app.post("/api/routes")
def get_routes(req: RouteRequest):
    volume=np.random.randint(40,120)
    speed=np.random.randint(20,50)
    hour=datetime.now().hour

    level=lstm_predict(volume,speed,hour)

    return [{
        "route_id":1,
        "name":"Primary Route",
        "distance":round(haversine(req.origin.lat,req.origin.lng,
                                  req.destination.lat,req.destination.lng),2),
        "duration":round(60/(speed+1),1),
        "traffic_level":level.lower(),
        "traffic_score":round(np.random.uniform(0.3,0.9),2),
        "coordinates":[
            [req.origin.lat,req.origin.lng],
            [req.destination.lat,req.destination.lng]
        ],
        "is_fastest": level!="High",
        "predicted_speed":speed,
        "services":{"hospitals":2,"fuel_stations":2,"restaurants":3}
    }]

@app.get("/api/services")
def services():
    return {
        "hospitals":[{"name":"Apollo","lat":12.91,"lng":77.62}],
        "fuel":[{"name":"HP","lat":12.92,"lng":77.63}],
        "restaurants":[{"name":"Empire","lat":12.93,"lng":77.64}]
    }

@app.get("/api/incidents")
def incidents():
    return [{"type":"Accident","lat":12.9165,"lng":77.6230}]

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)
