# main.py - FastAPI Backend for Traffic Prediction System
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from datetime import datetime
import math

app = FastAPI(
    title="Bengaluru Traffic Intelligence API",
    description="AI-powered traffic prediction and routing system",
    version="1.0.0"
)

# CORS middleware - Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DATA MODELS ====================
class Location(BaseModel):
    lat: float
    lng: float
    name: str

class RouteRequest(BaseModel):
    origin: Location
    destination: Location
    time_of_day: str = "now"

class RouteResponse(BaseModel):
    route_id: int
    name: str
    distance: float
    duration: float
    traffic_level: str
    traffic_score: float
    coordinates: List[List[float]]
    is_fastest: bool
    predicted_speed: float
    services: dict

class ServiceLocation(BaseModel):
    type: str
    name: str
    lat: float
    lng: float
    distance_from_route: float

class Incident(BaseModel):
    id: int
    type: str
    location: str
    severity: str
    lat: float
    lng: float
    description: str

# ==================== HELPER FUNCTIONS ====================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def predict_traffic_lstm(route_coords, time_of_day, route_index=0):
    """
    Simulate LSTM traffic prediction
    In production, this would call your trained LSTM model
    """
    # Time-based traffic patterns
    time_factors = {
        "morning": [0.75, 0.65, 0.85],    # High congestion
        "afternoon": [0.55, 0.45, 0.65],  # Medium congestion
        "evening": [0.85, 0.75, 0.90],    # Very high congestion
        "night": [0.25, 0.20, 0.30],      # Low congestion
        "now": [0.50, 0.40, 0.60]
    }
    
    # Get current hour if "now"
    if time_of_day == "now":
        hour = datetime.now().hour
        if 7 <= hour <= 10:
            time_of_day = "morning"
        elif 12 <= hour <= 15:
            time_of_day = "afternoon"
        elif 17 <= hour <= 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
    
    factors = time_factors.get(time_of_day, [0.5, 0.4, 0.6])
    base_factor = factors[route_index] if route_index < len(factors) else 0.5
    
    # Add randomness to simulate real prediction
    traffic_score = base_factor + np.random.uniform(-0.1, 0.1)
    traffic_score = max(0.0, min(1.0, traffic_score))
    
    # Classify traffic level
    if traffic_score < 0.4:
        traffic_level = "low"
        avg_speed = 50 + np.random.uniform(-5, 5)
    elif traffic_score < 0.7:
        traffic_level = "medium"
        avg_speed = 35 + np.random.uniform(-5, 5)
    else:
        traffic_level = "high"
        avg_speed = 20 + np.random.uniform(-5, 5)
    
    return traffic_level, traffic_score, avg_speed

def generate_route_variations(origin, destination, num_routes=3):
    """
    Generate alternative routes between two points
    In production, this would use OpenStreetMap/Google Maps API
    """
    routes = []
    
    # Calculate direct distance
    direct_distance = haversine_distance(
        origin.lat, origin.lng,
        destination.lat, destination.lng
    )
    
    # Generate route variations
    for i in range(num_routes):
        # Create waypoints for different routes
        if i == 0:  # Direct route
            waypoints = [
                [origin.lat, origin.lng],
                [origin.lat + (destination.lat - origin.lat) * 0.5,
                 origin.lng + (destination.lng - origin.lng) * 0.5],
                [destination.lat, destination.lng]
            ]
            route_name = "Fastest Route"
            distance_multiplier = 1.0
            
        elif i == 1:  # Eastern detour
            waypoints = [
                [origin.lat, origin.lng],
                [origin.lat + (destination.lat - origin.lat) * 0.3,
                 origin.lng + (destination.lng - origin.lng) * 0.3 + 0.02],
                [origin.lat + (destination.lat - origin.lat) * 0.7,
                 origin.lng + (destination.lng - origin.lng) * 0.7 + 0.02],
                [destination.lat, destination.lng]
            ]
            route_name = "Alternative Route 1"
            distance_multiplier = 1.15
            
        else:  # Western detour
            waypoints = [
                [origin.lat, origin.lng],
                [origin.lat + (destination.lat - origin.lat) * 0.3,
                 origin.lng + (destination.lng - origin.lng) * 0.3 - 0.02],
                [origin.lat + (destination.lat - origin.lat) * 0.7,
                 origin.lng + (destination.lng - origin.lng) * 0.7 - 0.01],
                [destination.lat, destination.lng]
            ]
            route_name = "Alternative Route 2"
            distance_multiplier = 1.2
        
        routes.append({
            "name": route_name,
            "coords": waypoints,
            "distance": direct_distance * distance_multiplier
        })
    
    return routes

def find_nearby_services(route_coords, service_type):
    """
    Find nearby services along the route
    In production, this would query Places API or local database
    """
    # Simulate service locations
    services_db = {
        "hospital": [
            {"name": "Fortis Hospital", "lat": 12.9352, "lng": 77.6245},
            {"name": "Apollo Hospital", "lat": 12.9140, "lng": 77.6220},
            {"name": "Manipal Hospital", "lat": 12.8880, "lng": 77.5970},
            {"name": "Columbia Asia", "lat": 12.9200, "lng": 77.6100},
        ],
        "fuel": [
            {"name": "Indian Oil Petrol Pump", "lat": 12.9300, "lng": 77.6200},
            {"name": "HP Petrol Pump", "lat": 12.9100, "lng": 77.6150},
            {"name": "Bharat Petroleum", "lat": 12.8950, "lng": 77.6000},
            {"name": "Shell Petrol Station", "lat": 12.9250, "lng": 77.6180},
            {"name": "Essar Petrol Pump", "lat": 12.8800, "lng": 77.6050},
        ],
        "restaurant": [
            {"name": "MTR Restaurant", "lat": 12.9320, "lng": 77.6210},
            {"name": "Vidyarthi Bhavan", "lat": 12.9180, "lng": 77.6180},
            {"name": "Cafe Coffee Day", "lat": 12.9050, "lng": 77.6050},
            {"name": "Truffles", "lat": 12.9280, "lng": 77.6240},
            {"name": "Empire Restaurant", "lat": 12.9150, "lng": 77.6120},
        ]
    }
    
    nearby = []
    mid_point = route_coords[len(route_coords) // 2]
    
    for service in services_db.get(service_type, []):
        distance = haversine_distance(
            mid_point[0], mid_point[1],
            service["lat"], service["lng"]
        )
        
        if distance < 5:  # Within 5 km
            nearby.append({
                "type": service_type,
                "name": service["name"],
                "lat": service["lat"],
                "lng": service["lng"],
                "distance_from_route": round(distance, 2)
            })
    
    return nearby

def detect_incidents(route_coords):
    """
    Detect traffic incidents along the route
    In production, this would use real-time traffic data/cameras
    """
    # Simulated incidents
    all_incidents = [
        {
            "id": 1,
            "type": "accident",
            "location": "Silk Board Junction",
            "severity": "high",
            "lat": 12.9165,
            "lng": 77.6230,
            "description": "Multi-vehicle collision causing major delays"
        },
        {
            "id": 2,
            "type": "roadblock",
            "location": "Electronic City Flyover",
            "severity": "medium",
            "lat": 12.8456,
            "lng": 77.6632,
            "description": "Construction work in progress"
        },
        {
            "id": 3,
            "type": "congestion",
            "location": "Marathahalli Bridge",
            "severity": "high",
            "lat": 12.9591,
            "lng": 77.6974,
            "description": "Heavy traffic buildup"
        }
    ]
    
    # Check if incidents are near the route
    incidents_on_route = []
    for incident in all_incidents:
        for coord in route_coords:
            distance = haversine_distance(
                coord[0], coord[1],
                incident["lat"], incident["lng"]
            )
            if distance < 2:  # Within 2 km
                incidents_on_route.append(incident)
                break
    
    return incidents_on_route

# ==================== API ENDPOINTS ====================

@app.get("/")
def root():
    return {
        "service": "Bengaluru Traffic Intelligence API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "routes": "/api/routes",
            "incidents": "/api/incidents",
            "stats": "/api/stats",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "bengaluru-traffic-api"
    }

@app.post("/api/routes", response_model=List[RouteResponse])
async def calculate_routes(request: RouteRequest):
    """
    Calculate alternative routes with traffic prediction
    """
    try:
        # Generate route variations
        routes = generate_route_variations(
            request.origin,
            request.destination,
            num_routes=3
        )
        
        route_responses = []
        fastest_route_idx = None
        min_duration = float('inf')
        
        for idx, route in enumerate(routes):
            # Predict traffic using LSTM (simulated)
            traffic_level, traffic_score, predicted_speed = predict_traffic_lstm(
                route["coords"],
                request.time_of_day,
                idx
            )
            
            # Calculate duration based on distance and predicted speed
            duration = (route["distance"] / predicted_speed) * 60  # in minutes
            
            # Track fastest route
            if duration < min_duration:
                min_duration = duration
                fastest_route_idx = idx
            
            # Find services along route
            hospitals = find_nearby_services(route["coords"], "hospital")
            fuel_stations = find_nearby_services(route["coords"], "fuel")
            restaurants = find_nearby_services(route["coords"], "restaurant")
            
            route_responses.append(RouteResponse(
                route_id=idx,
                name=route["name"],
                distance=round(route["distance"], 2),
                duration=round(duration, 1),
                traffic_level=traffic_level,
                traffic_score=round(traffic_score, 2),
                coordinates=route["coords"],
                is_fastest=False,  # Will be updated
                predicted_speed=round(predicted_speed, 1),
                services={
                    "hospitals": len(hospitals),
                    "fuel_stations": len(fuel_stations),
                    "restaurants": len(restaurants)
                }
            ))
        
        # Mark fastest route
        if fastest_route_idx is not None:
            route_responses[fastest_route_idx].is_fastest = True
        
        return route_responses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services/{route_id}")
async def get_route_services(route_id: int, service_type: Optional[str] = None):
    """
    Get detailed service information along a route
    """
    services = []
    
    if service_type is None or service_type == "hospital":
        services.extend(find_nearby_services([[12.92, 77.62]], "hospital"))
    if service_type is None or service_type == "fuel":
        services.extend(find_nearby_services([[12.92, 77.62]], "fuel"))
    if service_type is None or service_type == "restaurant":
        services.extend(find_nearby_services([[12.92, 77.62]], "restaurant"))
    
    return {"services": services}

@app.get("/api/incidents", response_model=List[Incident])
async def get_incidents():
    """
    Get all active traffic incidents
    """
    incidents = detect_incidents([[12.92, 77.62]])
    return incidents

@app.post("/api/predict")
async def predict_traffic(route_coords: List[List[float]], time_of_day: str = "now"):
    """
    Predict traffic conditions for a specific route
    """
    traffic_level, traffic_score, predicted_speed = predict_traffic_lstm(
        route_coords,
        time_of_day
    )
    
    return {
        "traffic_level": traffic_level,
        "traffic_score": round(traffic_score, 2),
        "predicted_speed": round(predicted_speed, 1),
        "recommendation": "Consider alternative route" if traffic_score > 0.7 else "Route looks good",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_traffic_stats():
    """
    Get overall traffic statistics for Bengaluru
    """
    return {
        "avg_speed": round(np.random.uniform(35, 50), 1),
        "congestion_level": np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2]),
        "active_incidents": np.random.randint(1, 5),
        "routes_analyzed": np.random.randint(100, 500),
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)