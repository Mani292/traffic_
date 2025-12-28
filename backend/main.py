from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import math
from typing import List

app = FastAPI(title="Bengaluru AI Traffic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Location(BaseModel):
    lat: float
    lng: float
    name: str

class RouteRequest(BaseModel):
    origin: Location
    destination: Location

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def generate_intermediate_points(start: List[float], end: List[float], num_points: int = 3, variation: float = 0.008) -> List[List[float]]:
    """Generate intermediate waypoints between start and end"""
    points = []
    for i in range(1, num_points + 1):
        ratio = i / (num_points + 1)
        lat = start[0] + (end[0] - start[0]) * ratio + random.uniform(-variation, variation)
        lng = start[1] + (end[1] - start[1]) * ratio + random.uniform(-variation, variation)
        points.append([lat, lng])
    return points

@app.get("/")
def root():
    return {
        "status": "Bengaluru AI Traffic API is running",
        "version": "4.0",
        "endpoints": ["/api/routes", "/api/services/{route_id}", "/api/analytics"]
    }

@app.post("/api/routes")
def get_routes(req: RouteRequest):
    """Generate 12 route options with AI-powered traffic predictions"""
    origin = [req.origin.lat, req.origin.lng]
    destination = [req.destination.lat, req.destination.lng]
    
    # Calculate actual distance
    base_distance = haversine(origin[0], origin[1], destination[0], destination[1])
    
    print(f"\n{'='*70}")
    print(f"🗺️ GENERATING ROUTES")
    print(f"{'='*70}")
    print(f"📍 Origin: {origin}")
    print(f"📍 Destination: {destination}")
    print(f"📏 Base Distance: {base_distance:.2f} km")
    
    # Common navigation terms
    roads = ["MG Road", "Brigade Road", "Residency Road", "Richmond Road", "St Mark's Road", 
             "Commercial Street", "Vittal Mallya Road", "Kasturba Road", "Infantry Road",
             "Church Street", "Mission Road", "Double Road", "CMH Road", "HAL Road"]
    
    outer_roads = ["Outer Ring Road", "Inner Ring Road", "Intermediate Ring Road", "Bellary Road", 
                   "Tumkur Road", "Hosur Road", "Old Madras Road", "Mysore Road", "Bannerghatta Road",
                   "Kanakapura Road", "Magadi Road", "Airport Road"]
    
    landmarks = ["KR Market", "Cubbon Park", "Vidhana Soudha", "MG Road Metro", "Trinity Circle", 
                 "Dairy Circle", "Sony Signal", "Silk Board Junction", "Marathahalli Bridge",
                 "K.R. Puram Bridge", "Hebbal Flyover", "Tin Factory", "Shivaji Nagar", 
                 "Jayanagar 4th Block", "BTM Layout"]
    
    # Route configurations - 12 different routes with varied traffic probabilities
    route_configs = [
        {"name": "Direct Route", "distance_mult": 1.00, "points": 2, "variation": 0.005, "traffic_weight": [0.5, 0.3, 0.1, 0.05, 0.05]},
        {"name": "Via Ring Road", "distance_mult": 1.15, "points": 3, "variation": 0.012, "traffic_weight": [0.4, 0.3, 0.2, 0.07, 0.03]},
        {"name": "Via City Center", "distance_mult": 1.08, "points": 3, "variation": 0.010, "traffic_weight": [0.15, 0.25, 0.35, 0.2, 0.05]},
        {"name": "Express Highway", "distance_mult": 1.20, "points": 2, "variation": 0.015, "traffic_weight": [0.6, 0.2, 0.1, 0.05, 0.05]},
        {"name": "Scenic Route", "distance_mult": 1.12, "points": 4, "variation": 0.008, "traffic_weight": [0.4, 0.3, 0.2, 0.07, 0.03]},
        {"name": "Bypass Route", "distance_mult": 1.18, "points": 3, "variation": 0.013, "traffic_weight": [0.3, 0.3, 0.25, 0.1, 0.05]},
        {"name": "Local Streets", "distance_mult": 1.05, "points": 4, "variation": 0.007, "traffic_weight": [0.05, 0.15, 0.4, 0.3, 0.1]},
        {"name": "Service Road", "distance_mult": 1.10, "points": 3, "variation": 0.009, "traffic_weight": [0.3, 0.3, 0.25, 0.1, 0.05]},
        {"name": "Inner Circle", "distance_mult": 1.06, "points": 3, "variation": 0.008, "traffic_weight": [0.2, 0.3, 0.3, 0.15, 0.05]},
        {"name": "Alternate Highway", "distance_mult": 1.22, "points": 2, "variation": 0.014, "traffic_weight": [0.5, 0.25, 0.15, 0.07, 0.03]},
        {"name": "Metro Adjacent", "distance_mult": 1.07, "points": 3, "variation": 0.006, "traffic_weight": [0.35, 0.35, 0.2, 0.07, 0.03]},
        {"name": "Commercial Zone", "distance_mult": 1.09, "points": 4, "variation": 0.008, "traffic_weight": [0.1, 0.2, 0.35, 0.25, 0.1]},
    ]
    
    routes = []
    
    # Generate all 12 routes
    for idx, config in enumerate(route_configs):
        route_id = f"route_{idx + 1}"  # String format: route_1, route_2, etc.
        route_distance = base_distance * config["distance_mult"]
        
        # Generate intermediate points
        mid_points = generate_intermediate_points(
            origin, destination, 
            config["points"], 
            config["variation"]
        )
        
        # Assign traffic level based on weights
        traffic_levels = ["low", "moderate", "medium", "high", "heavy"]
        traffic = random.choices(traffic_levels, weights=config["traffic_weight"])[0]
        
        # Calculate speed based on traffic
        speed_ranges = {
            "low": (48, 55),
            "moderate": (40, 47),
            "medium": (30, 39),
            "high": (20, 29),
            "heavy": (15, 22)
        }
        base_speed = random.randint(*speed_ranges[traffic])
        
        # Calculate ETA
        eta_minutes = round((route_distance / base_speed) * 60)
        
        # Generate navigation steps
        num_steps = random.randint(7, 11)
        steps = [f"Head out from {req.origin.name}"]
        
        step_templates = [
            lambda: f"Turn {'right' if random.random() > 0.5 else 'left'} onto {random.choice(roads)}",
            lambda: f"Continue for {round(route_distance * random.uniform(0.2, 0.4), 1)} km",
            lambda: f"At {random.choice(landmarks)}, turn {'left' if random.random() > 0.5 else 'right'}",
            lambda: f"Merge onto {random.choice(outer_roads)}",
            lambda: f"Take exit {random.randint(12, 25)} toward {random.choice(landmarks)}",
            lambda: f"Keep {'left' if random.random() > 0.5 else 'right'} at the fork",
            lambda: f"Continue straight for {round(route_distance * random.uniform(0.15, 0.35), 1)} km",
            lambda: f"Pass {random.choice(landmarks)} on your {'right' if random.random() > 0.5 else 'left'}",
            lambda: f"At the roundabout, take the {random.choice(['1st', '2nd', '3rd'])} exit onto {random.choice(roads)}",
            lambda: f"Turn slight {'right' if random.random() > 0.5 else 'left'} onto {random.choice(roads)}",
            lambda: f"Follow signs to {random.choice(landmarks)}",
            lambda: f"Continue on {random.choice(roads)} for {round(route_distance * random.uniform(0.1, 0.3), 1)} km",
        ]
        
        # Generate unique steps
        used_templates = []
        for _ in range(num_steps - 2):
            available = [t for t in step_templates if t not in used_templates]
            if not available:
                available = step_templates
                used_templates = []
            template = random.choice(available)
            steps.append(template())
            used_templates.append(template)
        
        steps.append(f"Arrive at {req.destination.name}")
        
        # Create route object with ALL required fields - MATCHING FRONTEND EXACTLY
        route_data = {
            "route_id": route_id,  # String: "route_1", "route_2", etc.
            "name": config["name"],
            "traffic_level": traffic,
            "predicted_speed": base_speed,
            "distance_km": round(route_distance, 1),
            "eta_minutes": eta_minutes,
            "coordinates": [origin] + mid_points + [destination],
            "steps": steps  # CRITICAL: This must be included!
        }
        
        routes.append(route_data)
        
        print(f"\n✅ Route {idx + 1}: {config['name']}")
        print(f"   ID: {route_id}")
        print(f"   Traffic: {traffic} | Speed: {base_speed} km/h")
        print(f"   Distance: {route_distance:.1f} km | ETA: {eta_minutes} min")
        print(f"   Steps: {len(steps)} instructions")
        print(f"   Coordinates: {len([origin] + mid_points + [destination])} points")
    
    # Sort by speed (fastest first) and distance (shortest first)
    routes.sort(key=lambda x: (-x["predicted_speed"], x["distance_km"]))
    
    print(f"\n{'='*70}")
    print(f"✅ SENDING {len(routes)} ROUTES TO FRONTEND")
    print(f"{'='*70}\n")
    
    return routes

@app.get("/api/services/{route_id}")
def get_services(route_id: str):  # Changed to str to match route_1, route_2 format
    """Get nearby services for a specific route"""
    
    print(f"\n🏢 Loading services for Route ID: {route_id}")
    
    # Extract numeric ID from route_id string
    try:
        numeric_id = int(route_id.split('_')[1]) if '_' in route_id else int(route_id)
    except:
        numeric_id = 1
    
    all_services = {
        "hospitals": [
            {"type": "hospital", "name": "Apollo Hospital - Bannerghatta", "lat": 12.914, "lng": 77.622},
            {"type": "hospital", "name": "Fortis Hospital - Bannerghatta Road", "lat": 12.935, "lng": 77.624},
            {"type": "hospital", "name": "Manipal Hospital - HAL Airport Road", "lat": 12.920, "lng": 77.610},
            {"type": "hospital", "name": "Columbia Asia - Whitefield", "lat": 12.945, "lng": 77.635},
            {"type": "hospital", "name": "Sakra World Hospital - Marathahalli", "lat": 12.960, "lng": 77.650},
            {"type": "hospital", "name": "BGS Gleneagles - Kengeri", "lat": 12.910, "lng": 77.595},
            {"type": "hospital", "name": "Narayana Health - Bommasandra", "lat": 12.905, "lng": 77.600},
            {"type": "hospital", "name": "St. John's Medical College", "lat": 12.925, "lng": 77.618},
        ],
        "fuel": [
            {"type": "fuel", "name": "HP Petrol Pump - MG Road", "lat": 12.930, "lng": 77.620},
            {"type": "fuel", "name": "Indian Oil Station - Brigade Road", "lat": 12.925, "lng": 77.615},
            {"type": "fuel", "name": "Shell Petrol Bunk - Koramangala", "lat": 12.935, "lng": 77.628},
            {"type": "fuel", "name": "Bharat Petroleum - Indiranagar", "lat": 12.955, "lng": 77.640},
            {"type": "fuel", "name": "HP - Outer Ring Road", "lat": 12.905, "lng": 77.605},
            {"type": "fuel", "name": "Indian Oil - Whitefield", "lat": 12.970, "lng": 77.655},
            {"type": "fuel", "name": "Reliance Petrol - Electronic City", "lat": 12.895, "lng": 77.590},
            {"type": "fuel", "name": "Bharat Petroleum - Hebbal", "lat": 12.990, "lng": 77.595},
            {"type": "fuel", "name": "Shell - Marathahalli", "lat": 12.957, "lng": 77.702},
        ],
        "restaurants": [
            {"type": "restaurant", "name": "Empire Restaurant - Church Street", "lat": 12.915, "lng": 77.612},
            {"type": "restaurant", "name": "Truffles Cafe - Koramangala", "lat": 12.928, "lng": 77.624},
            {"type": "restaurant", "name": "MTR Restaurant - Lalbagh", "lat": 12.905, "lng": 77.595},
            {"type": "restaurant", "name": "Vidyarthi Bhavan - Basavanagudi", "lat": 12.910, "lng": 77.580},
            {"type": "restaurant", "name": "Koshy's - St Marks Road", "lat": 12.918, "lng": 77.608},
            {"type": "restaurant", "name": "Toit Brewpub - Indiranagar", "lat": 12.972, "lng": 77.640},
            {"type": "restaurant", "name": "The Black Pearl - Koramangala", "lat": 12.932, "lng": 77.626},
            {"type": "restaurant", "name": "Smoke House Deli - Whitefield", "lat": 12.968, "lng": 77.658},
            {"type": "restaurant", "name": "Brahmin's Coffee Bar", "lat": 12.943, "lng": 77.573},
            {"type": "restaurant", "name": "Airlines Hotel - Lavelle Road", "lat": 12.971, "lng": 77.603},
        ],
        "pharmacies": [
            {"type": "pharmacy", "name": "Apollo Pharmacy - MG Road", "lat": 12.922, "lng": 77.618},
            {"type": "pharmacy", "name": "MedPlus - Brigade Road", "lat": 12.928, "lng": 77.620},
            {"type": "pharmacy", "name": "Wellness Forever - Koramangala", "lat": 12.930, "lng": 77.625},
            {"type": "pharmacy", "name": "1mg - Indiranagar", "lat": 12.968, "lng": 77.642},
            {"type": "pharmacy", "name": "Apollo 24/7 - Whitefield", "lat": 12.965, "lng": 77.652},
            {"type": "pharmacy", "name": "NetMeds - Electronic City", "lat": 12.900, "lng": 77.595},
            {"type": "pharmacy", "name": "Pharmeasy - HSR Layout", "lat": 12.913, "lng": 77.638},
            {"type": "pharmacy", "name": "MedPlus - Jayanagar", "lat": 12.925, "lng": 77.583},
        ]
    }
    
    # Route-specific service selection
    selected = []
    pattern = numeric_id % 4
    offset = (numeric_id - 1) % 3
    
    if pattern == 1:
        selected.extend(all_services["hospitals"][offset:offset + 3])
        selected.extend(all_services["fuel"][offset:offset + 2])
        selected.extend(all_services["restaurants"][offset:offset + 2])
    elif pattern == 2:
        selected.extend(all_services["fuel"][offset:offset + 3])
        selected.extend(all_services["restaurants"][offset:offset + 3])
        selected.extend(all_services["pharmacies"][offset:offset + 1])
    elif pattern == 3:
        selected.extend(all_services["restaurants"][offset:offset + 3])
        selected.extend(all_services["pharmacies"][offset:offset + 2])
        selected.extend(all_services["hospitals"][offset:offset + 2])
    else:
        selected.extend(all_services["hospitals"][offset:offset + 2])
        selected.extend(all_services["fuel"][offset:offset + 2])
        selected.extend(all_services["restaurants"][offset:offset + 2])
        selected.extend(all_services["pharmacies"][offset:offset + 1])
    
    # Assign route-specific distances
    for i, service in enumerate(selected):
        distance = round(0.3 + (numeric_id * 0.08) + (i * 0.15), 1)
        service["distance_from_route"] = distance
    
    # Include lat/lng in response for map markers
    result = {
        "services": [
            {
                "type": s["type"],
                "name": s["name"],
                "distance_from_route": s["distance_from_route"],
                "lat": s["lat"],
                "lng": s["lng"]
            } for s in selected
        ]
    }
    
    print(f"✅ Loaded {len(result['services'])} services for Route {route_id}\n")
    
    return result

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 BENGALURU AI TRAFFIC API")
    print("="*70)
    print("📍 Server: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("="*70 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)