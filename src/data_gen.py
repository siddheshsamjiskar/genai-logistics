import pandas as pd
import numpy as np
from geopy.distance import geodesic

# Step 1: Create random warehouse and delivery locations (latitude, longitude)
warehouse_location = (19.0760,72.8777)

np.random.seed(42)
# Generating 50 random delivery locations around the warehouse
def generate_random_location(center,num_points=50,radius=0.2):
    locations = []
    for _ in range(num_points):
        lat_offset = np.random.uniform(-radius, radius)
        lon_offset = np.random.uniform(-radius, radius)
        locations.append((center[0] + lat_offset,center[1] + lon_offset))
    return locations

delivery_locations = generate_random_location(warehouse_location, 50)

#Step 2: Create dataset
data = []
for idx ,loc in enumerate(delivery_locations):
    distance_km = geodesic(warehouse_location, loc).km
    traffic_level = np.random.choice(['Low', 'Medium', 'High'])
    weather = np.random.choice(['Clear', 'Fog', 'Rain'])
    delay = np.random.choice([0, 1], p=[0.7, 0.3])  # 0 = No delay, 1 = Delay
    
    data.append([idx + 1, warehouse_location, loc, distance_km, traffic_level, weather, delay])

    df = pd.DataFrame(data, columns=['Delivery_ID', 'Warehouse', 'Destination', 'Distance_km', 'Traffic', 'Weather', 'Delay'])

    # Step 3: Save dataset
    df.to_csv('delivery_data.csv', index=False)

    print("Dataset created and saved as 'delivery_data.csv'")
    print(df.head())
    
