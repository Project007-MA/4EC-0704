import random
import time
import json
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
# Store the data for visualization
sensor_data_history = []

# Function to generate a single IoT sensor data entry
def generate_sensor_data(sensor_id):
    data = {
        "sensor_id": f"Sensor_{sensor_id}",
        "temperature": round(random.uniform(20.0, 35.0), 2),  # Temperature in °C
        "traffic_load": random.randint(10, 100),  # Network traffic load (arbitrary units)
        "energy_level": round(random.uniform(50.0, 100.0), 2),  # Battery percentage
        "timestamp": time.time()  # Unix timestamp for real-time tracking
    }
    sensor_data_history.append(data)  # Store for visualization
    return data

# Function to generate multiple IoT sensor data entries
def generate_multiple_sensor_data(num_sensors=10):
    return [generate_sensor_data(i) for i in range(1, num_sensors + 1)]

# Function to simulate data transmission
def transmit_data(sensor_data_list):
    for data in sensor_data_list:
        json_data = json.dumps(data)  # Convert to JSON format
        print(f"📡 Transmitting Data: {json_data}")  # Simulating network transmission

# Simulate real-time IoT data collection and transmission
def real_time_iot_simulation(num_sensors=5, interval=2, duration=10):
    start_time = time.time()
    while time.time() - start_time < duration:
        sensor_data_batch = generate_multiple_sensor_data(num_sensors)
        transmit_data(sensor_data_batch)
        time.sleep(interval)  # Wait before generating the next batch

# Run the simulation
real_time_iot_simulation(num_sensors=5, interval=2, duration=10)

# Visualization
def plot_sensor_data():
    timestamps = [data["timestamp"] for data in sensor_data_history]
    temperatures = [data["temperature"] for data in sensor_data_history]
    traffic_loads = [data["traffic_load"] for data in sensor_data_history]
    energy_levels = [data["energy_level"] for data in sensor_data_history]

    plt.figure(figsize=(12, 6))

    # Plot Temperature
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, temperatures, marker='o', linestyle='-', color='r', label="Temperature (°C)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid()

    # Plot Traffic Load
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, traffic_loads, marker='s', linestyle='-', color='b', label="Traffic Load")
    plt.ylabel("Traffic Load")
    plt.legend()
    plt.grid()

    # Plot Energy Level
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, energy_levels, marker='d', linestyle='-', color='g', label="Energy Level (%)")
    plt.ylabel("Energy Level (%)")
    plt.xlabel("Timestamp")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Call the function to visualize data
plot_sensor_data()
