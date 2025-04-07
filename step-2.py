import random
import time
import json
import csv
import matplotlib.pyplot as plt
from collections import deque

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# CSV file to store data
csv_filename = "iot_network_data.csv"

# Write CSV headers before starting data logging
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Timestamp", "Sensor_ID", "Temperature (°C)", "Traffic Load", "Energy Level (%)",
        "Latency (ms)", "Packet Loss (%)", "Jitter (ms)", "Bandwidth (Mbps)"
    ])

### -------------------------------- STEP 1: IoT Sensing & Data Transmission -------------------------------- ###

# Function to simulate IoT sensor data
def generate_sensor_data(sensor_id):
    return {
        "sensor_id": f"Sensor_{sensor_id}",
        "temperature": round(random.uniform(20.0, 35.0), 2),  # Temperature in °C
        "traffic_load": random.randint(10, 100),  # Network traffic load (arbitrary units)
        "energy_level": round(random.uniform(50.0, 100.0), 2),  # Battery percentage
        "timestamp": time.strftime("%H:%M:%S", time.localtime())  # Readable timestamp
    }

# Function to generate multiple IoT sensor data entries
def generate_multiple_sensor_data(num_sensors=10):
    return [generate_sensor_data(i) for i in range(1, num_sensors + 1)]

# Function to simulate IoT data transmission over a 6G network
def transmit_data(sensor_data_list):
    for data in sensor_data_list:
        json_data = json.dumps(data)  # Convert to JSON format
        print(f"📡 Transmitting Data: {json_data}")  # Simulating network transmission

### -------------------------------- STEP 2: Network Monitoring & Visualization -------------------------------- ###

# Function to simulate real-time network performance monitoring
def monitor_network():
    return {
        "latency_ms": round(random.uniform(5, 100), 2),  # Network latency in milliseconds
        "packet_loss_pct": round(random.uniform(0, 10), 2),  # Packet loss percentage
        "jitter_ms": round(random.uniform(1, 20), 2),  # Network jitter in milliseconds
        "bandwidth_mbps": round(random.uniform(10, 200), 2)  # Available bandwidth in Mbps
    }

# Initialize deque for real-time data storage (fixed size for smooth visualization)
time_window = 30  # Store last 20 entries
latency_values = deque(maxlen=time_window)
packet_loss_values = deque(maxlen=time_window)
jitter_values = deque(maxlen=time_window)
bandwidth_values = deque(maxlen=time_window)
timestamps = deque(maxlen=time_window)

# Set up Matplotlib for real-time plotting
plt.ion()  # Turn on interactive mode
fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 subplot layout

# Function to update and visualize network data
def real_time_network_monitoring(interval=1, duration=35, num_sensors=5):
    """
    Simulates real-time IoT data transmission and network monitoring with visualization and CSV logging.

    :param interval: Time interval (seconds) between each log entry
    :param duration: Total duration (seconds) to run the simulation
    :param num_sensors: Number of IoT sensors generating data
    """
    start_time = time.time()

    while time.time() - start_time < duration:
        # Step 1: Generate and transmit IoT data
        sensor_data_batch = generate_multiple_sensor_data(num_sensors)
        transmit_data(sensor_data_batch)

        # Step 2: Monitor and log network performance
        network_data = monitor_network()
        latency_values.append(network_data["latency_ms"])
        packet_loss_values.append(network_data["packet_loss_pct"])
        jitter_values.append(network_data["jitter_ms"])
        bandwidth_values.append(network_data["bandwidth_mbps"])
        timestamps.append(time.strftime("%H:%M:%S", time.localtime()))

        # Save data to CSV file
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            for sensor_data in sensor_data_batch:
                writer.writerow([
                    sensor_data["timestamp"], sensor_data["sensor_id"],
                    sensor_data["temperature"], sensor_data["traffic_load"], sensor_data["energy_level"],
                    network_data["latency_ms"], network_data["packet_loss_pct"],
                    network_data["jitter_ms"], network_data["bandwidth_mbps"]
                ])

        # Clear previous plots
        axs[0, 0].cla()
        axs[0, 1].cla()
        axs[1, 0].cla()
        axs[1, 1].cla()

        # Plot Latency
        axs[0, 0].plot(timestamps, latency_values, marker='o', color='r', linestyle='-', label="Latency (ms)")
        # axs[0, 0].set_title("Latency Over Time")
        axs[0, 0].set_ylabel("Latency (ms)", fontweight='bold')
        axs[0, 0].set_xticklabels(timestamps, rotation=45)
        axs[0, 0].legend()

        # Plot Packet Loss
        axs[0, 1].plot(timestamps, packet_loss_values, marker='s', color='b', linestyle='-', label="Packet Loss (%)")
        # axs[0, 1].set_title("Packet Loss Over Time")
        axs[0, 1].set_ylabel("Packet Loss (%)",fontweight='bold')
        axs[0, 1].set_xticklabels(timestamps, rotation=45)
        axs[0, 1].legend()

        # Plot Jitter
        axs[1, 0].plot(timestamps, jitter_values, marker='^', color='g', linestyle='-', label="Jitter (ms)")
        # axs[1, 0].set_title("Jitter Over Time")
        axs[1, 0].set_ylabel("Jitter (ms)",fontweight='bold')
        axs[1, 0].set_xticklabels(timestamps, rotation=45)
        axs[1, 0].legend()

        # Plot Bandwidth
        axs[1, 1].plot(timestamps, bandwidth_values, marker='d', color='purple', linestyle='-', label="Bandwidth (Mbps)")
        # axs[1, 1].set_title("Bandwidth Over Time")
        axs[1, 1].set_ylabel("Bandwidth (Mbps)",fontweight='bold')
        axs[1, 1].set_xticklabels(timestamps, rotation=45)
        axs[1, 1].legend()

        # Refresh plots
        plt.tight_layout()
        plt.pause(interval)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show final plot

# Run the real-time IoT data transmission & network monitoring visualization
real_time_network_monitoring(interval=1, duration=35, num_sensors=5)

print(f"✅ Data has been saved in '{csv_filename}'.")
