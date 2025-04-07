import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import deque

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 25
### -------------------------------- STEP 1: IoT Sensing & Data Transmission -------------------------------- ###

def generate_sensor_data(sensor_id):
    return {
        "sensor_id": f"Sensor_{sensor_id}",
        "temperature": round(random.uniform(20.0, 35.0), 2),
        "traffic_load": random.randint(10, 100),
        "energy_level": round(random.uniform(50.0, 100.0), 2),
        "timestamp": time.time()
    }

def generate_multiple_sensor_data(num_sensors=10):
    return [generate_sensor_data(i) for i in range(1, num_sensors + 1)]

def transmit_data(sensor_data_list):
    for data in sensor_data_list:
        json_data = json.dumps(data)
        print(f"📡 Transmitting Data: {json_data}")

### -------------------------------- STEP 2: Network Monitoring -------------------------------- ###

def monitor_network():
    return {
        "latency_ms": round(random.uniform(5, 100), 2),
        "packet_loss_pct": round(random.uniform(0, 10), 2),
        "jitter_ms": round(random.uniform(1, 20), 2),
        "bandwidth_mbps": round(random.uniform(10, 200), 2),
        "timestamp": time.time()
    }

time_window = 35  
reward_values = deque(maxlen=time_window)
timestamps = deque(maxlen=time_window)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

### -------------------------------- STEP 3: AI-Based Optimization Using DQN -------------------------------- ###

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.95):
        self.model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma  
        self.memory = deque(maxlen=1000)  

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store_experience(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return  

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)

            q_values = self.model(state)
            next_q_values = self.model(next_state)

            target_q = reward + self.gamma * torch.max(next_q_values)
            loss = self.criterion(q_values[action], target_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def reward_function(latency, packet_loss, jitter):
    reward = 100
    if latency > 20:  
        reward -= 20
    if packet_loss > 2:  
        reward -= 30
    if jitter > 5:  
        reward -= 10
    return reward

ACTIONS = ["Reduce Power", "Change Routing", "Modify Bandwidth", "Prioritize Data"]
agent = DQNAgent(input_dim=4, output_dim=4) 

### -------------------------------- Running AI Optimization Loop (Only Reward Graph + CSV Logging) -------------------------------- ###

CSV_FILE = "ai_rewards.csv"

def initialize_csv():
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Reward"])

def save_to_csv(timestamp, reward):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, reward])

initialize_csv()  

def real_time_network_monitoring(interval=1, duration=35, num_sensors=5):
    start_time = time.time()

    while time.time() - start_time < duration:
        sensor_data_batch = generate_multiple_sensor_data(num_sensors)
        transmit_data(sensor_data_batch)

        network_state = list(monitor_network().values())[:-1]
        action = agent.select_action(network_state)
        new_network_state = list(monitor_network().values())[:-1]
        reward = reward_function(*new_network_state[:3])
        reward_values.append(reward)  
        timestamp = time.strftime("%H:%M:%S", time.localtime(time.time()))
        timestamps.append(timestamp)

        agent.store_experience(network_state, action, reward, new_network_state)
        agent.train()

        print(f"🤖 AI Action Taken: {ACTIONS[action]} | Reward: {reward}")

        save_to_csv(timestamp, reward)

        ax.cla()
        ax.plot(timestamps, reward_values, 'c-', label="Reward Score")
        # ax.set_title("AI Reward Over Time")
        ax.set_xlabel("Time", fontweight='bold')
        ax.set_ylabel("Reward", fontweight='bold')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)  # Rotates x-axis labels by 45 degrees

        plt.tight_layout()
        plt.pause(interval)

    plt.ioff()
    plt.show()

# Run the System with AI Optimization & Reward Logging
real_time_network_monitoring(interval=1, duration=20, num_sensors=5)
