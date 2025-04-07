import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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

### -------------------------------- STEP 2: Network Monitoring & Visualization -------------------------------- ###

def monitor_network():
    return {
        "latency_ms": round(random.uniform(5, 50), 2),
        "packet_loss_pct": round(random.uniform(0, 5), 2),
        "jitter_ms": round(random.uniform(1, 10), 2),
        "bandwidth_mbps": round(random.uniform(10, 100), 2),
        "timestamp": time.time()
    }

time_window = 20  
latency_values = deque(maxlen=time_window)
packet_loss_values = deque(maxlen=time_window)
jitter_values = deque(maxlen=time_window)
bandwidth_values = deque(maxlen=time_window)
timestamps = deque(maxlen=time_window)

plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

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

### -------------------------------- STEP 5: Performance Evaluation -------------------------------- ###

def evaluate_performance(results):
    """Compares AI optimization vs. Rule-Based Optimization."""
    ai_latency = [r["ai_latency"] for r in results]
    rule_latency = [r["rule_latency"] for r in results]
    timestamps = [r["timestamp"] for r in results]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, ai_latency, label="AI Optimized Latency", color="blue", linestyle="-", marker="o")
    plt.plot(timestamps, rule_latency, label="Rule-Based Latency", color="red", linestyle="--", marker="s")

    plt.xlabel("Time", fontweight="bold")
    plt.ylabel("Latency (ms)", fontweight="bold")
    # plt.title("AI Optimization vs Rule-Based Optimization")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

def real_time_network_monitoring(interval=1, duration=20, num_sensors=5):
    start_time = time.time()
    results = []

    while time.time() - start_time < duration:
        sensor_data_batch = generate_multiple_sensor_data(num_sensors)
        transmit_data(sensor_data_batch)

        network_state = list(monitor_network().values())[:-1]
        latency_values.append(network_state[0])
        packet_loss_values.append(network_state[1])
        jitter_values.append(network_state[2])
        bandwidth_values.append(network_state[3])
        timestamps.append(time.strftime("%H:%M:%S", time.localtime(time.time())))

        action = agent.select_action(network_state)
        new_network_state = list(monitor_network().values())[:-1]
        reward = reward_function(*new_network_state[:3])
        agent.store_experience(network_state, action, reward, new_network_state)
        agent.train()

        rule_based_latency = random.uniform(10, 40)  # Simulated Rule-Based Latency
        ai_optimized_latency = new_network_state[0]

        results.append({
            "timestamp": time.strftime("%H:%M:%S", time.localtime(time.time())),
            "rule_latency": rule_based_latency,
            "ai_latency": ai_optimized_latency
        })

        print(f"🤖 AI Action: {ACTIONS[action]} | AI Latency: {ai_optimized_latency:.2f}ms | Rule-Based Latency: {rule_based_latency:.2f}ms")

        plt.pause(interval)

    plt.ioff()
    evaluate_performance(results)

real_time_network_monitoring(interval=1, duration=20, num_sensors=5)
