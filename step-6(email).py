import random
import time
import json
import sqlite3
import smtplib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from email.mime.text import MIMEText

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

### -------------------------------- STEP 2: Network Monitoring & Logging (SQLite) -------------------------------- ###

def monitor_network():
    return {
        "latency_ms": round(random.uniform(5, 50), 2),
        "packet_loss_pct": round(random.uniform(0, 5), 2),
        "jitter_ms": round(random.uniform(1, 10), 2),
        "bandwidth_mbps": round(random.uniform(10, 100), 2),
        "timestamp": time.time()
    }

# Database setup
conn = sqlite3.connect("network_logs.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS NetworkLogs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        latency_ms REAL,
        packet_loss_pct REAL,
        jitter_ms REAL,
        bandwidth_mbps REAL
    )
""")
conn.commit()

def store_network_log(network_data):
    """Stores network data into an SQLite database."""
    cursor.execute("""
        INSERT INTO NetworkLogs (timestamp, latency_ms, packet_loss_pct, jitter_ms, bandwidth_mbps)
        VALUES (?, ?, ?, ?, ?)
    """, (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(network_data["timestamp"])),
          network_data["latency_ms"], network_data["packet_loss_pct"],
          network_data["jitter_ms"], network_data["bandwidth_mbps"]))
    conn.commit()

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

### -------------------------------- STEP 4: Alert System (Email Notification) -------------------------------- ###

def send_alert(latency, packet_loss):
    sender_email = "your_email@gmail.com"
    receiver_email = "alert_receiver@gmail.com"
    subject = "🚨 Network Alert: High Latency Detected!"
    body = f"High Latency Detected: {latency} ms\nPacket Loss: {packet_loss} %"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, "your_password")
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("🚨 Alert Sent Successfully!")
    except Exception as e:
        print("❌ Failed to Send Alert:", e)

### -------------------------------- FINAL: Real-Time Monitoring, Logging & AI Optimization -------------------------------- ###

def real_time_network_monitoring(interval=1, duration=20, num_sensors=5):
    start_time = time.time()

    while time.time() - start_time < duration:
        sensor_data_batch = generate_multiple_sensor_data(num_sensors)
        transmit_data(sensor_data_batch)

        network_data = monitor_network()
        store_network_log(network_data)

        # AI Decision-Making
        action = agent.select_action(list(network_data.values())[:-1])
        reward = 100 - network_data["latency_ms"]
        agent.store_experience(list(network_data.values())[:-1], action, reward, list(network_data.values())[:-1])
        agent.train()

        # Alert if latency > 30ms
        if network_data["latency_ms"] > 30:
            send_alert(network_data["latency_ms"], network_data["packet_loss_pct"])

        print(f"🤖 AI Action: {ACTIONS[action]} | Latency: {network_data['latency_ms']} ms")

        time.sleep(interval)

# Run the full AI-based IoT optimization with deployment
ACTIONS = ["Reduce Power", "Change Routing", "Modify Bandwidth", "Prioritize Data"]
agent = DQNAgent(input_dim=4, output_dim=4)
real_time_network_monitoring(interval=1, duration=20, num_sensors=5)
z