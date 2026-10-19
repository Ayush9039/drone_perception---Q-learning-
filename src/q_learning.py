import numpy as np
import json
import os

class DroneTrafficQAgent:
    def __init__(self, actions=None, lr=0.1, gamma=0.9, epsilon=0.2):
        # State: Vehicles count ke hisaab se (0: Low, 1: Medium, 2: High)
        self.states = [0, 1, 2]
        
        # Action: Drone perception / decision (0: Hold/Maintain, 1: Ascend/Expand View, 2: Alert/Focus)
        self.actions = actions if actions else [0, 1, 2]
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-Table initialize karna (States x Actions)
        self.q_table = np.zeros((len(self.states), len(self.actions)))
        self.history = []

    def get_state(self, vehicle_count):
        """Vehicle count ke basis par state decide karna"""
        if vehicle_count < 3:
            return 0  # Low Traffic
        elif vehicle_count <= 7:
            return 1  # Medium Traffic
        else:
            return 2  # High Traffic (Congestion)

    def choose_action(self, state):
        """Epsilon-greedy policy se action select karna"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        return int(np.argmax(self.q_table[state]))

    def get_reward(self, state, action):
        """Traffic state ke hisaab se action par reward dena"""
        # Example logic: Agar high traffic (state 2) me Alert/Focus (action 2) kiya toh positive reward
        if state == 2:
            return 10 if action == 2 else -5
        elif state == 1:
            return 5 if action == 1 else 0
        else:
            return 5 if action == 0 else -2

    def update(self, state, action, reward, next_state):
        """Bellman equation se Q-value update karna"""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        
        # Q(s, a) = Q(s, a) + lr * (r + gamma * max(Q(s', a')) - Q(s, a))
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state, action] = round(new_value, 4)

        # Log history
        self.history.append({
            "state": int(state),
            "action": int(action),
            "reward": float(reward),
            "next_state": int(next_state)
        })

    def save_to_json(self, file_path="q_learning_results.json"):
        """Q-Table aur logs ko clean JSON me export karna"""
        data = {
            "q_table": self.q_table.tolist(),
            "state_labels": {"0": "Low Traffic", "1": "Medium Traffic", "2": "High Traffic"},
            "action_labels": {"0": "Maintain", "1": "Expand View", "2": "Alert/Focus"},
            "total_steps": len(self.history),
            "step_logs": self.history[-30:]  # Last 30 steps ka record
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[Q-LEARNING LOG] Q-Table and logs successfully saved to: {file_path}")