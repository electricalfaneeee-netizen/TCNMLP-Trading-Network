import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

script_dir = Path(__file__).parent

class TCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv1d(5, 10, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(5, 10, kernel_size=7, padding=3)
        self.cnn3 = nn.Conv1d(5, 10, kernel_size=15, padding=7)
        self.batchNorm = nn.BatchNorm1d(30)
        self.activation = nn.Tanh()

    def forward(self, x):
        x1, x2, x3 = self.cnn1(x), self.cnn2(x), self.cnn3(x)
        out = torch.cat([x1, x2, x3], dim=1)

        return self.activation(self.batchNorm(out)).transpose(1, 2)

encoder = torch.compile(TCNEncoder().to(device))
encoder.load_state_dict(torch.load(f"{script_dir}/models/encoder.pt", map_location=device))
    
class TCNMLP(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        self.MLP_stack = nn.Sequential(
            nn.Linear(31, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 30),
            nn.SiLU()
         )

        self.actor = nn.Sequential(
            nn.Linear(61, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(61, 64),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1)
        )

    def forward(self, x, state):
        x = x.transpose(1, 2)

        encoded = self.encoder(x)
        last_encoding = encoded[:, -1, :]

        state_float = state.float()
        encoded = torch.cat((last_encoding, state_float.unsqueeze(-1)), dim=-1)

        encoded = torch.cat((self.MLP_stack(encoded), encoded), dim=-1)

        policy = self.actor(encoded)
        value = self.critic(encoded)

        return policy, value

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size: int = 100):
        super().__init__()
        self.data = df.to_numpy().astype(np.float32)
        self.window_size = window_size
        _windows = sliding_window_view(self.data, window_shape=self.window_size, axis=0)
        self.windows = np.ascontiguousarray(_windows.transpose(0, 2, 1))

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Dict({
            "chart": spaces.Box(-np.inf, np.inf, shape=(self.window_size, 5), dtype=np.float32),
            "state": spaces.Discrete(2)
        })
        
        self.current_step = 0
        self.active_state = 0
        self.entry_price_idx = 0
        self.idx_pos = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.active_state = 0
        self.entry_price_idx = 0
        self.idx_pos = self.window_size

        observations = {
            "chart": self.windows[self.idx_pos - self.window_size],
            "state": self.active_state
        }

        return observations, {}

    def step(self, action: int) -> Tuple:
        traded = action != self.active_state
        
        if traded:
            self.entry_price_idx = self.idx_pos
            
        self.idx_pos += 1
        current_return = self.data[self.idx_pos, 0]

        transaction_cost = 0.001 if traded else 0.0

        if action == 1:
            reward = current_return - transaction_cost
        else:
            reward = -transaction_cost
            reward -= 0.00001
        
        self.active_state = action

        done = self.idx_pos >= len(self.data) - 1
        
        observation = {
            "chart": self.windows[self.idx_pos - self.window_size],
            "state": self.active_state
        }
        return observation, reward, done, False, {}

