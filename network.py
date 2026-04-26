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
        encoded = self.encoder(x)
        last_encoding = encoded[:, -1, :]

        state_float = state.float()
        encoded = torch.cat((last_encoding, state_float.unsqueeze(-1)), dim=-1)

        encoded = torch.cat((self.MLP_stack(encoded), encoded), dim=-1)

        policy = self.actor(encoded)
        value = self.critic(encoded)

        return policy, value

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size=100, max_steps=2048):
        super().__init__()
        self.raw_data = df.to_numpy().astype(np.float32)
        self.window_size = window_size
        
        _windows = np.lib.stride_tricks.sliding_window_view(self.raw_data, window_shape=window_size, axis=0)
        windows = np.ascontiguousarray(_windows.transpose(0, 2, 1))
        
        price_slice = windows[:, :4, :]
        p_mean = price_slice.mean(axis=(1, 2), keepdims=True)
        p_std = price_slice.std(axis=(1, 2), keepdims=True) + 1e-9
        windows[:, :4, :] = (price_slice - p_mean) / p_std
        
        vol_slice = windows[:, 4:, :]
        v_mean = vol_slice.mean(axis=(1, 2), keepdims=True)
        v_std = vol_slice.std(axis=(1, 2), keepdims=True) + 1e-9
        windows[:, 4:, :] = (vol_slice - v_mean) / v_std
        
        self.windows = windows

        self.log_returns = normalize_reward_data(df)

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Dict({
            "chart": spaces.Box(-np.inf, np.inf, shape=(self.window_size, 5), dtype=np.float32),
            "state": spaces.Discrete(2)
        })
        
        self.current_step = 0
        self.active_state = 0
        self.idx_pos = 0

        self.max_steps = max_steps
        self.total_rows = len(self.windows)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self.total_rows - self.max_steps - 1
        self.start_idx = np.random.randint(self.window_size, max_start)

        self.current_step = 0
        self.active_state = 0

        self.idx_pos = self.start_idx
        self.steps_taken = 0

        observations = {
            "chart": self.windows[self.idx_pos],
            "state": self.active_state
        }

        return observations, {}

    def step(self, action: int) -> Tuple:
        fee = 0
        if action != self.active_state:
            fee = -0.01

        self.idx_pos += 1
        self.steps_taken += 1

        market_return = self.log_returns[self.idx_pos, 0]
    
        reward = (market_return * 100) if action == 1 else 0
        reward += fee

        self.active_state = action

        done = self.steps_taken >= self.max_steps
        
        observation = {
            "chart": self.windows[self.idx_pos],
            "state": self.active_state
        }

        return observation, reward, done, False, {}

def normalize_reward_data(df: pd.DataFrame):
    df_nn = pd.DataFrame(index=df.index)
    
    prev_close = df['Close'].shift(1)
    
    df_nn['close'] = np.log(df['Close'] / prev_close)
    
    df_nn['high'] = np.log(df['High'] / prev_close)
    
    df_nn['low'] = np.log(df['Low'] / prev_close)
    
    df_nn['open'] = np.log(df['Open'] / prev_close)
    df_nn['volume'] = np.log(df['Volume'] + 1) - np.log(df['Volume'].rolling(20).mean() + 1)   
    
    df_nn = df_nn.replace([np.inf, -np.inf], np.nan)
    return df_nn.fillna(0).to_numpy()
