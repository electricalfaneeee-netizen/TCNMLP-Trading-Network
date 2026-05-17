import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent

class TCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(5, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 12, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(12),
            nn.SiLU(),
            nn.Conv1d(12, 8, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(8),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x).transpose(-1, -2)

class TCNMLP(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder

        self.feature_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.LayerNorm(32)
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.SiLU(),
            nn.LayerNorm(8)
        )

        self.unrealized_pnl_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.SiLU(),
            nn.LayerNorm(8)
        )

        self.actor = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 16, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 16, 128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 1)
        )

        self.transformer_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.SiLU(),
            nn.LayerNorm(64)
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(self.encoder_layer, 4)

        self.pos_emb = nn.Embedding(25, 8)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, state, unrealized_pnl):
        encoded_features = self.encoder(x)

        seq_len = encoded_features.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        t_in = self.transformer_mlp(encoded_features + self.pos_emb(positions))

        transformer_features = self.transformer(t_in, mask=mask)
        
        chart_features = self.feature_mlp(transformer_features)
        state_features = torch.cat((self.state_mlp(state.unsqueeze(-1)), self.unrealized_pnl_mlp(unrealized_pnl.unsqueeze(-1))), dim=-1)

        avg_pool = self.avgpool(chart_features.transpose(1, 2)).squeeze(-1)
        max_pool = self.maxpool(chart_features.transpose(1, 2)).squeeze(-1)
        last_step = chart_features[:, -1, :]

        combined = torch.cat((avg_pool, max_pool, last_step, state_features.squeeze(1)), dim=-1)

        policy = self.actor(combined)
        value = self.critic(combined)

        return policy, value

class TradingEnv(gym.Env):
    def __init__(self, coin_vault, window_size=100, max_steps=3072):
        super().__init__()

        self.coin_vault = coin_vault

        self.action_space = spaces.Discrete(2)

        self.window_size = window_size
        self.observation_space = spaces.Dict({
            "chart": spaces.Box(-np.inf, np.inf, shape=(5, self.window_size), dtype=np.float32),
            "state": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "unrealized_pnl": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        })
        
        self.current_step = 0
        self.active_state = 0
        self.idx_pos = 0
        self.max_steps = max_steps

        self.coin_names = list(self.coin_vault.keys())

        self.ema_decay = 0.01

        # running mean of returns
        self.eta = 0

        # running mean of squared returns
        self.sigma = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        selected_coin = np.random.choice(self.coin_names)
        active_data = self.coin_vault[selected_coin]
        
        self.active_windows = active_data["windows"]
        self.active_returns = active_data["log_returns"]

        max_start = len(self.active_windows) - self.max_steps - 1
        self.start_idx = np.random.randint(self.window_size, max_start)

        self.current_step = 0
        self.active_state = 0

        self.idx_pos = self.start_idx
        self.steps_taken = 0

        self.eta = 0
        self.sigma = 0

        observations = {
            "chart": self.active_windows[self.idx_pos].cpu().numpy().astype(np.float32),
            "state": np.array([self.active_state], dtype=np.float32),
            "unrealized_pnl": np.array([0.0], dtype=np.float32)
        }

        return observations, {}

    def step(self, action: int) -> Tuple:
        fee = 0
        if action != self.active_state:
            fee = -0.0001

        if action == 1 and self.active_state == 0:
            self.entry_idx = self.idx_pos

        self.idx_pos += 1
        self.steps_taken += 1

        market_return = self.active_returns[self.idx_pos, 0].item()
        step_returns = np.exp(market_return) if action == 1 else 0

        delta_eta = step_returns - self.eta
        delta_sigma = (step_returns ** 2) - self.sigma

        raw_dsr = ((self.sigma * delta_eta) - (0.5 * self.eta * delta_sigma)) / ((np.maximum(self.sigma - self.eta ** 2, 1e-8)) ** 1.5)
        dsr = np.clip(raw_dsr, -1.0, 1.0)

        self.eta += self.ema_decay * delta_eta
        self.sigma += self.ema_decay * delta_sigma

        returns = (market_return * 10) if action == 1 else 0
        returns += fee

        if action == 1:
            unrealized_pnl = torch.sum(self.active_returns[self.entry_idx:self.idx_pos + 1, 0]).item() * 10
        else:
            unrealized_pnl = 0.0

        self.active_state = action
        done = self.steps_taken >= self.max_steps
        
        observation = {
            "chart": self.active_windows[self.idx_pos].cpu().numpy().astype(np.float32),
            "state": np.array([self.active_state], dtype=np.float32),
            "unrealized_pnl": np.array([unrealized_pnl], dtype=np.float32)
        }

        info = {
            "returns": returns
        }

        return observation, float(dsr), done, False, info

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
