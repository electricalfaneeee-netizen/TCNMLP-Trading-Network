import torch
import torch.nn as nn
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
    def __init__(self):
        super().__init__()

        self.encoder = encoder

        self.MLP_stack = nn.Sequential(
                nn.Linear(30, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 30),
                nn.SiLU(),
         )

        self.actor = nn.Sequential(
                nn.Linear(60, 64),
                nn.SiLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
                nn.LogSoftmax(dim=-1),
        )

        self.critic = nn.Sequential(
                nn.Linear(60, 64),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(64, 1),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = torch.cat((self.MLP_stack(encoded), encoded), dim=-1)

        policy = self.actor(encoded)
        value = self.critic(encoded)

        return policy, value
