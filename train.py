from ccxt.xt import Num
import torch
import torch.optim as optim
import ccxt
import pandas as pd
import time
import gymnasium as gym
from pathlib import Path
from network import TCNMLP, TCNEncoder, TradingEnv, device, script_dir
from torch.optim.lr_scheduler import LinearLR

# envsironment variables
exchange = ccxt.binance()
timeframe = '5m'
symbols = ["SOL/USDT", "AVAX/USDT", "APT/USDT", "ETH/USDT"]

# hyperparameters
ROUNDS = 1000
EPISODES_PER_ROUND = 10
NUM_ENVS = 10

LR = 3e-4

PPO_EPOCHS = 8
CLIP_EPSILON = 0.12
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5

def yank5mMarketData(start_date, end_date, symbol):
    # Parse dates to milliseconds timestamp
    start_date = start_date
    end_date = end_date
    since = exchange.parse8601(start_date)
    end_timestamp = exchange.parse8601(end_date)

    all_ohlcv = []
    limit = 1000  # Binance usually allows 1000 per call

    print(f"Fetching {symbol} data from {start_date}...")

    # 2. The Pagination Loop
    while since < end_timestamp:
        try:
            # Fetch a chunk of data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                break  # No more data available
                
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to be the time of the last candle + 1 timeframe duration
            # ohlcv[-1][0] is the timestamp of the last candle
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            
            # Respect rate limits (binance is fast, but safety first)
            time.sleep(exchange.rateLimit / 1000)
            
            # Safety break if we pass the end date (though fetch_ohlcv usually handles this)
            if last_timestamp >= end_timestamp:
                break

        except Exception as e:
            print(f"Error fetching data: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
    # 4. Cleanup
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)

    # Filter out any data that might have gone slightly past the end date
    df = df[df.index <= pd.Timestamp(end_date)]

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Drop timestamp
    df = df.reset_index(drop=True)
    return df

def compute_gae(rewards, values, next_value, masks, gamma=0.99, lam=0.95):
    n = len(rewards)
    advantages = torch.zeros((n, NUM_ENVS), device=device)
    last_gae_lam = 0
    
    next_v = next_value
    
    for t in reversed(range(n)):
        delta = rewards[t] + gamma * next_v * masks[t] - values[t]
        
        last_gae_lam = delta + gamma * lam * masks[t] * last_gae_lam
        advantages[t] = last_gae_lam
        
        next_v = values[t]
        
    returns = advantages + values
    return advantages, returns

def ppo_training_loop(envs, network, optimizer, scheduler, episodes):

    round_stats = []

    obs, _ = envs.reset()
    
    old_log_probs = torch.zeros((3072, NUM_ENVS, 2), device=device)
    values_tensor = torch.zeros((3072, NUM_ENVS), device=device)
    actions_tensor = torch.zeros((3072, NUM_ENVS), device=device)
    rewards_tensor = torch.zeros((3072, NUM_ENVS), device=device)
    masks_tensor = torch.zeros((3072, NUM_ENVS), device=device)

    charts_tensor = torch.zeros((3072, NUM_ENVS, 5, 100), device=device)
    states_tensor = torch.zeros((3072, NUM_ENVS, 2), device=device)

    episode_reward = 0

    for step in range(3072):
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            with torch.no_grad():
                log_probs, values = network(obs["chart"], obs["state"])
            
                probs = torch.exp(log_probs)
                actions = torch.multinomial(probs, 1).squeeze()
        
        charts_tensor[step] = obs["chart"].detach()
        states_tensor[step] = obs["state"].detach()

        old_log_probs[step] = log_probs.detach()
        values_tensor[step] = values.detach()
        actions_tensor[step] = actions.detach()
        
        new_obs, rewards, terms, truncs, _ = envs.step(actions)
        masks_tensor[step] = torch.tensor(1.0 - (terms | truncs), device=device)
        rewards_tensor[step] = torch.tensor(rewards, dtype=torch.float32, device=device)
        episode_reward += rewards
        obs = new_obs

    with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
        with torch.no_grad():
            chart_input = torch.as_tensor(obs["chart"], device=device)
            state_input = torch.as_tensor(obs["state"], device=device)
            
            _, next_value = network(chart_input, state_input)
            next_value = next_value.squeeze()

    advantages, returns = compute_gae(
        rewards_tensor,
        values_tensor,
        next_value,
        masks_tensor,
        GAMMA,
        GAE_LAMBDA
    )

    old_log_probs = old_log_probs.reshape(-1, 2)
    values_tensor = values_tensor.reshape(-1)
    actions_tensor = actions_tensor.reshape(-1)

    charts_tensor = charts_tensor.reshape(-1, 5, 100)
    states_tensor = states_tensor.reshape(-1, 2)

    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)


    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(PPO_EPOCHS):
            new_log_probs, new_values = network(charts_tensor, states_tensor)

            new_values = new_values.squeeze()

            current_log_probs = new_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            old_action_log_probs = old_log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()

            ratio = torch.exp(current_log_probs - old_action_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            v_targ = returns
            v_clipped = values_tensor + torch.clamp(new_values - values_tensor, -CLIP_EPSILON, CLIP_EPSILON)
            v_loss1 = (new_values - v_targ) ** 2
            v_loss2 = (v_clipped - v_targ) ** 2
            value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            entropy = -(torch.exp(new_log_probs) * new_log_probs).sum(dim=-1).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.3)
            
            optimizer.step()

        round_stats.append({
                "reward": episode_reward,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item()
        })   

        scheduler.step()

    torch.xpu.empty_cache()

    return network, round_stats


def main():
    encoder = TCNEncoder().to(device)
    encoder.load_state_dict(torch.load(f"{script_dir}/models/encoder.pt", map_location=device))

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()

    model_path = Path(f"{script_dir}/models/TCNMLP.pt")
        
    model = torch.compile(TCNMLP(encoder).to(device))
    model.train()

    if model_path.exists():
        load_existing_weights = input("do you want to use use the pre-trained model? y or n\n")
        if load_existing_weights == "y":
            model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-5)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.08, total_iters=10000)

    dfs = {}

    for symbol in symbols:

        base_name = symbol.split("/")[0].lower()
        path = Path(f"{script_dir}/{base_name}_5m.csv")

        if path.exists():
            print(f"Loading cached data for {symbol}...")
            dfs[symbol] = pd.read_csv(path)
        else:
            print(f"Fetching data for {symbol}...")
            df = yank5mMarketData('2023-01-01 00:00:00', '2025-12-31 23:59:59', symbol)
            df.to_csv(path, index=False)
            dfs[symbol] = df
    
    envs = gym.vector.SyncVectorEnv([lambda: TradingEnv(dfs, window_size=100, max_steps=3072) for _ in range(NUM_ENVS)])

    history = []
    history_path = Path(f"{script_dir}/training_log.csv")

    for r in range (ROUNDS):
        model, round_stats = ppo_training_loop(envs, model, optimizer, scheduler, EPISODES_PER_ROUND)
        avg_reward = sum(s['reward'] for s in round_stats) / len(round_stats)
        avg_v_loss = sum(s['value_loss'] for s in round_stats) / len(round_stats)
        
        print(f"Round {r + 1} | Avg Reward: {avg_reward:.4f} | Value Loss: {avg_v_loss:.6f}")
        
        for stat in round_stats:
            stat['round'] = r
            history.append(stat)

        if (r + 1) % 10 == 0:
            pd.DataFrame(history).to_csv(history_path, index=False)
            torch.save(model.state_dict(), f"{script_dir}/models/TCNMLP.pt")
            print(f"Round {r + 1} completed. Model and training log data saved.")

if __name__ == "__main__":
    main()
