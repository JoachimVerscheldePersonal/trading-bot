import tensorflow as tf
from gym_trading_env.environments import TradingEnv
import gymnasium as gym
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import gym_trading_env
from time import time
import mlflow

from data_provider import MarketDataProvider
from agent import TradingAgent

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

mlflow.set_tracking_uri("azureml://westeurope.api.azureml.ms/mlflow/v1.0/subscriptions/b98e4480-8f42-4f17-ae4a-d9b2dd9df01a/resourceGroups/Personal/providers/Microsoft.MachineLearningServices/workspaces/Personal_WorkSpace")
tf.get_logger().setLevel('INFO')

data_provider = MarketDataProvider(exchange_name="binance", symbol="BTC/USDT", timeframe="1h", data_directory="data", since=datetime(year= 2020, month= 1, day=1))
df = data_provider.fetch_data()

# discount factor
gamma = .99
# update frequency between online model and target model
tau =100
# Adam learning reate
learning_rate=0.0001
 # L2 regularization using norm 2 euclidian distance
l2_reg = 1e-6
# size of the prioritized replay buffer
replay_capacity = int(1e6)
# batch size to fetch from replay buffer
batch_size=4096
# epsilon greedy policy parameters 
epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_steps = 250
epsilon_exponential_decay = .99

tf.keras.backend.clear_session()

risk_free_rate = 0.03
def reward_function(history):
    # Calculate portfolio daily returns
    portfolio_values = history["portfolio_valuation"]
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Calculate excess daily returns over risk-free rate
    excess_returns = daily_returns - risk_free_rate
    
    # Calculate the mean and standard deviation of excess returns
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns)
    
    # Calculate Sharpe ratio
    if std_dev_excess_return != 0:
        sharpe_ratio = mean_excess_return / std_dev_excess_return
    else:
        sharpe_ratio = 0
    
    return sharpe_ratio

env = gym.make(
        "TradingEnv",
        name= "BTCUSD",
        df = df,
        windows= 5,
        positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
        initial_position = 'random', #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 1000, # in FIAT (here, USD)
        max_episode_duration = 500,
        max_episode_steps=500
    )

state_dimensions = 85
num_actions=env.action_space.n

trading_agent = TradingAgent(state_dimensions=state_dimensions,
                 num_actions=num_actions,
                 learning_rate=learning_rate,
                 gamma=gamma,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                 epsilon_decay_steps=epsilon_decay_steps,
                 epsilon_exponential_decay=epsilon_exponential_decay,
                 replay_capacity=replay_capacity,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size,)

total_steps = 0
max_episodes = 1000

start = time()
results = []

for episode in tqdm(range(1, max_episodes + 1)):
    this_state = env.reset()[0].reshape(1,-1)
    for episode_step in range(env.spec.max_episode_steps):
        action = trading_agent.epsilon_greedy_policy(this_state.reshape(1, -1))
        next_state, reward, done, truncated, info = env.step(action)

        trading_agent.memorize_transition(this_state, 
                                 action, 
                                 reward, 
                                 next_state.reshape(1,-1), 
                                 0.0 if done | truncated else 1.0)
        if trading_agent.train:
            trading_agent.experience_replay(episode_step)
        if done:
            env.unwrapped.save_for_render(dir="render_logs")
            break
        this_state = next_state.reshape(1,-1)

    
env.close()