import tensorflow as tf
from gym_trading_env.environments import TradingEnv
import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
import gym_trading_env
import traceback
from agent import TradingAgent
import mlflow
import warnings
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import os

starting_capital = 10000
# total episodes
max_episodes = 100
# discount factor
gamma = .99
# update frequency between online model and target model
tau =100
# Adam learning rate
learning_rate=0.0001
 # L2 regularization using norm 2 euclidian distance
l2_reg = 1e-6
# size of the prioritized replay buffer
replay_capacity = int(1e5)
# batch size to fetch from replay buffer
batch_size=4096
# epsilon greedy policy parameters
epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_episodes = 0.8* max_episodes
epsilon_exponential_decay = .99
actions = [-0.5-0.1,0,0.1,0.5]

tf.keras.backend.clear_session()
mlflow.set_experiment("TBOT")

def reward_function(history):
    return 800*np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

number_of_features = 16
window_size = 15
state_dimensions = 95

def get_file_paths_for_folder(folder_path):
    
    # Get the list of files in the folder
    files = os.listdir(folder_path)
    return [os.path.join(folder_path, file) for file in files]

def test_agent(environment: gym.Env, agent: TradingAgent, test_df_file_path: int, portfolio_df: pd.DataFrame):
    dataframe_row = [test_df_file_path]
    done, truncated = False, False
    this_state = np.reshape(environment.reset()[0],(window_size, number_of_features))
    while not done and not truncated:
        action_index = agent.predict(this_state)
        next_state, reward, done, truncated, info = environment.step(action_index)
        next_state = np.reshape(next_state,(window_size, number_of_features))
        this_state = next_state

    environment.unwrapped.save_for_render(dir="outputs")
    model_return = env.unwrapped.historical_info["portfolio_valuation"][-1]/starting_capital

    
    done, truncated = False, False
    this_state = np.reshape(environment.reset()[0],(window_size, number_of_features))
    while not done and not truncated:
        action_index = actions.index(0.1)
        next_state, reward, done, truncated, info = environment.step(action_index)
        next_state = np.reshape(next_state,(window_size, number_of_features))
        this_state = next_state

    buy_and_hold_return = env.unwrapped.historical_info["portfolio_valuation"][-1]/starting_capital

    mlflow.log_metric('Model return', model_return)
    mlflow.log_metric('Buy and hold return', buy_and_hold_return)
    dataframe_row.append(model_return)
    dataframe_row.append(buy_and_hold_return)
    
    print(dataframe_row)
    portfolio_df.loc[len(portfolio_df)] = dataframe_row
    environment.close()
    
    del environment
    del agent

portfolio_ending_df = pd.DataFrame(columns=['file','model_return', 'buy_and_hold_return'])

test_df_paths = get_file_paths_for_folder("data/test/preprocessed")
for test_df_path in test_df_paths:
    print(f'testing {test_df_path}')
    df = pd.read_pickle(test_df_path)
    checkpoint_file_path = "checkpoints/episode-{episode:04d}/checkpoint"

    env = gym.make(
        "TradingEnv",
        name= "BTCUSD",
        df = df,
        windows= window_size,
        positions = actions,
        initial_position = 0, #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = starting_capital, # in FIAT (here, USD)
        verbose=1,
    )

    num_actions=env.action_space.n

    trading_agent = TradingAgent(
                    num_actions=num_actions,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay_episodes=epsilon_decay_episodes,
                    epsilon_exponential_decay=epsilon_exponential_decay,
                    replay_capacity=replay_capacity,
                    l2_reg=l2_reg,
                    tau=tau,
                    batch_size=batch_size,
                    window_size=window_size,
                    number_of_features=number_of_features,
                    weights_path=checkpoint_file_path.format(episode=330))

    test_agent(env, trading_agent, test_df_path, portfolio_ending_df)


portfolio_ending_df.to_csv('outputs/test_results.csv')


