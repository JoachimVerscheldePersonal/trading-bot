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

starting_capital = 10000
# max steps per episode
max_episode_steps=100
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
actions = [-0.1,0,0.1]

tf.keras.backend.clear_session()
mlflow.set_experiment("TBOT")

def reward_function(history):
    return 800*np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )

number_of_features = 16
window_size = 15
state_dimensions = 95

def test_agent(environment: gym.Env, agent: TradingAgent, weight_episode: int, portfolio_df: pd.DataFrame):
    dataframe_row = [weight_episode]
    profits = []
    for episode in range(100):
        this_state =  np.reshape(environment.reset()[0],(window_size, number_of_features))
        for episode_step in range(environment.spec.max_episode_steps):
            action_index = agent.predict(this_state)
            next_state, _, _, _, _ = environment.step(action_index)
            next_state = np.reshape(next_state,(window_size, number_of_features))
            this_state = next_state

        environment.unwrapped.save_for_render(dir="render_logs/validation-episode-{episode:04d}".format(episode=weights_episode))
        profits.append(env.unwrapped.historical_info["portfolio_valuation"][-1]-starting_capital)

    total_profit = np.array(profits).sum()
    profit_std = np.array(profits).std()
    mlflow.log_metric('Total profit', total_profit)
    mlflow.log_metric('profit standard deviation', profit_std)
    dataframe_row.append(total_profit)
    dataframe_row.append(profit_std)
    
    print(dataframe_row)
    portfolio_df.loc[len(portfolio_df)] = dataframe_row
    environment.close()
    
    del environment
    del agent

portfolio_ending_df = pd.DataFrame(columns=['episode','total_profit', 'profit_deviation'])
episodes_to_validate = range(0,620,10)

for weights_episode in tqdm(episodes_to_validate):
    
    checkpoint_file_path = "checkpoints/episode-{episode:04d}/checkpoint"
    env = gym.make(
        "MultiDatasetTradingEnv",
        name= "BTCUSD",
        dataset_dir = 'data/validation/*.pkl',
        windows= window_size,
        positions = actions,
        initial_position = 0, #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = starting_capital, # in FIAT (here, USD)
        max_episode_steps=max_episode_steps,
        episodes_between_dataset_switch = 1,
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
                    weights_path=checkpoint_file_path.format(episode=weights_episode))

    test_agent(env, trading_agent, weights_episode, portfolio_ending_df)


portfolio_ending_df.to_csv('outputs/test_results.csv')


