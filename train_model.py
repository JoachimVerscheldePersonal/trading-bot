import tensorflow as tf
from gym_trading_env.environments import TradingEnv
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import gym_trading_env
import traceback
from agent import TradingAgent
import mlflow
import warnings
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning) 
# max steps per episode
max_episode_steps=100
# total episodes
max_episodes = 1000
# discount factor
gamma = .99
# update frequency between online model and target model
tau =100
# Adam learning rate
learning_rate=0.0001

 # L2 regularization using norm 2 euclidian distance
l2_reg = 1e-6
# size of the prioritized replay buffer
replay_capacity = int(8e5)
# batch size to fetch from replay buffer
batch_size=4096
# epsilon greedy policy parameters
epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_episodes = 0.7* max_episodes
epsilon_exponential_decay = .99
actions = [-0.5-0.1,0,0.1,0.5]

tf.keras.backend.clear_session()




def sharpe_ratio(history):
    # Calculate portfolio daily returns
    portfolio_values = history["portfolio_valuation"]
    
    # Assuming portfolio_valuations is a Python list containing floats representing portfolio valuations at each timestep
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]  # Calculate daily returns
    portfolio_returns_mean = np.mean(portfolio_returns)  # Mean of daily returns
    portfolio_returns_std = np.std(portfolio_returns)    # Standard deviation of daily returns

    if portfolio_returns_std == 0:
        return 0

    hourly_risk_free_rate = 3.375e-6

    sharpe_ratio = (portfolio_returns_mean - hourly_risk_free_rate) / portfolio_returns_std
 
    return sharpe_ratio 

def test_reward_function(history):
    # Calculate portfolio daily returns
    portfolio_values = history["portfolio_valuation"]
    return (portfolio_values[-1] -  portfolio_values[-2]) / portfolio_values[-1] 

def reward_function(history):
    return 800*np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )


number_of_features = 16
window_size = 15

def calculate_validation_metrics(env: gym.Env, agent: TradingAgent, window_size: int, number_of_features:int, episode: int):
    total_return = []
    this_state = np.reshape( env.reset()[0],(window_size, number_of_features))
    
    episode_return = []
    for validaton_episode in range(10):
        for _ in range(env.spec.max_episode_steps):
            action_index = agent.predict(this_state)
            next_state, reward, done, truncated, _ = env.step(action_index)
            next_state = np.reshape(next_state,(window_size, number_of_features))
            episode_return.append(reward)
            this_state = next_state

        total_return.append(np.array(episode_return).mean())

    mean_average_return =  np.array(total_return).mean()

    mlflow.log_metric('Validation mean average return', mean_average_return)
    print(f'mean average return: {mean_average_return}')    


env = gym.make(
        "MultiDatasetTradingEnv",
        name= "BTCUSD",
        dataset_dir = 'data/train/preprocessed/*.pkl',
        windows= window_size,
        positions = actions,
        initial_position = 0, #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 10000, # in FIAT (here, USD)
        max_episode_steps=max_episode_steps,
        max_episode_duration = 100,
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
                 number_of_features=number_of_features)

total_steps = 0
def train_agent():
    for episode in tqdm(range(1, max_episodes + 1)):
            done, truncated = False, False
            this_state = np.reshape(env.reset()[0],(window_size, number_of_features))
            for episode_step in range(env.spec.max_episode_steps):
                action_index = trading_agent.epsilon_greedy_policy(this_state)
                next_state, reward, done, truncated, _ = env.step(action_index)
                next_state = np.reshape(next_state,(window_size, number_of_features))
                if not done or truncated:
                    trading_agent.memorize_transition(this_state, action_index, reward, next_state)
                    trading_agent.experience_replay()
                this_state = next_state

            trading_agent.finish_episode()
            
            if episode % 10 == 0:
                mlflow.log_metric('Training ending portfolio value', env.unwrapped.historical_info["portfolio_valuation"][-1])
                calculate_validation_metrics(env, trading_agent, window_size, number_of_features, episode)

            
    env.close()

mlflow.set_experiment("TBOT")

    # Start the run, log metrics, end the run
with mlflow.start_run() as run:
    try:
        train_agent()
    except Exception as e:
        # Log the exception traceback
        
        print(f"An error occurred: {e}")
        print(traceback.format_exc())