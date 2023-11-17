import tensorflow as tf
import os
from collections import deque
import numpy as np
from random import sample
import mlflow
import pickle


class TradingAgent:
    def __init__(self, 
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_episodes,
                 epsilon_exponential_decay,
                 replay_capacity,
                 l2_reg,
                 tau,
                 batch_size,
                 window_size,
                 number_of_features,
                 weights_path: str = None):
        
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.window_size =window_size
        self.number_of_features = number_of_features

        # double DQN 2 models, 1 for predicting the values, another for defining the targets
        self.online_model = self.build_model()
        self.target_model = self.build_model(trainable=False)

        self.epsilon = epsilon_start
        self.eplison_decay_episodes = epsilon_decay_episodes
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes
        self.epsilon_exponential_decay = epsilon_exponential_decay

        self.total_steps = 0
        self.current_episode = self.episode_length  = 0
        self.episode_rewards = []

        self.tau = tau
        self.idx = tf.range(batch_size)
        self.losses = []

        if weights_path:
            self.load_model_weights(weights_path)
       
        
    def load_model_weights(self, weights_path):
        
        self.online_model.load_weights(weights_path).expect_partial()

    
    def build_model(self, trainable = True):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=256, input_shape=(self.window_size, self.number_of_features), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg), name=f'Dense_1',return_sequences=True, trainable=True),
            tf.keras.layers.LSTM(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg), name=f'Dense_2', trainable=True),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.Dense(units=self.num_actions, trainable=True, name='Output')
        ])

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    

    def named_logs(self, model, logs):
        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())
    
    def epsilon_greedy_policy(self, state):
        self.total_steps +=1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        
        return self.predict(state)

    def predict(self, state):
        q_values = self.online_model.predict(np.reshape(state,(1, self.window_size, self.number_of_features)), verbose=0)
        return np.argmax(q_values, axis=1).squeeze()

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states = minibatch

        next_q_values = self.online_model.predict_on_batch(next_states.reshape(self.batch_size, self.window_size , self.number_of_features))
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_model.predict_on_batch(next_states.reshape(self.batch_size, self.window_size , self.number_of_features))
        target_q_values = tf.gather_nd(next_q_values_target, tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + 1 * self.gamma * target_q_values

        q_values = self.online_model.predict_on_batch(states).reshape(-1, self.num_actions)
        q_values[self.idx, actions] = targets

        loss = self.online_model.train_on_batch(x=states.reshape(self.batch_size, self.window_size , self.number_of_features), y=q_values)
        self.losses.append(loss)
        if self.total_steps % self.tau == 0:
            self.update_target_model()
    
    def finish_episode(self):
        if self.current_episode < self.eplison_decay_episodes:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_exponential_decay
        
        

        if self.current_episode % 10==0:
            mlflow.log_metric('Epsilon', self.epsilon)
            
            if len(self.losses):
                mlflow.log_metric('Training MSE', np.array(self.losses).mean())

            episode_folder_path = "./outputs/episode-{episode:04d}".format(episode=self.current_episode)
            if not os.path.exists(episode_folder_path):
                os.makedirs(episode_folder_path)

            checkpoint_path = os.path.join(episode_folder_path, 'model.keras')
            self.online_model.save_weights(checkpoint_path)
        
        self.current_episode += 1
        self.losses = []
        self.episode_rewards = []
        self.episode_length =  0

    def memorize_transition(self, s, a, r, s_prime):
        self.episode_rewards.append(r)
        self.episode_length += 1

        self.experience.append((s, a, r, s_prime))
        