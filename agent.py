import tensorflow as tf
from collections import deque
import numpy as np
from random import sample

class TradingAgent:
    def __init__(self, state_dimensions,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 l2_reg,
                 tau,
                 batch_size):
        
        self.state_dimensions = state_dimensions
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.l2_reg = l2_reg

        # double DQN 2 models, 1 for predicting the values, another for defining the targets
        self.online_model = self.build_model()
        self.target_model = self.build_model(trainable=False)
        self.update_target_model()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

        self.tensorboard = self.setup_tensorboard()
        
    def build_model(self, trainable = True):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=265, input_dim=self.state_dimensions, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg), name=f'Dense_1', trainable=trainable),
            tf.keras.layers.Dense(units=265, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(self.l2_reg), name=f'Dense_2', trainable=trainable),
            tf.keras.layers.Dropout(.1),
            tf.keras.layers.Dense(units=self.num_actions, trainable=trainable, name='Output')
        ])

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def setup_tensorboard(self):
        
        tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='/tensorboard',
        histogram_freq=0,
        batch_size=self.batch_size,
        write_graph=True,
        write_grads=True
        )

        tensorboard.set_model(self.target_model)
        return tensorboard

    def named_logs(self, model, logs):
        # Transform train_on_batch return value
        # to dict expected by on_batch_end callback
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def update_target_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def update_models_from_checkpoint(self):
        self.online_model.load_weights('./checkpoints/my_checkpoint')
        self.target_model.load_weights('./checkpoints/my_checkpoint')
    

    def epsilon_greedy_policy(self, state):
        self.total_steps +=1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        
        q_values = self.online_model.predict(state)
        return np.argmax(q_values, axis=1).squeeze()

    def experience_replay(self, episode_step: int):
        if self.batch_size > len(self.experience):
            return
        
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        print(minibatch)
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_model.predict_on_batch(next_states.reshape(-1,self.state_dimensions))
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_model.predict_on_batch(next_states.reshape(-1,self.state_dimensions))
        target_q_values = tf.gather_nd(next_q_values_target, tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_model.predict_on_batch(states).reshape(-1, self.num_actions)
        q_values[self.idx, actions] = targets

        loss = self.online_model.train_on_batch(x=states.reshape(-1, self.state_dimensions), y=q_values)
        self.tensorboard.on_epoch_end(episode_step, self.named_logs(self.target_model, [loss]))
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.target_model.save_weights('checkpoints/my_checkpoint')
            self.update_target_model()
    
    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

        