from functions import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import cv2
from matplotlib import pyplot as plt
import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
from environment import GridWorld


def OurModel(input_shape, action_space, dueling):
    X_input = Input(input_shape)
    X = X_input
    
    #X = Conv2D(64, 5, strides=(3, 3),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_first")(X)
    X = Conv2D(32, 8, strides=(4, 4),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_first")(X)
    X = Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_first")(X)
    X = Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_first")(X)
    X = Flatten()(X)
    
    # 'Dense' is the basic form of a neural network layer
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X)
    #model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.compile(optimizer=Adam(lr=0.00025), loss='mean_squared_error')
    #model.compile(optimizer=Adam(lr=0.00005), loss='mean_squared_error')

    model.summary()
    return model

class DQNAgent:
    def __init__(self, env_name, env):
        self.env_name = env_name       
        self.env = env
        self.action_size = 8
        self.EPISODES = 1000000
        
        # Instantiate memory
        memory_size = 25000
        self.MEMORY = Memory(memory_size)
        self.memory = deque(maxlen=memory_size)

        self.gamma = 0.99    # discount rate
        
        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.02  # minimum exploration probability 
        self.epsilon_decay = 0.00002  # exponential decay rate for exploration prob
        
        self.batch_size = 32

        # defining model parameters
        self.ddqn = False # use doudle deep q network
        self.dueling = False # use dealing netowrk
        self.epsilon_greedy = False # use epsilon greedy strategy
        self.USE_PER = False # use priority experienced replay
        

        self.Save_Path = 'models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        self.Model_name = os.path.join(self.Save_Path, self.env_name+"_CNN.h5")

        self.ROWS = 50
        self.COLS = 80
        self.REM_STEP = 4
        self.update_model_steps = 1000
        
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)
        self.image_memory = np.zeros(self.state_size)
        
        # create main model and target model
        self.model = OurModel(input_shape=self.state_size, action_space = self.action_size, dueling = self.dueling)
        self.target_model = OurModel(input_shape=self.state_size, action_space = self.action_size, dueling = self.dueling)  

    # after some time interval update the target model to be same with model
    def update_target_model(self, game_steps):
        if game_steps % self.update_model_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
            return

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state, decay_step):
        # EPSILON GREEDY STRATEGY
        if self.epsilon_greedy:
        # Here we'll use an improved version of our epsilon greedy strategy for Q-learning
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)
        # OLD EPSILON STRATEGY
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= (1-self.epsilon_decay)
            explore_probability = self.epsilon
    
        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state)), explore_probability
                
    def replay(self):
        if self.USE_PER:
            # Sample minibatch from the PER memory
            tree_idx, minibatch  = self.MEMORY.sample(self.batch_size)
        else:
            if len(self.memory) > self.batch_size:
            # Randomly sample minibatch from the deque memory
                minibatch = random.sample(self.memory, self.batch_size)
            else:
                return

        state = np.zeros((self.batch_size, *self.state_size), dtype=np.float32)
        action = np.zeros(self.batch_size, dtype=np.int32)
        reward = np.zeros(self.batch_size, dtype=np.float32)
        next_state = np.zeros((self.batch_size, *self.state_size), dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=np.uint8)

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop       
        for i in range(len(minibatch)):
            state[i], action[i], reward[i], next_state[i], done[i] = minibatch[i]

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        target_old = np.array(target)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)


        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                if self.ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * target_val[i][a]
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    # when using target model in simple DQN rules, we get better performance
                    target[i][action[i]] = reward[i] + self.gamma * np.amax(target_val[i])
            
        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, action]-target[indices, action])

            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)
                
        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model.load_weights(name, by_name=False)

    def save(self, name):
        model = self.model
        model.save_weights(name)

    pylab.figure(figsize=(18, 9))
    plt.figure(figsize=(18,9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Games', fontsize=18)
        dqn = 'DQN_'
        dueling = ''
        greedy = ''
        PER = ''
        if self.ddqn: dqn = '_DDQN'
        if self.dueling: dueling = '_Dueling'
        if self.epsilon_greedy: greedy = '_Greedy'
        if self.USE_PER: PER = '_PER'
        try:
            if not os.path.exists('training_images'): os.makedirs('training_images')
            pylab.savefig('training_images/'+self.env_name+dqn+dueling+greedy+PER+"_CNN.png")
            
        except OSError as e:
            pass
        # no need to worry about model, when doing a lot of experiments
        self.Model_name = os.path.join(self.Save_Path, self.env_name+dqn+dueling+greedy+PER+"_CNN.h5")

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow("cartpole"+str(rem_step), image[rem_step,...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self, frame):

        # push our data by 1 frame, similar as deq() function work
        self.image_memory = np.roll(self.image_memory, 1, axis = 0)

        # inserting new frame to free space
        self.image_memory[0,:,:] = frame

        # show image frame   
        #self.imshow(self.image_memory,0)
        #self.imshow(self.image_memory,1)
        #self.imshow(self.image_memory,2)
        #self.imshow(self.image_memory,3)
        
        return np.expand_dims(self.image_memory, axis=0)

    def reset(self):
        frame = self.env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage(frame)
        return state

    def step(self,action):
        info = None
        state, reward, total_reward, next_state, done = self.env.step(action)
        next_state = self.GetImage(next_state)
        return next_state, reward, done, info
    
    def run(self):
        decay_step = 0
        max_average = -21.0
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            score = 0
            SAVING = ''
            while not done:
                decay_step += 1
                action, explore_probability = self.act(state, decay_step)
                next_state, reward, done, _ = self.step(action)
                '''
                if reward != 0.0:
                    print('States -------------------------------------------------------------------------')
                    plt.imshow(state[0][0])
                    plt.show()
                    plt.imshow(state[0][1])
                    plt.show()
                    plt.imshow(state[0][2])
                    plt.show()
                    plt.imshow(state[0][3])
                    plt.show()
                    print('Action:', action)
                    print('Reward:', reward)
                    '''
                self.remember(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    # every episode, plot the result
                    average = self.PlotModel(score, e)

                    # saving best models
                    if average >= max_average:
                        max_average = average
                        self.save(self.Model_name)
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, e: {:.2f}, average: {:.2f} {}".format(e, self.EPISODES, score, explore_probability, average, SAVING))
                    
                # update target model
                self.update_target_model(decay_step)

                # train model
                self.replay()

        # close environemnt when finish training
        #self.env.close()

    def test(self, Model_name):
        import time
        self.load(Model_name)
        for e in range(self.EPISODES):
            state = self.reset()
            done = False
            score = 0
            while not done:
                time.sleep(.01)
                action = np.argmax(self.model.predict(state))
                state, reward, total_reward, next_state, done = self.env.step(action)
                state = self.GetImage(next_state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        #self.env.close()

debug_mode = False
show_graph_every = False
means = True

env = GridWorld(show_graph_every, debug_mode, means)


if __name__ == "__main__":
    env_name = 'GridWorldMeans'
    agent = DQNAgent(env_name, env)
    agent.run()
    #agent.test('models/GridWorldDQN__CNN.h5')