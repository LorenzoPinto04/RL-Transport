# SINGLE ARRAY AS INPUT

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from environment import City

import pickle
import pygame
import random
import numpy as np
import keras
from keras import Sequential
from collections import deque
from keras.layers import Dense, Conv2D, Conv3D, Flatten, MaxPooling2D, MaxPooling3D
import matplotlib.pyplot as plt
from keras.optimizers import adam
from keras.models import model_from_json



debug_mode = False
exploration = True
load_model = False


print('start')

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    img = to_grayscale(downsample(img))
    img = img.reshape((105, 80, 1))
    return img


def transform_reward(reward):
    return np.sign(reward)

# Import the gym module
import gym

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame
frame = env.reset()
print('[INFO] Shape input:', preprocess(frame).shape)
np.random.seed(0)




class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space, debug_mode = False, load_model = True):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        # discount value 
        # 0 for present and 1 for future
        self.gamma = .2
        self.batch_size = 64
        
        # epsilon denotes the fraction of time we will dedicate to exploring
        #self.epsilon_min = .01
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.01
        self.memory = deque(maxlen=100000)
        if load_model:
            print("[INFO] Loading model from disk")
            # load json and create model
            json_file = open('models/CNN_atari.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("models/CNN_atari.h5")
            self.model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
            print("[INFO] Model loaded")
            return
        #self.model = self.atari_model()
        self.model = self.build_model_conv()

    
    def build_model_conv(self):
        model = Sequential()
        model.add(Conv3D(16, (2, 8, 8),  activation='relu', 
                         input_shape=(2, self.state_space[0], self.state_space[1], 1)))
        #model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Conv3D(32, (1, 4, 4), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(4, 4)))
        #model.add(Conv2D(64, (1, 1), activation='relu', 
        #                 input_shape=(self.state_space[0], self.state_space[1], 3)))
        #model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(4))
        model.compile(loss="mse",
                           optimizer=adam(lr=self.learning_rate))
        return model
    
 
    
    
    
    
    
    
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if debug_mode:    
            print('State: \n', state[0].reshape(y_size, x_size))
            print('action: ', action)
            print('Next state: \n', next_state[0].reshape(y_size, x_size))
            print('reward: ', reward)
            print('------------------------------------------------')

    def act(self, state):
        # if the random float is smaller than epsilon reduced, it takes a random action (explore)
        if np.random.rand() <= self.epsilon and exploration:
            #print('Exploration step')
            return random.randrange(self.action_space)
        # else exploit
        state = state.reshape((1, 2, 105, 80, 1))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        '''
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('States -------------------------------------------------------------------------')
        plt.imshow(states[0][0].squeeze(axis=2))
        plt.show()
        plt.imshow(states[0][1].squeeze(axis=2))
        plt.show()
        
        print('Next states -------------------------------------------------------------------------')
        plt.imshow(next_states[0][0].squeeze(axis=2))
        plt.show()
        plt.imshow(next_states[0][1].squeeze(axis=2))
        plt.show()
        
        '''
        
        # First, predict the Q values of the next states. Note how we are passing ones as the mask.
        next_Q_values = self.model.predict(next_states)
        # The Q values of the terminal states is 0 by definition, so override them
        next_Q_values[dones] = 0
        # The Q values of each start state is the reward + gamma * the max next state Q value
        targets = rewards + self.gamma * np.max(next_Q_values, axis=1)
        targets_full = self.model.predict_on_batch(states)
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        # every new iteration reduce epsilon to push the exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return self.model
    
    

def train_dqn(episode):
    print(episode)
    loss = []
    agent = DQN(4, (105, 80), debug_mode, load_model)
    for e in range(episode):
        state = (preprocess(env.reset()))
        set_states = np.array([state, state])
        score = 0
        max_steps = 10000
        
        set_states_past = np.array([state, state])
        set_states_fut = np.array([state])
        
        
        for i in range(max_steps):
            action = agent.act(set_states)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            
            set_states_fut = np.array([set_states_fut[-1], next_state])
            
            score += reward
            
            agent.remember(set_states_past, action, reward, set_states_fut, done)
            state = next_state

            set_states_past = np.array([set_states_past[-1], next_state])

            
            model = agent.replay()
            #env.render()
            if done:
                break
        loss.append(score)
        print("episode: {}/{}, moves:{}, score: {}".format(e, episode, i, str(score)[:4]))
        if (e+1) % 1000 == 0:
            print('[INFO] Saving checkpoint iter:', e)
            # serialize model to JSON
            model_json = model.to_json()
            with open("models/CNN_atari.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("models/CNN_atari.h5")
            print("[INFO] Saved model to disk")
            # with open(r"models/model_auto6.pickle", "wb") as f:
            #    pickle.dump(agent, f)
            plt.figure(figsize=(20,10))
            plt.plot([i for i in range(e)], loss[-e:])
            plt.xlabel('episodes')
            plt.ylabel('reward')
            #plt.savefig('training_graph_check{}'.format(e))
            plt.show()
    return loss

ep = 100000
loss = train_dqn(ep)


















