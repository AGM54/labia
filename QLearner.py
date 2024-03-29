import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam


class QAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.01
        self.discount_rate = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = (reward + self.discount_rate * np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#fitting function 
def check_connection(board, row, col, piece,cols,rows):
# Check horizontal connection 
    if col <= cols - 4:
        if np.count_nonzero(board[row, col:col+4] == piece) == 4:
            return True
    
    # Check vertical connection
    if row <= rows - 4:
        if np.count_nonzero(board[row:row+4, col] == piece) == 4:
            return True
    
    # Check diagonal connection (positive slope)
    if col <= cols - 4 and row <= rows - 4:
        if np.count_nonzero([board[row+i, col+i] for i in range(4)] == piece) == 4:
            return True
    
    # Check diagonal connection (negative slope)
    if col >= 3 and row <= rows - 4:
        if np.count_nonzero([board[row+i, col-i] for i in range(4)] == piece) == 4:
            return True
    
    return False