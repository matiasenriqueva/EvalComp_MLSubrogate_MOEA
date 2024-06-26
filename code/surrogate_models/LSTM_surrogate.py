from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from surrogate_models.surrogate import Surrogate



class LSTMSurrogate(Surrogate):

    def __init__(self, dense_layers=64, activation='tanh', optimizer='adam', loss='mse', epochs=20, batch_size=10, verbose=True):
        self.model = Sequential()
        self.scaler = StandardScaler()
        self.dense_layers = dense_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.is_train = False
        self.previous_train = None
        self.internal_execution = 0
        self.train_counter = 0
        '''diversity: float:total, float:valid'''
        self.diversity = list()
        self.previous_len_data=0
        self.len_data = 0
        self.len_entry_data = 0

    def evaluate(self, data):
        '''Evaluate the regressor chain with the given data'''
        n_attributes = len(data[0].variables)
        complete_data = list()
        for solution in data:
           complete_data.append(solution.variables + solution.objectives)
        complete_data = self.scaler.transform(complete_data)
        X = complete_data[:, :n_attributes]
        y = complete_data[:, n_attributes:]
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(np.hstack((X, predictions)))[:, n_attributes:]
        

        for index, item in enumerate(data):
            item.objectives = predictions[index].tolist()     
        

        self.internal_execution += 1
        return data
            

    def fit(self, data):
        if self.verbose:
            if not self.is_train: print("Training algorithm ") 
            else: print("Partial training algorithm")
            '''Initialize the multioutput learner with data'''
        complete_data = list()
        n_attributes = len(data[0].variables)

        '''Check total amount of new data'''
        if self.len_data == 0:
            self.len_data = len(data)
        else:
            self.previous_len_data = self.len_data
            self.len_data = len(data)
        
        self.len_entry_data = self.len_data - self.previous_len_data

        '''LSTM initial variables'''
        n_features = len(data[0].variables) # Número de características por paso temporal
        n_outputs = len(data[0].objectives)     # Número de valores de salida

        '''initialize the model'''

        self.model.add(Dense(self.dense_layers, activation=self.activation, input_shape=(n_features,)))
        self.model.add(Dense(self.dense_layers, activation=self.activation))
        self.model.add(Dense(n_outputs))  # 2 nodos de salida para tus objetivos
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        if self.verbose:
            self.model.summary()


        '''Clean the duplicates from the data'''
        for solution in data:
           complete_data.append(solution.variables + solution.objectives)
        complete_data = pd.DataFrame(complete_data)
        no_duplicates_data = complete_data.drop_duplicates()
        if self.verbose:
            print("total data rows: ", complete_data.shape[0])
            print("duplicates rows: ", complete_data.shape[0] - no_duplicates_data.shape[0])

        '''Add the actual data to previous train for not repeat the data'''
        if self.previous_train is not None:
            if self.verbose:
                print("previous ", len(self.previous_train))
                print("no_duplicates ", len(no_duplicates_data))
            valid_data = no_duplicates_data.merge(self.previous_train, how='left', indicator=True)
            valid_data = valid_data[valid_data['_merge'] == 'left_only'].drop(columns='_merge')            
        else:
            valid_data = no_duplicates_data

        if self.verbose:
            print("valida data: ", valid_data.shape[0])

        '''Scale the data'''
        scaling_data = self.scaler.fit_transform(valid_data)

        '''Split the data into train and test'''
        X = scaling_data[:, :n_attributes]
        y = scaling_data[:, n_attributes:]
        
        X_train = X
        y_train = y

        if self.verbose:
            verb = 1
        else:
            verb = 0

        self.model.fit(X_train, y_train, epochs=self.epochs, verbose=verb, batch_size=self.batch_size)

        
        self.add_data(valid_data)

        self.train_counter += 1

        self.diversity.append([self.len_entry_data, valid_data.shape[0]])

        
    def add_data(self, data):
        '''Add data to the regressor chain'''
        if self.previous_train is None:
            self.previous_train = data
        else:
            self.previous_train = pd.concat([self.previous_train, data])

    def get_internal_execution(self):
        return self.internal_execution

    def add_sample_data(self, X, Y):
        if self.sample_x is None:
            self.sample_x = pd.DataFrame(X)
            self.sample_y = pd.DataFrame(Y)
        else:
            self.sample_x = pd.concat([self.sample_x, pd.DataFrame(X)])
            self.sample_y = pd.concat([self.sample_y, pd.DataFrame(Y)])
    
    def get_diversity(self):
        return self.diversity
        
    