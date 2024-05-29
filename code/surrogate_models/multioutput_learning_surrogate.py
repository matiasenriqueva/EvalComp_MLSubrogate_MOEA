from skmultiflow.meta import MultiOutputLearner

from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from surrogate_models.surrogate import Surrogate



class MultiOutputLearnerSurrogate(Surrogate):
    
    def __init__(self, random_state=42 , max_iter=500, hidden_layer_sizes=50, solver='adam', learning_rate='adaptive', learning_rate_init=0.01, verbose=False, warm_start=True, activation='tanh'):
        self.model = MultiOutputLearner(base_estimator=MLPRegressor(random_state=random_state , max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes, solver=solver, learning_rate=learning_rate, learning_rate_init=learning_rate_init, verbose=verbose, warm_start=warm_start, activation=activation))
        self.scaler = StandardScaler()
        self.is_train = False
        self.previous_train = None
        self.internal_execution = 0
        self.train_counter = 0
        self.data_train = None
        '''diversity: float:total, float:valid'''
        self.diversity = list()
        self.previous_len_data=0
        self.len_data = 0
        self.len_entry_data = 0
        self.verbose = verbose

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


        '''Clean the duplicates from the data'''
        for solution in data:
           complete_data.append(solution.variables + solution.objectives)
        complete_data = pd.DataFrame(complete_data)
        no_duplicates_data = complete_data.drop_duplicates()
        if self.verbose:
            print("duplicates rows: ", complete_data.shape[0] - no_duplicates_data.shape[0])
            print("No duplicates rows: ", no_duplicates_data.shape[0])

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
            print("valid data rows: ", valid_data.shape[0])
        '''Scale the data'''
        scaling_data = self.scaler.fit_transform(valid_data)

        '''Split the data into train and test'''
        X = scaling_data[:, :n_attributes]
        y = scaling_data[:, n_attributes:]
        
        X_train = X
        y_train = y

        if not self.is_train:
            '''Fit the model with the data'''
            self.model.fit(X_train, y_train)
            self.is_train = True
        else:
            self.model.partial_fit(X_train, y_train)
        
        self.add_data(valid_data)
        self.data_train = valid_data

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

    def get_data_train(self):
        return self.data_train
    
    def get_diversity(self):
        return self.diversity
    