from sklearn.linear_model import SGDRegressor
from skmultiflow.meta import RegressorChain

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from surrogate_models.surrogate import Surrogate



class RegressorChainSurrogate(Surrogate):
       
    
    def __init__(self, verbose=False):
        self.rc = RegressorChain(SGDRegressor(loss='squared_error', random_state=1))
        self.scaler = StandardScaler()
        self.is_train = False
        self.previous_train = None
        self.internal_execution = 0
        self.train_counter = 0
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
        predictions = self.rc.predict(X)
        predictions = self.scaler.inverse_transform(np.hstack((X, predictions)))[:, n_attributes:]
        

        for index, item in enumerate(data):
            item.objectives = predictions[index].tolist()     
        

        self.internal_execution += 1
        return data
            

    def fit(self, data):
        if self.verbose:
            if not self.is_train: print("Training algorithm ") 
            else: print("Partial training algorithm")
        '''Initialize the regressor chain with data'''
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
        
        if self.train_counter <= 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            if self.train_counter == 0:
                self.sx1 = X_test
                self.sy1 = y_test
            elif self.train_counter == 1:
                self.sx2 = X_test
                self.sy2 = y_test
        else:
            X_train = X
            y_train = y


        if not self.is_train:
            '''Fit the regressor chain'''
            self.rc.fit(X_train, y_train)
            self.is_train = True
        else:
            self.rc.partial_fit(X_train, y_train)
                
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