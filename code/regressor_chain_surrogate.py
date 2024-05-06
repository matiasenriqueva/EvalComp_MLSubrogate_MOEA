from sklearn.linear_model import SGDRegressor
from skmultiflow.meta import RegressorChain

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



class RegressorChainSurrogate():
    is_train = False
    previous_train = None
    internal_execution = 0
    
    def __init__(self):
        self.rc = RegressorChain(SGDRegressor(loss='squared_error', random_state=1))
        self.scaler = StandardScaler()

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
        if not self.is_train: print("Training algorithm ") 
        else: print("Partial training algorithm")
        '''Initialize the regressor chain with data'''
        complete_data = list()
        n_attributes = len(data[0].variables)

        '''Clean the duplicates from the data'''
        for solution in data:
           complete_data.append(solution.variables + solution.objectives)
        complete_data = pd.DataFrame(complete_data)
        no_duplicates_data = complete_data.drop_duplicates()
        print("duplicates rows: ", complete_data.shape[0] - no_duplicates_data.shape[0])

        '''Add the actual data to previous train for not repeat the data'''
        if self.previous_train is not None:
            print("previous ", self.previous_train)
            print("no_duplicates ", no_duplicates_data)
            valid_data = self.previous_train.merge(no_duplicates_data, how='left', indicator=True)
            print(valid_data)
            valid_data = valid_data[valid_data['_merge'] == 'left_only'].drop(columns='_merge')
            print(valid_data)
        else:
            valid_data = no_duplicates_data

        '''Scale the data'''
        scaling_data = self.scaler.fit_transform(valid_data)

        '''Split the data into train and test'''
        X = scaling_data[:, :n_attributes]
        y = scaling_data[:, n_attributes:]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        if not self.is_train:
            '''Fit the regressor chain'''
            self.rc.fit(X_train, y_train)
            self.is_train = True
        else:
            self.rc.partial_fit(X_train, y_train)
        
        mse = mean_squared_error(y_test, self.rc.predict(X_test))
        print("MSE evaluation train: ", mse) 

        self.add_data(valid_data)       

        
    def add_data(self, data):
        '''Add data to the regressor chain'''
        if self.previous_train is None:
            self.previous_train = data
        else:
            self.previous_train = pd.concat([self.previous_train, data])

    def get_internal_execution(self):
        return self.internal_execution
    