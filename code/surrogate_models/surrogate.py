# We define the surrogate class, which is the parent class of all surrogate models.

class Surrogate:
    is_train = False
    previous_train = None
    sample_x = None
    sample_y = None
    internal_execution = 0
    train_counter = 0
    
    def __init__(self):
        pass

    def evaluate(self, data):
        pass
            

    def fit(self, data):
        pass

        
    def add_data(self, data):
        pass

    def get_internal_execution(self):
        pass

    def add_sample_data(self, X, Y):
        pass

    def get_name(self):
        pass
    