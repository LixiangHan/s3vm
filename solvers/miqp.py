from docplex.mp.model import Model
import numpy as np
import pickle
import time

class MIQPS3VM:
    def __init__(self, time_limit=100):
        self.model = None
        self.weights = None
        self.bias = None
        self.w = None
        self.b = None
        self.eta = None
        self.epsilon = None
        self.z = None
        self.d = None
        self.time_limit=time_limit

    def fit(self, label_x, label_y, unlabel_x, C, M, logout=False):
        self.model = Model(name='miqps3vm', log_output=logout)
        self.model.set_time_limit(self.time_limit)
        self.w = [ self.model.continuous_var(name='w_%d' % i) for i in range(label_x.shape[1]) ]
        self.b = self.model.continuous_var(name='b')
        self.eta = [ self.model.continuous_var(name='eta_%d' % i, lb=0) for i in range(label_x.shape[0]) ]
        self.epsilon = [ self.model.continuous_var(name='epsilon_%d' % i, lb=0) for i in range(unlabel_x.shape[0]) ]
        self.z = [ self.model.continuous_var(name='z_%d' % i, lb=0) for i in range(unlabel_x.shape[0]) ]
        self.d = [ self.model.binary_var(name='d_%d' % i) for i in range(unlabel_x.shape[0]) ]
        for x, y, eta in zip(label_x, label_y, self.eta):
            self.model.add_constraint(y * (self.model.sum([x[i] * self.w[i] for i in range(len(x))]) - self.b) + eta >= 1)
        
        for x, epsilon, z, d in zip(unlabel_x, self.epsilon, self.z, self.d):
            self.model.add_constraint(self.model.sum([x[i] * self.w[i] for i in range(len(x))]) - self.b + epsilon + M * (1 - d) >= 1)
            self.model.add_constraint(-(self.model.sum([x[i] * self.w[i] for i in range(len(x))]) - self.b) + z + M * d >= 1)
        
        self.model.set_objective('min', C * (self.model.sum(self.eta) + self.model.sum([self.epsilon[i] + self.z[i] for i in range(unlabel_x.shape[0])])) + sum(self.w[i] * self.w[i] for i in range(label_x.shape[1])))

        if logout:
            self.model.print_information()


        time_start = time.time()
        self.model.solve()
        time_end = time.time()
        

        if logout:
            self.model.print_solution()
        self.weights = np.array([ w_i.solution_value for w_i in self.w ])
        self.bias = self.b.solution_value
        return time_end - time_start

    def predict(self, x):
        if type(self.weights) == type(None) or type(self.bias) == type(None):
            print('Model have not been trained.')
        else:
            pred = np.matmul(x, self.weights) - self.bias
            pred = pred.flatten()
            return np.sign(pred)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
