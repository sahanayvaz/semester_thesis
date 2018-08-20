import numpy as np
import pickle

class Scaler(object):
    def __init__(self, obs_dim):
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True
        self.obs_dim = obs_dim

    def update(self, x):
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            
            # before update save previous means and std for some functions
            self.vars_prev = self.vars 
            self.means_prev = self.means

            # update
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))

            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    def get(self):
        if self.first_pass:
            return np.ones(self.obs_dim), np.zeros(self.obs_dim)
        else:
            if np.isnan(np.sqrt(self.vars)).any():
                idx = np.argwhere(np.isnan(np.sqrt(self.vars)))
                print(self.vars[idx])
                raise ValueError('NaNs')
            else:
                return 1 / (np.sqrt(self.vars) + 0.1), self.means

    def get_previous(self):
        return self.vars_prev, self.means_prev
        
class Logger(object):
    def __init__(self):
        self.logger = {}

    def log(self, log_name, log_cont):
        if log_name in self.logger:
            self.logger[log_name].append(log_cont)
        else:
            self.logger[log_name] = [log_cont]

    def dump(self, filename):
        fileobject = open(filename, 'wb')
        pickle.dump(self.logger, fileobject)
        fileobject.close()

    def get(self, filename):
        fileobject = open(filename, 'rb')
        self.logger = pickle.load(fileobject)
        fileobject.close()
        return self.logger