from sklearn.metrics import make_scorer, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import struct
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import random
import sys

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

class myClassifier(object):
    """
    ovr
    """
    def __init__(self, C=1000, eta=0.1, batch_size=20, epochs=100, epsilon=1e-8, 
                 shuffle=True, params=None, w=0, b=0):
        self.C = C
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.epsilon = epsilon
        self.class_num = 0
        self.shuffle = shuffle
        self.update_count = 0
        self.w = 0
        self.b = 0
#         self.params['aver_w'] = w
#         self.params['aver_b'] = b
        
    def fit(self, X, y, params=None, w=0, b=0, testscore = None, eval_score=None):
        # X_num = m, X_fea = n
        # m = np.shape(X)[0], n = np.shape(X)[1]
        
        X_num, X_fea = np.shape(X)
        #X_num=60000 X_fea=28*28
        self.class_num=len(np.unique(y))
        #class_num=10
        
        if params is None:
            print('fit params=None')
            self.params = {
                'w': np.random.randn(X_fea, self.class_num), #(10, 784) 정규분포난수
                'b': np.random.randn(1, self.class_num),
                'w_': np.random.randn(X_fea, self.class_num),
                'b_': np.random.randn(1, self.class_num),
                'tmpw': 0,
                'tmpb': 0
            }
        cnt=1
        if eval_score is None:
            self.score_val = 0
                
        for Xi in range(self.epochs):
            s_data, s_labels = self.shuffling(X, y)
            encoded_y=self.encoding(s_labels)
            avg_loss = 0
            batch_count = int(X_num / self.batch_size)
            for t in range(int(batch_count)):
#                self.params['tmpw'] = temp_w, self.params['tmpb'] = temp_b
                batch_X, batch_y, bs=self.batching(s_data, encoded_y, t)
                batch_X = np.reshape(batch_X, (bs, X_fea))
                batch_y = np.reshape(batch_y, (bs, self.class_num))
                z = self.net_input(batch_X)
                loss = self.hinge_loss(batch_y, z)
                self.update_w_b(batch_X, batch_y, z, bs, cnt)
                cnt+=1
                avg_loss += loss
                self.update_count += 1

            self.params['tmpw'] = (cnt * (cnt/(cnt+1)) * 
                                   self.params['w_'] + (1/(cnt+1))*self.params['w'])
            self.params['tmpb'] = (cnt * (cnt/(cnt+1)) * 
                                   self.params['b_'] + (1/(cnt+1))*self.params['b'])
            prev_score = self.score_val
            pres_score = self.score(X, y)
            if Xi % 10 == 0:
                print("epochs: ", Xi)
                print("prev_score: ", prev_score)
                print("pres_score: ", pres_score)
                print()
            if prev_score < pres_score:
                self.score_val = pres_score
            if self.det_weight(X, y, 1) < self.det_weight(X, y): # temp_w, temp_b
                self.params['w_'] = self.params['tmpw']
                self.params['b_'] = self.params['tmpb']
            avg_loss /= batch_count
        return self
    
    def det_weight(self, X, y, aver=0):
        if aver:
            w1 = self.params['w_']
            b1 = self.params['b_']
        else:
            w1 = self.params['tmpw']
            b1 = self.params['tmpb']
        temp = np.dot(X, w1) + b1
#         temp = temp.T
        pred = np.argmax(temp, axis=1)
        sco = np.mean(pred == y)
        return sco
    
    def update_w_b(self, batch_X, batch_y, z, bs, cnt):
        n = np.shape(batch_X)[1]  # num of features
        delta_w = np.zeros(self.params['w'].shape)
        delta_b = np.zeros(self.params['b'].shape)
        z = np.reshape(z, (bs, self.class_num))
        temp = 1 - np.multiply(batch_y, z)
        temp[temp <= 0] = 0
        temp[temp > 0] = 1
        y_temp = np.multiply(batch_y, temp.reshape(bs, self.class_num))
        delta_w = -(1 / bs) * np.matmul(batch_X.T, y_temp) + (1 / self.C) * self.params['w']
        delta_b = -(1 / bs) * np.sum(y_temp, axis=0)
        self.params['w'] = self.params['w'] - (self.eta / (1 + self.epsilon * cnt)) * delta_w
        self.params['b'] = self.params['b'] - (self.eta / (1 + self.epsilon * cnt)) * delta_b
        
        return self.params
    
    def hinge_loss(self, y, z):
        loss = 1 - np.multiply(y, z)
        loss[loss < 0] = 0
        loss = np.mean(loss)
        return loss
    
    def net_input(self, X):  # net_input() = forward_prop(), generate z
        z = np.matmul(X, self.params['w']) + self.params['b']
        return z

    def encoding(self, y):
        encoded_y=-1*np.ones((np.shape(y)[0],self.class_num))
        for i in range(np.shape(y)[0]):
            encoded_y[i,y[i]] = 1
        return encoded_y

    def shuffling(self, X, y):
        temp_s=list(zip(X,y))
        random.shuffle(temp_s)
        X,y=zip(*temp_s)
        return X,y

    def batching(self, X, y, t):                         
        batch_X = X[t * self.batch_size : min(len(X), (t+1) * self.batch_size)]
        batch_y = y[t * self.batch_size : min(len(X), (t+1) * self.batch_size)]
        last_size = min(len(X), (t+1) * self.batch_size) - t * self.batch_size
        
        return batch_X, batch_y,last_size
    
    def predict(self, X):
        m = np.shape(X)[0]
        class_score = self.net_input(X)  # return z
        pred = np.argmax(class_score, axis=1)

        return pred
    
    def score(self, X, y):
        pred = self.predict(X)
        score = np.mean(pred == y)
        
        return score
    
    def get_params(self, deep=True):
        return {'C':self.C, 'batch_size':self.batch_size, 'epochs':self.epochs,
               'eta': self.eta, 'w':self.params['w'], 'b':self.params['b']}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def test(self, X, w, b):
        print("============== TESTING =================")
        z = np.dot(X, np.array(w)) + np.array(b)
        p = np.argmax(z, axis=1)
        return p

def main(training_image, training_label, test_image):
    ## loading mnist dataset ##
    raw_train = read_idx(training_image)
    train_data = np.reshape(raw_train, (80000, 28*28))
    train_label = read_idx(training_label)

    raw_test = read_idx(test_image)
    test_data = np.reshape(raw_test, (60000, 28*28))
    ## test_label = read_idx("./data/test-labels-idx1-ubyte")

    ## Standardzation ##
    # X_train_std = StandardScaler().fit_transform(train_data)
    # X_test_std = StandardScaler().fit_transform(test_data)
    X_train_std = train_data / 255
    X_test_std = test_data / 255

    ## SVM model ##
    mysvm = myClassifier(C=1000, batch_size=20, epochs=200, eta= 0.01).fit(X_train_std, train_label)

    w = mysvm.get_params()['w']
    b = mysvm.get_params()['b']

    pred = mysvm.test(X_test_std, w, b)

    file=open('./prediction.txt','w')
    for i in range(len(pred)):
        file.write('%s\n' %pred[i])
    file.close()

""" input
    argv : PATH
    eg) python team1.py ./data/newtrain-images-idx3-ubyte ./data/newtrain-labels-idx1-ubyte ./data/mnist_new_testall-patterns-idx3-ubyte
"""
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
