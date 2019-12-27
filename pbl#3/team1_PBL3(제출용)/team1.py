import os
import struct
import random
cupy_import = 0
if cupy_import:
    import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from skimage.feature import hog
from scipy.ndimage import interpolation

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

## deskew
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

def deskewAll(X):
    currents = []
    for i in range(len(X)):
        currents.append(deskew(X[i].reshape(28,28)).flatten())
    return np.array(currents)

## hog
def calc_hog_features(X, image_shape=(28, 28), pixels_per_cell=(8, 8)):
    fd_list = []
    for row in X:
        img = row.reshape(image_shape)
        fd = hog(img, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1))
        fd_list.append(fd)
    
    return np.array(fd_list)

## convolution
kenel = [
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
]

def convolution(X,kenel):
    con_X =np.zeros((X.shape[0],26*26))
    for y in range(1,26):
        for x in range(1,26):
            temp=X[y-1:y+2,x-1:x+2]
            mul_=np.dot(temp,kenel)
            sum_=np.sum(mul_)
            con_X[y-1][x-1]=sum_
    return con_X

## classifer
class myClassifier(object):    
    """
    ovr
    """
    def __init__(self, C=1000, eta=0.01, batch_size=60, epochs=200, epsilon=1e-8, 
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
        
    def fit(self, X, y, params=None, w=0, b=0, testscore = None, eval_score=None):
        self.class_num=len(np.unique(y))
        #self.class_num=len(cp.unique(y))
        
        if params is None:
            print('fit params=None')
            self.params = {
                'w': np.random.randn(X.shape[1], self.class_num), #(10, 784) 정규분포난수
                #'w': cp.random.randn(X.shape[1], self.class_num),
                'b': np.random.randn(1, self.class_num),
                #'b': cp.random.randn(1, self.class_num),
                'w_': np.random.randn(X.shape[1], self.class_num),
                #'w_': cp.random.randn(X.shape[1], self.class_num),
                'b_': np.random.randn(1, self.class_num),
                #'b_': cp.random.randn(1, self.class_num),
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
            batch_count = int(X.shape[0] / self.batch_size)
            for t in range(int(batch_count)):
                batch_X, batch_y, bs=self.batching(s_data, encoded_y, t)
                batch_X = np.reshape(batch_X, (bs, X.shape[1]))
                #batch_X = cp.reshape(batch_X, (bs, X.shape[1]))
                batch_y = np.reshape(batch_y, (bs, self.class_num))
                #batch_y = cp.reshape(batch_y, (bs, self.class_num))
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
            print("epochs: ", Xi)
            print("prev_score: ", prev_score)
            print("pres_score: ", pres_score,"\n")
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
        #temp = cp.dot(X, w1) + b1
        
        pred = np.argmax(temp, axis=1)
        #pred = cp.argmax(temp, axis=1)
        sco = np.mean(pred == y)
        #sco = cp.mean(pred == y)
        return sco
    
    def update_w_b(self, batch_X, batch_y, z, bs, cnt):
        n = np.shape(batch_X)[1]  # num of features
        #n = cp.shape(batch_X)[1] 
        delta_w = np.zeros(self.params['w'].shape)
        #delta_w = cp.zeros(self.params['w'].shape)
        delta_b = np.zeros(self.params['b'].shape)
        #delta_b = cp.zeros(self.params['b'].shape)
        z = np.reshape(z, (bs, self.class_num))
        #z = cp.reshape(z, (bs, self.class_num))
        temp = 1 - np.multiply(batch_y, z)
        #temp = 1 - cp.multiply(batch_y, z)
        temp[temp <= 0] = 0
        temp[temp > 0] = 1
        y_temp = np.multiply(batch_y, temp.reshape(bs, self.class_num))
        #y_temp = cp.multiply(batch_y, temp.reshape(bs, self.class_num))
        delta_w = -(1 / bs) * np.matmul(batch_X.T, y_temp) + (1 / self.C) * self.params['w']
        #delta_w = -(1 / bs) * cp.matmul(batch_X.T, y_temp) + (1 / self.C) * self.params['w']
        delta_b = -(1 / bs) * np.sum(y_temp, axis=0)
        #delta_b = -(1 / bs) * cp.sum(y_temp, axis=0)
        self.params['w'] = self.params['w'] - (self.eta / (1 + self.epsilon * cnt)) * delta_w
        self.params['b'] = self.params['b'] - (self.eta / (1 + self.epsilon * cnt)) * delta_b
        
        return self.params
    
    def hinge_loss(self, y, z):
        loss = 1 - np.multiply(y, z)
        #loss = 1 - cp.multiply(y, z)
        loss[loss < 0] = 0
        loss = np.mean(loss)
        #loss = cp.mean(loss)
        return loss
    
    def net_input(self, X):  # net_input() = forward_prop(), generate z
        z = np.matmul(X, self.params['w']) + self.params['b']
        #z = cp.matmul(X, self.params['w']) + self.params['b']
        return z

    def encoding(self, y):
        y_array=np.array(y).reshape(-1,1)

        enc=OneHotEncoder('auto')
        enc.fit(y_array)
        y_enc=enc.transform(y_array).toarray()
        
        encoded_y=-1*np.ones((np.shape(y)[0],10))
        
        y_enc=y_enc * 2 + encoded_y
        
        """
        encoded_y=-1*np.ones((np.shape(y)[0],self.class_num))
        #encoded_y=-1*cp.ones((cp.shape(y)[0],self.class_num))
        for i in range(np.shape(y)[0]):
            encoded_y[i,y[i]] = 1
        """
        return y_enc
                
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
        #m = cp.shape(X)[0]
        class_score = self.net_input(X)  # return z
        pred = np.argmax(class_score, axis=1)
        #pred = cp.argmax(class_score, axis=1)

        return pred
    
    def score(self, X, y):
        pred = self.predict(X)
        score = np.mean(pred == y)
        #score = cp.mean(pred == y)
        
        return score
    
    def get_params(self, deep=True):
        return {'C':self.C, 'batch_size':self.batch_size, 'epochs':self.epochs,
               'eta': self.eta, 'w':self.params['w_'], 'b':self.params['b_']}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def test(self, X, w, b):

        if cupy_import : 
            X = cp.array(X)
            z = cp.matmul(X, cp.array(w)) + cp.array(b)
            p = cp.argmax(z, axis=1)
        else :
            z = np.matmul(X, np.array(w)) + np.array(b)
            p = np.argmax(z, axis=1)
            
        return p

def main(training_image, training_label, test_image):
    from time import time
    ## loading mnist dataset ##
    start = time()
    raw_train = read_idx(training_image)
    X_train = np.reshape(raw_train, (80000, 28*28))
    y_train = read_idx(training_label)

    raw_test = read_idx(test_image)
    X_test = np.reshape(raw_test, (60000, 28*28))


    print('train_sample_number:\t:%d, column_number:%d' %(X_train.shape[0], X_train.shape[1]))
    print('testall_sample_number\t:%d, column_number:%d' %(X_test.shape))

    ## Standardzation ##
    X_train = X_train / 255
    X_test = X_test / 255

    ## Deskew ##
    X_train_deskewed = deskewAll(X_train)
    # X_test_deskewed = deskewAll(X_test)
    X_test_deskewed = deskewAll(X_test)

    ## Hog ##
    hoged_X_train = calc_hog_features(X_train_deskewed, pixels_per_cell=(8, 8))
    hoged_X_test = calc_hog_features(X_test_deskewed, pixels_per_cell=(8, 8))
    hoged_X_fea = hoged_X_test.shape[1]

    ## Convolution ##
    con_X_train=convolution(hoged_X_train,kenel)
    con_X_test=convolution(hoged_X_test,kenel)
    con_X_fea=con_X_test.shape[1]

    ## Preprocesing ##
    X_train=np.hstack((X_train_deskewed, con_X_train, hoged_X_train))
    X_test=np.hstack((X_test_deskewed, con_X_test, hoged_X_test))

    ## SVM model ##
    mysvm = myClassifier(C=1000, batch_size=60, epochs=200, eta= 0.01).fit(X_train, y_train)

    w = mysvm.get_params()['w']
    b = mysvm.get_params()['b']

    pred = mysvm.test(X_test, w, b)
    end = time()
    print('elasped time : % s' %(end - start))
    file=open('./prediction.txt','w')
    for i in range(len(pred)):
        file.write('%s\n' %pred[i])
    file.close()

"""
    Execution code:
    python team1.py newtrain-images-idx3-ubyte newtrain-labels-idx1-ubyte testall-patterns-idx3-ubyte
"""
if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
