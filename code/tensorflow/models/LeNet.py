import random
import numpy as np
import tensorflow as tf
from copy import deepcopy
from IPython import display
from tensorflow.contrib.layers import flatten


class LeNet:
    def __init__(self, x, y_):
        
        # LeNet
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        self.x = x

        # Hyperparameters
        mu = 0
        sigma = 0.1
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        self.conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
        self.conv1_b = tf.Variable(tf.zeros(6))
        self.conv1 = tf.nn.conv2d(self.x,self.conv1_w, strides = [1,1,1,1], padding = 'VALID') + self.conv1_b
        # Activation.
        self.conv1 = tf.nn.relu(self.conv1)
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        self.pool_1 = tf.nn.max_pool(self.conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        # Layer 2: Convolutional. Output = 10x10x16.
        self.conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
        self.conv2_b = tf.Variable(tf.zeros(16))
        self.conv2 = tf.nn.conv2d(self.pool_1, self.conv2_w, strides = [1,1,1,1], padding = 'VALID') + self.conv2_b
        # Activation.
        self.conv2 = tf.nn.relu(self.conv2)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        self.pool_2 = tf.nn.max_pool(self.conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        # Flatten. Input = 5x5x16. Output = 400.
        self.fla = flatten(self.pool_2)
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        self.fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
        self.fc1_b = tf.Variable(tf.zeros(120))
        self.fc1 = tf.matmul(self.fla,self.fc1_w) + self.fc1_b
        # Activation.
        self.fc1 = tf.nn.relu(self.fc1)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
        self.fc2_b = tf.Variable(tf.zeros(84))
        self.fc2 = tf.matmul(self.fc1,self.fc2_w) + self.fc2_b
        # Activation.
        self.fc2 = tf.nn.relu(self.fc2)
        # Layer 5: Fully Connected. Input = 84. Output = number of features.
        self.fc3_w = tf.Variable(tf.truncated_normal(shape = (84,out_dim), mean = mu , stddev = sigma))
        self.fc3_b = tf.Variable(tf.zeros(out_dim))
        self.y = tf.matmul(self.fc2, self.fc3_w) + self.fc3_b

        # lists
        self.var_list = [self.conv1_w, self.conv1_b, self.conv2_w, self.conv2_b,
                         self.fc1_w,   self.fc1_b,   self.fc2_w,   self.fc2_b,
                         self.fc3_w,   self.fc3_b]
        self.hidden_list = [self.conv1, self.conv2, self.fc1, self.fc2, self.y]
        self.input_list = [self.x, self.conv1, self.fla, self.fc1, self.fc2]
        
        # vanilla single-task loss
        one_hot_targets = y_
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_targets , logits=self.y))

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def compressed_model_FC(self,x,y_):
        
        self.x = x
        in_dim = int(x.get_shape()[1])
        out_dim = int(y_.get_shape()[1])
        
        # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        self.conv1_w = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[0])))
        self.conv1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[1])))
        self.conv1 = tf.nn.conv2d(self.x,self.conv1_w, strides = [1,1,1,1], padding = 'VALID') + self.conv1_b
        # Activation.
        self.conv1 = tf.nn.relu(self.conv1)
        # Pooling. Input = 28x28x6. Output = 14x14x6.
        self.pool_1 = tf.nn.max_pool(self.conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        # Layer 2: Convolutional. Output = 10x10x16.
        self.conv2_w = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[1])))
        self.conv2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[3])))
        self.conv2 = tf.nn.conv2d(self.pool_1, self.conv2_w, strides = [1,1,1,1], padding = 'VALID') + self.conv2_b
        # Activation.
        self.conv2 = tf.nn.relu(self.conv2)
        # Pooling. Input = 10x10x16. Output = 5x5x16.
        self.pool_2 = tf.nn.max_pool(self.conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
        # Flatten. Input = 5x5x16. Output = 400.
        self.fla = flatten(self.pool_2)
        # Layer 3: Fully Connected. Input = 400. Output = 120.
        self.fc1_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[2])))
        self.fc1_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[3])))
        self.fc1_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[5])))
        self.fc1 = tf.matmul(tf.matmul(self.fla,self.fc1_w1),self.fc1_w2) + self.fc1_b
        # Activation.
        self.fc1 = tf.nn.relu(self.fc1)
        # Layer 4: Fully Connected. Input = 120. Output = 84.
        self.fc2_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[4])))
        self.fc2_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[5])))
        self.fc2_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[7])))
        self.fc2 = tf.matmul(tf.matmul(self.fc1,self.fc2_w1),self.fc2_w2) + self.fc2_b
        # Activation.
        self.fc2 = tf.nn.relu(self.fc2)
        # Layer 5: Fully Connected. Input = 84. Output = number of features.
        self.fc3_w1 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[6])))
        self.fc3_w2 = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_svd[7])))
        self.fc3_b = tf.Variable(tf.convert_to_tensor(np.float32(self.weights_w[9])))
        self.y = tf.matmul(tf.matmul(self.fc2,self.fc3_w1),self.fc3_w2) + self.fc3_b

        # lists
        self.var_list = [self.conv1_w, self.conv1_b, 
                         self.conv2_w, self.conv2_b, 
                         self.fc1_w1, self.fc1_w2, self.fc1_b, 
                         self.fc2_w1, self.fc2_w2, self.fc2_b, 
                         self.fc3_w1, self.fc3_w2, self.fc3_b]
        self.hidden_list = [self.conv1, self.conv2, self.fc1, self.fc2, self.y]
        self.input_list = [self.x, self.conv1, self.fla, self.fc1, self.fc2]
        
        # loss
        one_hot_targets = y_
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_targets , logits=self.y))

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.global_step = tf.contrib.framework.get_or_create_global_step()
 

    def get_activations(self, imgset, sess, num_samples=200):
        # initialize
        self.acts = []
        for v in range(len(self.input_list)):
            self.acts.append(np.zeros([num_samples] + self.input_list[v].get_shape().as_list()[1:]))

        # random non-repeating selected images
        rnd_indx = random.sample(xrange(0,imgset.shape[0]),num_samples)
        for m in range(num_samples):
            # select random input image
            im_ind = rnd_indx[m]
            # extract activations from samples
            acts = sess.run(self.input_list, feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # store them in the corresponding variable
            for v in range(len(self.input_list)):
                self.acts[v][m] = acts[v]

 
    def count_params(self):
        # counting the total amount of parameters
        cont = []
        for v in range(len(self.var_list)):
            cont.append(int(np.product(self.var_list[v].shape)))
        print("Number of parameters in network is {}.".format(sum(cont)))
        # counting the amount of parameters in CONV
        cont = []
        for v in range(4):
            cont.append(int(np.product(self.var_list[v].shape)))
        print("Number of parameters in CONV is {}.".format(sum(cont)))
        # counting the amount of parameters in FC
        cont = []
        for v in range(4, len(self.var_list)):
            cont.append(int(np.product(self.var_list[v].shape)))
        print("Number of parameters in FC is {}.".format(sum(cont)))


    def star(self):
        # used for saving optimal weights training
        self.star_vars = []
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())


    def restore(self, sess):
        # reassign optimal weights
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))


    def set_loss(self, sess, lr):
        self.train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.cross_entropy)


    def compute_svd(self, sess, keep):
        # compute SVD method
        self.weights_svd = []
        self.weights_w = []
        for v in range(len(self.var_list)):
            self.weights_w.append(sess.run(self.var_list[v]))
        
        for v in range(len(self.hidden_list)):
            # check which type of variable it is -- if 1 then bias
            if len(self.weights_w[v*2].shape)==2:
                # it's a fully-connected layer - apply SVD method
                U, S, V = np.linalg.svd(self.weights_w[v*2], full_matrices=False)
                A = U[:, 0:keep[v]]
                B = np.dot(np.diag(S[0:keep[v]]), V[0:keep[v], :])
                self.weights_svd.append(A)
                self.weights_svd.append(B)
            else:
                if len(self.weights_w[v*2].shape)==4:
                    # it's a convolutional layer - do nothing
                    self.weights_svd.append(self.weights_w[v*2])


    def compute_dalr(self, sess, keep, lam=1000):
        # compute DALR method
        self.weights_svd = []
        self.weights_w = []
        for v in range(len(self.var_list)):
            self.weights_w.append(sess.run(self.var_list[v]))
        
        for v in range(len(self.hidden_list)):
            # check which type of variable it is -- if 1 then bias
            if len(self.weights_w[v*2].shape)==2:
                # it's a fully-connected layer - apply DALR method
                Z = np.dot(self.weights_w[v*2].T, self.acts[v].T)
                U, _, _ = np.linalg.svd(Z, full_matrices=False)
                XXT = np.dot(self.acts[v].T, self.acts[v])
                A = U[:, 0:keep[v]]
                B = np.dot(np.dot(np.dot(U[:, 0:keep[v]].T, self.weights_w[v*2].T), XXT), np.linalg.inv(XXT + lam*np.eye(self.acts[v].shape[1])))
                self.weights_svd.append(B.T)
                self.weights_svd.append(A.T)
            else:
                if len(self.weights_w[v*2].shape)==4:
                    # it's a convolutional layer - do nothing
                    self.weights_svd.append(self.weights_w[v*2])
