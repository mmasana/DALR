import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_utils


# Vanilla training function
def train_net(sess, model, x, y_, data, num_iter=100, disp_freq=100, batch_size=100, lr=0.01):

    model.restore(sess)
    model.set_loss(sess, lr)
    
    # initialize test accuracy for train and validation
    test_accs = []
    for imgset in range(2):
        test_accs.append(np.zeros(num_iter/disp_freq))
    
    # train
    for iter in range(num_iter):
        # randomly sample a batch of images, and corresponding labels
        rnd_im = np.random.randint(data.train.images.shape[0], size=batch_size)
        batch_x = data.train.images[rnd_im,:]
        batch_y = data.train.labels[rnd_im]
        batch_y = np.eye(10)[batch_y]
        # train batch
        model.train_step.run(feed_dict={x: batch_x, y_: batch_y})
        # plotting
        if iter % disp_freq == 0:
            plots = []
            colors = ['r', 'b']
            # train plot
            feed_dict={x: data.train.images, y_: np.eye(10)[data.train.labels]}
            test_accs[0][iter/disp_freq] = model.accuracy.eval(feed_dict=feed_dict)            
            plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[0][:iter/disp_freq+1], 'r', label="train " + " (%1.2f)" % test_accs[0][iter/disp_freq])
            plots.append(plot_h)
            # validation plot
            feed_dict={x: data.validation.images, y_: np.eye(10)[data.validation.labels]}
            test_accs[1][iter/disp_freq] = model.accuracy.eval(feed_dict=feed_dict)            
            plot_h, = plt.plot(range(1,iter+2,disp_freq), test_accs[1][:iter/disp_freq+1], 'b', label="valid " + " (%1.2f)" % test_accs[1][iter/disp_freq])
            plots.append(plot_h)
            # merge both
            plot_utils.plot_test_acc(plots)
            plt.title("training")
            plt.gcf().set_size_inches(5, 3.5)


# Inference function
def test_net(sess, model, x, y_, data):
    model.restore(sess)
    acc_test = model.accuracy.eval({x: data.test.images, y_: np.eye(10)[data.test.labels]})
    print("Accuracy on test data is {}%".format(acc_test*100, decimals=3))
