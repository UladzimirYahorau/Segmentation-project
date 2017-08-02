import tensorflow as tf
import os 
import numpy as np
import time

from brain_data import *

##########################################################################
####################### Create a 2-paths model ########################### 
##########################################################################


# Load the data
data = load_data()
labels = load_labels()

input = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 4), name = "input")
y = tf.placeholder(dtype = tf.int8, shape = (None, 2), name = "y")
y_valid = tf.placeholder(dtype = tf.int8, shape = (None, None, None, 2), name = "y_valid")
y_valid_flatten = tf.reshape(y_valid, [-1, 2], name = "y_valid_flatten")

weights_local_1 = tf.Variable(tf.random_normal(shape = [7,7,4,64], stddev = 0.1, dtype = tf.float32),
                name = "weights_local_1")
biases_local_1 = tf.Variable(tf.zeros(shape = [64], dtype = tf.float32),
                name = "biases_local_1")

hidden_local_1_conv = tf.nn.conv2d(input, weights_local_1, strides = [1,1,1,1], padding = "VALID", data_format = "NHWC")
hidden_local_1_add_bias = tf.nn.bias_add(hidden_local_1_conv, biases_local_1)
hidden_local_1_relu = tf.nn.relu(hidden_local_1_add_bias)
hidden_local_1_max_pool = tf.nn.max_pool(value = hidden_local_1_relu, ksize = [1,4,4,1], strides = [1,1,1,1], padding = "VALID",
                data_format = "NHWC")
hidden_local_1 = tf.identity(hidden_local_1_max_pool, name = "hidden_local_1")

weights_local_2 = tf.Variable(tf.random_normal(shape = [3,3,64,64], stddev = 0.1, dtype = tf.float32), 
                name = "weights_local_2")
biases_local_2 = tf.Variable(tf.zeros(shape = [64], dtype = tf.float32), name = "biases_local_2")


hidden_local_2_conv = tf.nn.conv2d(input = hidden_local_1, filter = weights_local_2, strides = [1,1,1,1], padding = "VALID",
                data_format = "NHWC", name = "hidden_local_2")
hidden_local_2_add_bias = tf.nn.bias_add(hidden_local_2_conv, biases_local_2)
hidden_local_2_relu = tf.nn.relu(hidden_local_2_add_bias)
hidden_local_2_max_pool = tf.nn.max_pool(value = hidden_local_2_relu, ksize = [1,2,2,1], strides = [1,1,1,1], padding = "VALID",
                data_format = "NHWC")
hidden_local_2 = tf.identity(hidden_local_2_max_pool, name = "hidden_local_2")

weights_global = tf.Variable(tf.random_normal(shape = [13,13,4,160], stddev = 0.1, dtype = tf.float32),
                name = "weights_global")
biases_global = tf.Variable(tf.zeros(shape = [160], dtype = tf.float32),
                name = "biases_global")

hidden_global_conv = tf.nn.conv2d(input = input, filter = weights_global, strides = [1,1,1,1], padding = "VALID",
                data_format = "NHWC")
hidden_global_add_bias = tf.nn.bias_add(hidden_global_conv, biases_global)
hidden_global_relu = tf.nn.relu(hidden_global_add_bias)
hidden_global = tf.identity(hidden_global_relu, name = "hidden_global") 

concat = tf.concat([hidden_local_2, hidden_global], axis = 3, name = "concat")

weights_output = tf.Variable(tf.random_normal(shape = [21,21,224,2], stddev = 0.1, dtype = tf.float32))
biases_output = tf.Variable(tf.zeros(shape = [2], dtype = tf.float32))

output_conv = tf.nn.conv2d(input = concat, filter = weights_output, strides = [1,1,1,1], padding = "VALID",
                    data_format = "NHWC")
output = tf.nn.bias_add(output_conv, biases_output, name = "output")
# output_probs = tf.nn.softmax(output, dim = -1, name = "output_probs")

logits = tf.reshape(output, [-1, 2], name = "logits")

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y),
        name = "cost") 

cost_valid = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_valid_flatten),
        name = "cost_valid")

optimizer_1 = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost, name = 'optimizer_1')
optimizer_2 = tf.train.AdamOptimizer().minimize(cost, name = 'optimizer_2', var_list = [weights_output, biases_output])

correct_predictions_1 = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1), name = "correct_predictions_1")
accuracy = tf.reduce_mean(tf.cast(correct_predictions_1, dtype = tf.float32), name = "accuracy")

correct = tf.argmax(y, 1)
predictions = tf.argmax(logits, 1)
precision = tf.divide(tf.reduce_sum(tf.multiply(correct, predictions)), tf.reduce_sum(predictions))
recall = tf.divide(tf.reduce_sum(tf.multiply(correct, predictions)), tf.reduce_sum(correct))
dice = tf.divide(tf.cast(2, tf.float64), tf.add(tf.divide(1, precision), tf.divide(1, recall)), name = "dice")



correct_predictions_valid = tf.equal(tf.argmax(logits, 1), tf.argmax(y_valid_flatten, 1), name = "correct_predictions_valid")
accuracy_valid = tf.reduce_mean(tf.cast(correct_predictions_valid, dtype = tf.float32), name = "accuracy_valid")

correct_valid = tf.argmax(y_valid_flatten, 1)
predictions_valid = tf.argmax(logits, 1)
precision_valid = tf.divide(tf.reduce_sum(tf.multiply(correct_valid, predictions_valid)), tf.reduce_sum(predictions_valid))
recall_valid = tf.divide(tf.reduce_sum(tf.multiply(correct_valid, predictions_valid)), tf.reduce_sum(correct_valid))
dice_valid = tf.divide(tf.cast(2, tf.float64), tf.add(tf.divide(1, precision_valid), tf.divide(1, recall_valid)), name = "dice_valid")


EPOCHS = 100

folder = os.path.dirname(os.path.realpath(__file__))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
   
    with open(folder + '/valid_history.txt', 'a') as f:
        f.write('First phase of training \n')
        f.write('epoch    dice        cost      accuracy     time\n')
    
    valid_data, valid_labels = get_uniform_samples(data, labels, range(21,28), 1000, 33)
    valid_data = np.transpose(valid_data, (0, 2, 3, 1))
    min_cost = 100000
   
    start_time = time.time()
####  for epoch in range(EPOCHS):     for now!!!!
    for epoch in range(EPOCHS):
        data_batch, label_batch = get_uniform_samples(data, labels, range(21), 1000, 33)
        data_batch = np.transpose(data_batch, (0, 2, 3, 1))
        sess.run(optimizer_1, feed_dict = {input: data_batch, y: label_batch})
        dice_epoch, cost_epoch, accuracy_epoch = sess.run([dice, cost, accuracy], feed_dict = {input: valid_data, y: valid_labels})
        if cost_epoch < min_cost:
            min_cost = cost_epoch
            saver.save(sess, folder + '/first_phase.chkp')
            
        with open(folder + '/valid_history.txt', 'a') as f:
            f.write(str(epoch+1) + '  ')
            f.write("{:10.2f}".format(dice_epoch)+'  ')
            f.write("{:10.1f}".format(cost_epoch) + '  ')
            f.write("{:10.2f}".format(accuracy_epoch) + '   ' )
            f.write("{:10.1f}".format(time.time()-start_time) + '\n')
        
            
    
    
    with open(folder + '/valid_history.txt', 'a') as f:
        f.write('Second phase of training \n')
        f.write('epoch    dice        cost      accuracy\n')
    
    valid_data, valid_labels = get_samples(data, labels, range(21,28), 1000, 33)
    valid_data = np.transpose(valid_data, (0, 2, 3, 1))
    min_cost = 100000
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        data_batch, label_batch = get_samples(data, labels, range(21), 1000, 33)
        data_batch = np.transpose(data_batch, (0, 2, 3, 1))
        sess.run(optimizer_2, feed_dict = {input: data_batch, y: label_batch})
        print("hi!")
#        dice_epoch, cost_epoch, accuracy_epoch = sess.run([dice_valid, cost_valid, accuracy_valid], feed_dict = {input: #valid_data, y_valid: valid_labels})
        dice_epoch, cost_epoch, accuracy_epoch = sess.run([dice, cost, accuracy], feed_dict = {input: valid_data, y: valid_labels})
        
        if cost_epoch < min_cost:
            min_cost = cost_epoch
            saver.save(sess, folder + '/second_phase.chkp')
        
        with open(folder + '/valid_history.txt', 'a') as f:
            f.write(str(epoch+1) + '  ')
            f.write("{:10.2f}".format(dice_epoch)+'  ')
            f.write("{:10.1f}".format(cost_epoch) + '  ')
            f.write("{:10.2f}".format(accuracy_epoch) +'  ' ) 
            f.write("{:10.2f}".format(time.time()-start_time) + '\n')
            
    
        
        
        
        
        
        
        
        
        
        
        
        
    
    
       
   

    
    