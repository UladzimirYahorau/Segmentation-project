import tensorflow as tf
import numpy as np
import time
import os
from brain_data import *

# __file__ might not be defined when you run the following line in an interactive shell.
dir = "/home/uladzimir/Segmentation project"

# Let's load a previously saved metagraph in the default graph.
# This function returns a Saver
saver = tf.train.import_meta_graph( dir + "/second_phase.chkp.meta")

# We can now access the default graph where all the data has been loaded.
graph = tf.get_default_graph()
#[print(n.name) for n in graph.as_graph_def().node if "accuracy" in n.name ]

# Finally we can retrieve tensors, operations, collections etc.
dice_valid = graph.get_tensor_by_name('dice_valid:0')
cost_valid = graph.get_tensor_by_name('cost_valid:0')
accuracy_valid = graph.get_tensor_by_name('accuracy_valid:0')
input = graph.get_tensor_by_name('input:0')
y_valid = graph.get_tensor_by_name('y_valid:0')
output = graph.get_tensor_by_name('output:0')


data = load_data()
labels = load_labels()

folder = os.path.dirname(os.path.realpath(__file__))

with tf.Session() as sess:
    saver.restore(sess, dir + '/second_phase.chkp')
    
    with open(folder + '/valid_history.txt', 'a') as f:
        f.write("Stats on brains in a valid set FROM model_testing.py!!!\n")
        f.write('brain    dice        cost      accuracy      time\n')
    
    for index in range(1,28):
        valid_data, valid_labels = get_validation_samples(data, labels, range(index,index+1), 33)
        valid_data = np.transpose(valid_data, (0, 2, 3, 1))
    
        start_time = time.time()
        pre_brain_mask, dice_epoch, cost_epoch, accuracy_epoch = sess.run([output, dice_valid, cost_valid, accuracy_valid], feed_dict = {input: valid_data, y_valid: valid_labels})
        
        mask = brain_mask(pre_brain_mask)
        save_segmentation_to_file(mask, index+1)
        
        with open(folder + '/valid_history.txt', 'a') as f:
            f.write("{:10}".format(index)+'  ')
            f.write("{:10.2f}".format(dice_epoch) + '  ')
            f.write("{:10.1f}".format(cost_epoch) + '  ')
            f.write("{:10.2f}".format(accuracy_epoch) +'   ' ) 
            f.write("{:10.2f}".format(time.time() - start_time) + '\n')