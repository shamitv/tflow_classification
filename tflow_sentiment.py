import numpy as np
from numpy import genfromtxt
from operator import neg
from pathlib import Path
import tensorflow as tf
import pickle
import time

def load_file(path, output_value):
    data={};
    print("Reading CSV : " + path);
    inputs = genfromtxt(path, delimiter=' ');
    num_rows = inputs.shape[0];
    outputs = np.full(num_rows, output_value);
    data['inputs']=inputs;
    data['outputs'] = outputs;
    return data;

def transform_to_one_hot(values):
    val_size = values.size;
    val_max = values.max();
    ret=np.zeros((val_size, val_max+1));
    ret[np.arange(val_size),values] = 1
    return ret;

def split_training_test(data,training_percent):
    x=data['inputs'];
    y=data['outputs'];
    num_examples=x.shape[0];
    num_training_post_split=int((training_percent/100)*num_examples)
    indices = np.random.permutation(num_examples)
    training_idx, test_idx = indices[:num_training_post_split], indices[num_training_post_split:]
    training_inputs, test_inputs = x[training_idx, :], x[test_idx, :]
    training_outputs, test_outputs = y[training_idx, :], y[test_idx, :]
    data['inputs']=training_inputs;
    data['outputs']=training_outputs;
    data['test_inputs']=test_inputs;
    data['test_outputs']=test_outputs;
    return data;

def load_data_from_csv():
    data = {}
    negative_file = "K:/nlp/sentiment/data/yelp/split/negative_vectors.csv";
    positive_file = "K:/nlp/sentiment/data/yelp/split/positive_vectors.csv";
    neutral_file = "K:/nlp/sentiment/data/yelp/split/neutral_vectors.csv";
    negative_data = load_file(negative_file,0);
    positive_data = load_file(positive_file,2);
    neutral_data = load_file(neutral_file,1);
    data['inputs'] = np.append(negative_data['inputs'],positive_data['inputs'],axis=0);
    #data['inputs'] = np.append(data['inputs'],neutral_data['inputs'],axis=0);
    data['outputs'] = np.append(negative_data['outputs'], positive_data['outputs']);
    #data['outputs'] = np.append(data['outputs'], neutral_data['outputs']);
    data['outputs'] = transform_to_one_hot(data['outputs']);
    data=split_training_test(data,80.);
    return data;

def load_data():
    pickle_path= "K:/nlp/sentiment/data/yelp/train_test.p";
    data_file = Path(pickle_path)
    if data_file.is_file():
        print("Loading data from Pickle : "+pickle_path);
        data=pickle.load( open( pickle_path, "rb" ) );
    else:
        print("Missing Pickle : " + pickle_path+" loading data from source");
        data=load_data_from_csv();
        pickle.dump(data, open(pickle_path, "wb"));
        print("Saving data to Pickle : " + pickle_path);
    return data;


tagged_data = load_data();

num_rows = tagged_data['inputs'].shape[0];
num_cols=tagged_data['inputs'].shape[1];
print("Training examples= ",num_rows)
print("Training attributes= ",num_cols)


# Learning Parameters
learning_rate = 0.001
training_epochs = 10000
batch_size = 100
display_step = 500


# Network Parameters
n_hidden_1 =  num_cols*3# 1st layer number of features
n_hidden_2 = num_cols*2 # 2nd layer number of features
n_hidden_3 = num_cols # 3nd layer number of features
n_hidden_4 = int(num_cols/2) # 4nd layer number of features
n_input = num_cols #
n_classes = 3 #

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    #out_layer = tf.nn.softmax(out_layer)
    return out_layer;


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
start_time = time.time()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        _, epoch_cost = sess.run([optimizer, cost], feed_dict={x: tagged_data['inputs'],y: tagged_data['outputs']})
        if (epoch == 0 or (epoch+1) % display_step == 0):
            print("Epoch:", '%04d' % (epoch+1), "cost=","{:.25f}".format(epoch_cost))
    print("Optimization Finished! Total iteration = ", epoch+1)
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: tagged_data['test_inputs'],y: tagged_data['test_outputs']}))

elapsed_time = time.time() - start_time
print("Elapsed time (seconds) :: ",elapsed_time)

