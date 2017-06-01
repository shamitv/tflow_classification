import numpy as np
from numpy import genfromtxt
from operator import neg
from pathlib import Path
import tensorflow as tf
import pickle
import time
import tflearn
import codecs

def load_file(path, sentence_path, output_value):
    data={};
    print("Reading CSV : " + path);
    inputs = genfromtxt(path, delimiter=' ');
    num_rows = inputs.shape[0];
    outputs = np.full(num_rows, output_value);
    file = codecs.open(sentence_path, 'r',encoding='utf-8');
    lines=file.readlines();
    data['inputs']=inputs;
    data['outputs'] = outputs;
    data['sentences'] = np.array(lines, dtype=object);
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
    s=data['sentences'];
    num_examples=x.shape[0];
    num_training_post_split=int((training_percent/100)*num_examples)
    indices = np.random.permutation(num_examples)
    training_idx, test_idx = indices[:num_training_post_split], indices[num_training_post_split:]
    training_inputs, test_inputs = x[training_idx, :], x[test_idx, :]
    training_outputs, test_outputs = y[training_idx, :], y[test_idx, :]
    training_sentences, test_sentences= s[training_idx, :], s[test_idx, :]
    data['inputs']=training_inputs;
    data['outputs']=training_outputs;
    data['sentences'] = training_sentences;
    data['test_inputs']=test_inputs;
    data['test_outputs']=test_outputs;
    data['test_sentences'] = test_sentences;
    return data;

def load_data_from_csv():
    data = {}
    negative_file = "K:/nlp/sentiment/data/yelp/split/negative_vectors.csv";
    positive_file = "K:/nlp/sentiment/data/yelp/split/positive_vectors.csv";
    neutral_file = "K:/nlp/sentiment/data/yelp/split/neutral_vectors.csv";
    negative_sentence_file = "K:/nlp/sentiment/data/yelp/split/negative.txt";
    positive_sentence_file = "K:/nlp/sentiment/data/yelp/split/positive.txt";
    neutral_sentence_file = "K:/nlp/sentiment/data/yelp/split/neutral.txt";
    negative_data = load_file(negative_file,negative_sentence_file,0);
    positive_data = load_file(positive_file,positive_sentence_file ,2);
    neutral_data = load_file(neutral_file,neutral_sentence_file,1);
    data['inputs'] = np.append(negative_data['inputs'],positive_data['inputs'],axis=0);
    #data['inputs'] = np.append(data['inputs'],neutral_data['inputs'],axis=0);
    data['sentences'] = np.append(negative_data['sentences'],positive_data['sentences'],axis=0);
    # data['sentences'] = np.append(data['sentences'],neutral_data['sentences'],axis=0);
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


def build_network (num_input_attributes,num_classes):
    num_cols=num_input_attributes;
    # Building deep neural network
    input_layer = tflearn.input_data(shape=[None, num_cols]);
    dense1 = tflearn.fully_connected(input_layer, num_cols, activation='tanh',regularizer='L2', weight_decay=0.001);
    dropout1 = tflearn.dropout(dense1, 0.8);
    dense2 = tflearn.fully_connected(dropout1, num_cols, activation='tanh',regularizer='L2', weight_decay=0.001);
    dropout2 = tflearn.dropout(dense2, 0.8);
    softmax = tflearn.fully_connected(dropout2, n_classes, activation='softmax');
    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000);
    top_k = tflearn.metrics.Top_k(3);
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,loss='categorical_crossentropy');
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    return model;

tagged_data = load_data();

num_rows = tagged_data['inputs'].shape[0];
num_cols=tagged_data['inputs'].shape[1];
print("Training examples= ",num_rows)
print("Training attributes= ",num_cols)
n_classes = 3 #
model=build_network(num_cols,n_classes);
model.fit( tagged_data['inputs'],  tagged_data['outputs'], n_epoch=5, validation_set=(tagged_data['test_inputs'], tagged_data['test_outputs']),
show_metric=True, run_id="dense_model")
