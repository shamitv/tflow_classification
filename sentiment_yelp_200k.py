import numpy as np
from numpy import genfromtxt
from tensorflow.python.saved_model import builder as saved_model_builder
from pathlib import Path
import tensorflow as tf
import pickle
import tflearn
import codecs
import os
import time

modelPath="G:/work/nlp/datasets/yelp/yelp_dataset_challenge_round9/tflearn_model"
basePath="G:/work/nlp/datasets/yelp/yelp_dataset_challenge_round9/split/";

def load_file(path, sentence_path, output_value):
    data={};
    print("Reading CSV : " + path);
    inputs = genfromtxt(path, delimiter=' ');
    num_rows = inputs.shape[0];
    outputs = np.full(num_rows, output_value);
    print("Reading Text : " + sentence_path);
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

def shuffle_data(data):
    x=data['inputs'];
    y=data['outputs'];
    s=data['sentences'];
    num_examples=x.shape[0];
    indices = np.random.permutation(num_examples)
    training_inputs = x[indices, :]
    training_outputs = y[indices, :]
    training_sentences = s[indices]

    data['inputs']=training_inputs;
    data['outputs']=training_outputs;
    data['sentences'] = training_sentences;


    x=data['test_inputs'];
    y=data['test_outputs'];
    s=data['test_sentences'];
    num_examples = x.shape[0];
    indices = np.random.permutation(num_examples)
    test_inputs = x[indices, :]
    test_outputs = y[indices, :]
    test_sentences = s[indices]

    data['test_inputs']=test_inputs;
    data['test_outputs']=test_outputs;
    data['test_sentences'] = test_sentences;
    return data;

def load_data_from_csv():
    data = {}



    negative_file = basePath + "1_star_training.csv";
    positive_file = basePath + "5_star_training.csv";
    negative_sentence_file = basePath + "1_star_training.txt";
    positive_sentence_file = basePath + "5_star_training.txt";
    negative_data = load_file(negative_file,negative_sentence_file,0);
    positive_data = load_file(positive_file,positive_sentence_file ,2);

    test_negative_file = basePath + "1_star_test.csv";
    test_positive_file = basePath + "5_star_test.csv";
    test_negative_sentence_file = basePath + "1_star_test.txt";
    test_positive_sentence_file = basePath + "5_star_test.txt";
    test_negative_data = load_file(test_negative_file,test_negative_sentence_file,0);
    test_positive_data = load_file(test_positive_file,test_positive_sentence_file ,2);


    data['inputs'] = np.append(negative_data['inputs'],positive_data['inputs'],axis=0);
    data['sentences'] = np.append(negative_data['sentences'],positive_data['sentences'],axis=0);
    data['outputs'] = np.append(negative_data['outputs'], positive_data['outputs']);
    data['outputs'] = transform_to_one_hot(data['outputs']);


    data['test_inputs'] = np.append(test_negative_data['inputs'],test_positive_data['inputs'],axis=0);
    data['test_sentences'] = np.append(test_negative_data['sentences'],test_positive_data['sentences'],axis=0);
    data['test_outputs'] = np.append(test_negative_data['outputs'], positive_data['outputs']);
    data['test_outputs'] = transform_to_one_hot(data['test_outputs']);

    shuffle_data(data);

    return data;

def load_data():
    pickle_path= basePath + "/train_test.p";
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
    dense1 = tflearn.fully_connected(input_layer, num_cols*2, activation='relu',regularizer='L2', weight_decay=0.001);
    dense2 = tflearn.fully_connected(dense1, num_cols, activation='relu',regularizer='L2', weight_decay=0.001);
    #dense3 = tflearn.fully_connected(dense2, num_cols, activation='tanh', regularizer='L2', weight_decay=0.001);
    softmax = tflearn.fully_connected(dense2, n_classes, activation='softmax');
    # Regression using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000);
    #top_k = tflearn.metrics.Top_k(3);
    acc=tflearn.metrics.Accuracy();
    net = tflearn.regression(softmax, optimizer=sgd, metric=acc,loss='categorical_crossentropy');
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=3, best_checkpoint_path=modelPath+"/checkpoint", best_val_accuracy=.6)
    return model;

def test_model(model, tagged_data):
    labels = ['Negative', 'Neutral', 'Positive'];
    correct = 0;
    test_count=len(tagged_data['test_sentences'])
    for i in range(0, test_count):
        sent = tagged_data['test_sentences'][i].rstrip();
        vect = tagged_data['test_inputs'][i];
        expected_result = tagged_data['test_outputs'][i];
        prediction = model.predict([vect]);
        result = np.argmax(prediction);
        if (result == np.argmax(expected_result)):
            correct = correct + 1;
        label = labels[result];
        print(sent, label, prediction, expected_result);
    print("Correct predictions == ", correct, " out of :: ",test_count)
    print("% correct :: ", (correct/test_count)*100)
    return;

def train_or_load_model(tagged_data):
    model = build_network(num_cols, n_classes);
    model_file = modelPath + "/model/sentiment.tfl";
    serving_path= modelPath + "/model/serving_model";
    #Check if model exists
    if(os.path.isfile(model_file+".meta")):
        print("Loading model from :: ",model_file)
        model.load(model_file);
    else:
        start_time=time.time();
        model.fit(tagged_data['inputs'], tagged_data['outputs'], n_epoch=80,  batch_size=100,
                  validation_set=(tagged_data['test_inputs'], tagged_data['test_outputs']),
                  show_metric=True, run_id="yelp_200k");
        model.save(model_file);
        finish_time = time.time();
        elapsed_time=finish_time-start_time;
        print("Time taken for training the model :: ",elapsed_time)
        print("Saving model bundle to :: ",serving_path, " TAG :: ",tf.saved_model.tag_constants.SERVING);
        saver = tf.train.Saver(sharded=True)
        builder = saved_model_builder.SavedModelBuilder(serving_path)
        builder.add_meta_graph_and_variables(
             model.session,
             tags=[tf.saved_model.tag_constants.SERVING]
        )
        builder.save()
    return model;

tagged_data = load_data();

num_rows = tagged_data['inputs'].shape[0];
num_cols=tagged_data['inputs'].shape[1];
print("Training examples= ",num_rows)
print("Training attributes= ",num_cols)
n_classes = 3 #

model=train_or_load_model(tagged_data);

test_model(model,tagged_data);
