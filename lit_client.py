import collections
import functools
import os
import sys
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import flwr as fl

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# read the dataset from Drive
df = pd.read_csv("../HumanMobilityPredictionMA/4square/processed_transformed.csv")

categories = df.cat_id
vocab_size = categories.nunique()

# Select the clients
sample_clients = [293, 185, 354, 315, 84]
if len(sys.argv) > 1:
    index = int(sys.argv[1])
else:
    index = 0

client_id = sample_clients[index]

user_df = df.loc[df.user_id == client_id].copy()
user_df.drop(['user_id'], axis=1, inplace=True)

df_train, df_test = train_test_split(user_df, test_size=0.2, shuffle=False)
df_train, df_val = train_test_split(df_train, test_size=0.2, shuffle=False)

columns_names = df_train.columns.values  # todo this is wrong now, column order has been changed
columns_names = np.delete(columns_names, np.where(columns_names == 'user_id'))
print(columns_names)

# List of numerical column names
numerical_column_names = ['clock_sin', 'clock_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_day_sin',
                          'week_day_cos']

NUM_EPOCHS = 4


def sliding_window(arr, N):
    """
    Splits an array into a list of subarrays using a sliding window approach.

    Parameters:
        arr: a numpy array
        N: the number of elements in each subarray

    Returns:
        A list of numpy arrays.
    """
    # Get the number of subarrays
    num_subarrays = arr.shape[0] - N + 1

    # Create an empty list to store the subarrays
    subarrays = []

    # Loop through the array and create the subarrays
    for i in range(num_subarrays):
        subarray = arr[i:i + N]
        subarrays.append(subarray)

    return subarrays


# Split the data into chunks of N
def split_data_sliding(N):
    return sliding_window(df_train, N), sliding_window(df_val, N), sliding_window(df_test, N)


# Takes a dictionary with train, validation and test sets and the desired set type
def create_dataset(my_list, N):
    input_dict = collections.OrderedDict()

    # If the last dataframe of the list is not complete # todo: test this "diff" -> does it even work as  intended???
    if len(my_list[-1]) < N:
        diff = 1
    else:
        diff = 0

    if len(my_list) > 0:
        # Create the dictionary to create a clientData
        for header in columns_names:
            input_dict[header] = [my_list[i][header].values[:-1] for i in range(0, len(my_list) - diff)]

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_dict, np.array([my_list[i][columns_names[0]].values[1:] for i in range(0, len(my_list) - diff)])))

    return dataset


# Create a model
def create_keras_model(vocab_size, N, batch_size):
    # Shortcut to the layers package
    l = tf.keras.layers

    # List of numeric feature columns to pass to the DenseLayer
    numeric_feature_columns = []

    # Handling numerical columns
    for header in numerical_column_names:
        # Append all the numerical columns defined into the list
        numeric_feature_columns.append(feature_column.numeric_column(header, shape=N - 1))

    feature_inputs = {}
    for c_name in numerical_column_names:
        feature_inputs[c_name] = tf.keras.Input((N - 1,), batch_size=batch_size, name=c_name)

    # We cannot use an array of features as always because we have sequences
    # We have to do one by one in order to match the shape
    num_features = []
    for c_name in numerical_column_names:
        f = feature_column.numeric_column(c_name, shape=(N - 1))
        feature = l.DenseFeatures(f)(feature_inputs)
        feature = tf.expand_dims(feature, -1)
        num_features.append(feature)

    # Declare the dictionary for the categories sequence as before # todo: refactor
    sequence_input = {
        'cat_id': tf.keras.Input((N - 1,), batch_size=batch_size, dtype=tf.dtypes.int32, name='cat_id')
        # add batch_size=batch_size in case of stateful GRU
    }

    # Handling the categorical feature sequence using one-hot
    category_one_hot = feature_column.sequence_categorical_column_with_vocabulary_list(
        'cat_id', [i for i in range(vocab_size)])

    # one-hot encoding
    category_indicator = feature_column.indicator_column(category_one_hot)  # todo: check

    # With an input sequence we can't use the DenseFeature layer, we need to use the SequenceFeatures
    sequence_features, sequence_length = tf.keras.experimental.SequenceFeatures(category_indicator)(sequence_input)

    input_sequence = l.Concatenate(axis=2)([sequence_features] + num_features)

    # Rnn
    recurrent = l.GRU(128,  # rnn_units -> todo: hyperparameter
                      batch_size=batch_size,  # in case of stateful
                      return_sequences=True,
                      stateful=True,
                      activation='relu',
                      recurrent_initializer='glorot_uniform')(input_sequence)

    recurrent_2 = l.GRU(64,
                        batch_size=batch_size,  # in case of stateful
                        return_sequences=True,
                        dropout=0.5,
                        stateful=True,
                        activation='relu',
                        recurrent_initializer='glorot_uniform')(recurrent)

    # Softmax output layer
    # Last layer with an output for each places
    output = layers.Dense(vocab_size, activation='softmax')(recurrent_2)

    # To return the Model, we need to define it's inputs and outputs
    # In out case, we need to list all the input layers we have defined
    inputs = list(feature_inputs.values()) + list(sequence_input.values())

    # Return the Model
    return tf.keras.Model(inputs=inputs, outputs=output)


n = 17
list_train, list_val, list_test = split_data_sliding(n)

train_dataset = create_dataset(list_train, n)
val_dataset = create_dataset(list_val, n)
test_dataset = create_dataset(list_test, n)

# Batch size
BATCH_SIZE = 16

train_batch = train_dataset.batch(BATCH_SIZE, drop_remainder=True)  # .shuffle(BUFFER_SIZE)
val_batch = val_dataset.batch(BATCH_SIZE, drop_remainder=True)  # .shuffle(BUFFER_SIZE)
test_batch = test_dataset.batch(BATCH_SIZE, drop_remainder=True)  # .shuffle(BUFFER_SIZE)

# Get the model and compile it
model = create_keras_model(vocab_size, n, batch_size=BATCH_SIZE)
# Define the optimizer
adam = tf.keras.optimizers.Adam(lr=0.002)

# Compile the model
model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy())


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return model.get_weights()

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        model.set_weights(parameters)
        N_EPOCHS = 1
        # Fit the model
        model.fit(train_batch, validation_data=val_batch, epochs=N_EPOCHS)
        return model.get_weights(), len(train_batch), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        model.set_weights(parameters)
        loss, sparse_categorical_accuracy = model.evaluate(test_batch)
        print(loss, len(test_batch), {"sparse_categorical_accuracy": sparse_categorical_accuracy})
        return loss, len(test_batch), {"sparse_categorical_accuracy": sparse_categorical_accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client(index))
