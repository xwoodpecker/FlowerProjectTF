import os

import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow import feature_column
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tqdm import tqdm
import collections

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# read the dataset from Drive
df = pd.read_csv("../HumanMobilityPredictionMA/4square/processed_transformed.csv")

count = df.user_id.value_counts()

idx = count.loc[count.index[:5]].index  # count >= 1000
df = df.loc[df.user_id.isin(idx)]

# List the df for each user
users_locations = []

# For each user
for user_id in tqdm(idx):
    users_locations.append(df.loc[df.user_id == user_id].copy())

# List the dfs fo train, val and test for each user
users_locations_train = []
users_locations_val = []
users_locations_test = []

for user_df in users_locations:
    # Split in train, test and validation
    train, test = train_test_split(user_df, test_size=0.2, shuffle=False)
    train, val = train_test_split(train, test_size=0.2, shuffle=False)

    # Append the sets
    users_locations_train.append(train)
    users_locations_val.append(val)
    users_locations_test.append(test)

# Merge back the dataframes
df_train = pd.concat(users_locations_train)

# Merge back the dataframes
df_val = pd.concat(users_locations_val)

# Merge back the dataframes
df_test = pd.concat(users_locations_test)

user_ids = df_train.user_id.unique()

columns_names = df_train.columns.values #todo this is wrong now, column order has been changed
columns_names = np.delete(columns_names, np.where(columns_names == 'user_id'))
print(columns_names)

# List of numerical column names
numerical_column_names = ['clock_sin', 'clock_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_day_sin',
                          'week_day_cos']

NUM_CLIENTS = user_ids.size
NUM_EPOCHS = 4
BATCH_SIZE = 16
# SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 5


# Split the data into chunks of N
def split_data(N):
    # dictionary of list of df
    df_dictionary = {}

    for uid in tqdm(user_ids):
        # Get the records of the user
        user_df_train = df_train.loc[df_train.user_id == uid].copy()
        user_df_val = df_val.loc[df_val.user_id == uid].copy()
        user_df_test = df_test.loc[df_test.user_id == uid].copy()

        # Get a list of dataframes of length N records
        user_list_train = [user_df_train[i:i + N] for i in range(0, user_df_train.shape[0], N)]
        user_list_val = [user_df_val[i:i + N] for i in range(0, user_df_val.shape[0], N)]
        user_list_test = [user_df_test[i:i + N] for i in range(0, user_df_test.shape[0], N)]

        # Save the list of dataframes into a dictionary
        df_dictionary[uid] = {
            'train': user_list_train,
            'val': user_list_val,
            'test': user_list_test
        }

    return df_dictionary


# Takes a dictionary with train, validation and test sets and the desired set type
def create_clients_dict(df_dictionary, set_type, N):
    dataset_dict = {}

    for uid in tqdm(user_ids):

        c_data = collections.OrderedDict()
        values = df_dictionary[uid][set_type]

        # If the last dataframe of the list is not complete # todo: test this "diff" -> does it even work as  intended???
        if len(values[-1]) < N:
            diff = 1
        else:
            diff = 0

        if len(values) > 0:
            # Create the dictionary to create a clientData
            for header in columns_names:
                c_data[header] = [values[i][header].values for i in range(0, len(values) - diff)]
            dataset_dict[uid] = c_data

    return dataset_dict


# preprocess dataset to tf format
def preprocess(dataset, N):
    def batch_format_fn(element):
        x = collections.OrderedDict()

        for name in columns_names:
            x[name] = tf.reshape(element[name][:, :-1], [-1, N - 1])

        y = tf.reshape(element[columns_names[0]][:, 1:], [-1, N - 1])

        return collections.OrderedDict(x=x, y=y)

    return dataset.repeat(4).batch(16, drop_remainder=True).map(batch_format_fn).prefetch(5)


def select(dict, client_id):
    tensor_slices = dict[client_id]
    if tensor_slices:
        return tf.data.Dataset.from_tensor_slices(tensor_slices)
    else:
        raise ValueError('No data found for client {}'.format(client_id))


# create federated data for every client
def make_federated_data(client_dict, client_ids, N):
    return [
        preprocess(select(client_dict, x), N)
        for x in tqdm(client_ids)
    ]


# Create a model
def create_keras_model(number_of_places, N, batch_size):
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
        'cat_id', [i for i in range(number_of_places)])

    # Embed the one-hot encoding # embedding_dim -> todo: hyperparameter
    category_embed = feature_column.embedding_column(category_one_hot, embedding_dim)

    # With an input sequence we can't use the DenseFeature layer, we need to use the SequenceFeatures
    sequence_features, sequence_length = tf.keras.experimental.SequenceFeatures(category_embed)(sequence_input)

    input_sequence = l.Concatenate(axis=2)([sequence_features] + num_features)

    # Rnn # rnn_units -> todo: hyperparameter
    recurrent = l.GRU(rnn_units,
                      batch_size=batch_size,  # in case of stateful
                      dropout=0.3,
                      return_sequences=True,
                      stateful=True,
                      recurrent_initializer='glorot_uniform')(input_sequence)

    # Last layer with an output for each places
    dense_1 = layers.Dense(number_of_places)(recurrent)

    # Softmax output layer
    output = l.Softmax()(dense_1)

    # To return the Model, we need to define it's inputs and outputs
    # In out case, we need to list all the input layers we have defined
    inputs = list(feature_inputs.values()) + list(sequence_input.values())

    # Return the Model
    return tf.keras.Model(inputs=inputs, outputs=output)


N = 17
# Generate the dictionaries for each set
df_dict = split_data(N)
clients_train_dict = create_clients_dict(df_dict, 'train', N)
clients_val_dict = create_clients_dict(df_dict, 'val', N)
clients_test_dict = create_clients_dict(df_dict, 'test', N)

# Select the clients
sample_clients = [293, 185, 354, 315, 84]

federated_train_data = make_federated_data(clients_train_dict, sample_clients, N)
federated_test_data = make_federated_data(clients_val_dict, sample_clients, N)
federated_test_data = make_federated_data(clients_test_dict, sample_clients, N)

# All the different places in the dataset
categories = df.cat_id

# Length of the vocabulary of places
vocab_size = categories.nunique()

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 256

# Create the model
model = create_keras_model(vocab_size, N, 16)

# Define an optimizer
adam = tf.keras.optimizers.Adam(lr=0.002)

# Compile the model with optimizer, loss and metrics
model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])


# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(federated_train_data, epochs=1, batch_size=32)
        return model.get_weights(), len(federated_train_data), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(federated_test_data)
        return loss, len(federated_test_data), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
