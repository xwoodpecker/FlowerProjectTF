import os

import flwr as fl
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow import feature_column
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import collections
import tensorflow_datasets as tfds

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# read the dataset from Drive
df = pd.read_csv("../HumanMobilityPredictionMA/4square/processed_v2.csv")
df['clock_sin'] = np.sin(2 * np.pi * df['clock'] / 24.0)
df['clock_cos'] = np.cos(2 * np.pi * df['clock'] / 24.0)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 30.0)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 30.0)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
df['week_day_sin'] = np.sin(2 * np.pi * df['week_day'] / 7.0)
df['week_day_cos'] = np.cos(2 * np.pi * df['week_day'] / 7.0)

df['cat_id'] = df.groupby('uber_category', sort=False).ngroup()

df.drop(['clock', 'day', 'month', 'week_day', 'uber_category'], axis=1, inplace=True)
count = df.user_id.value_counts()

idx = count.loc[count.index[:100]].index  # count >= 1000
df = df.loc[df.user_id.isin(idx)]
# List the df for each user
users_locations = []

# For each user
for user_id in tqdm(idx):
    users_locations.append(df.loc[df.user_id == user_id].copy())

sizes = []
# Number of locations for each user
for user_df in users_locations:
    sizes.append(user_df.shape[0])

print('Mean number of locations: ', np.mean(np.array(sizes)))
print('Max number of locations: ', np.max(np.array(sizes)))
print('Min number of locations: ', np.min(np.array(sizes)))

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

sizes = []
# Number of locations for each user in the validation set
for user_df in users_locations_val:
    sizes.append(user_df.shape[0])

print('Mean number of locations: ', np.mean(np.array(sizes)))
print('Max number of locations: ', np.max(np.array(sizes)))
print('Min number of locations: ', np.min(np.array(sizes)))

# Merge back the dataframes
df_train = pd.concat(users_locations_train)

# Merge back the dataframes
df_val = pd.concat(users_locations_val)

# Merge back the dataframes
df_test = pd.concat(users_locations_test)

user_ids = df_train.user_id.unique()

N = 17

# dictionary of list of df
df_dictionary = {}

for user_id in tqdm(user_ids):
    # Get the records of the user
    user_df_train = df_train.loc[df_train.user_id == user_id].copy()
    user_df_val = df_val.loc[df_val.user_id == user_id].copy()
    user_df_test = df_test.loc[df_test.user_id == user_id].copy()

    # Get a list of dataframes of length N records
    user_list_train = [user_df_train[i:i + N] for i in range(0, user_df_train.shape[0], N)]
    user_list_val = [user_df_val[i:i + N] for i in range(0, user_df_val.shape[0], N)]
    user_list_test = [user_df_test[i:i + N] for i in range(0, user_df_test.shape[0], N)]

    # Save the list of dataframes into a dictionary
    df_dictionary[user_id] = {
        'train': user_list_train,
        'val': user_list_val,
        'test': user_list_test
    }

# Create the dictionary to create a clientData
columns_names = df_train.columns.values[1:]


# Takes a dictionary with train, validation an test sets and the desired set type
def create_clients_dict(df_dictionary, set_type):
    dataset_dict = {}

    for user_id in tqdm(user_ids):

        c_data = collections.OrderedDict()
        values = df_dictionary[user_id][set_type]

        # If the last dataframe of the list is not complete
        if len(values[-1]) < N:
            diff = 1
        else:
            diff = 0

        if len(values) > 0:
            for header in columns_names:
                # c_data[header] = values[header].values.tolist()
                c_data[header] = [values[i][header].values for i in range(0, len(values) - diff)]  # [:-1]
                # c_data['y'] = values['dropoff_location_id'].values.tolist()
            dataset_dict[user_id] = c_data

    return dataset_dict


# Generate the dictionaries for each set
clients_train_dict = create_clients_dict(df_dictionary, 'train')
clients_val_dict = create_clients_dict(df_dictionary, 'val')
clients_test_dict = create_clients_dict(df_dictionary, 'test')

# Convert the dictionary to a dataset
client_train_data = tff.simulation.FromTensorSlicesClientData(clients_train_dict)
client_val_data = tff.simulation.FromTensorSlicesClientData(clients_val_dict)
client_test_data = tff.simulation.FromTensorSlicesClientData(clients_test_dict)

NUM_CLIENTS = user_ids.size
NUM_EPOCHS = 4
BATCH_SIZE = 16
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 5


def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=collections.OrderedDict(
                category=tf.reshape(element['cat_id'][:, :-1], [-1, N - 1]),
                clock_sin=tf.reshape(element['clock_sin'][:, :-1], [-1, N - 1]),
                clock_cos=tf.reshape(element['clock_cos'][:, :-1], [-1, N - 1]),
                day_sin=tf.reshape(element['day_sin'][:, :-1], [-1, N - 1]),
                day_cos=tf.reshape(element['day_cos'][:, :-1], [-1, N - 1]),
                month_sin=tf.reshape(element['month_sin'][:, :-1], [-1, N - 1]),
                month_cos=tf.reshape(element['month_cos'][:, :-1], [-1, N - 1]),
                week_day_sin=tf.reshape(element['week_day_sin'][:, :-1], [-1, N - 1]),
                week_day_cos=tf.reshape(element['week_day_cos'][:, :-1], [-1, N - 1]),
            ),
            y=tf.reshape(element['cat_id'][:, 1:], [-1, N - 1]))

    return dataset.repeat(NUM_EPOCHS).batch(BATCH_SIZE, drop_remainder=True).map(batch_format_fn).prefetch(
        PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in tqdm(client_ids)
    ]


sample_clients = client_train_data.client_ids[0:NUM_CLIENTS]

# Federate the clients datasets
federated_train_data = make_federated_data(client_train_data, sample_clients)

print(federated_train_data)

federated_val_data = make_federated_data(client_val_data, sample_clients)
federated_test_data = make_federated_data(client_test_data, sample_clients)

fed_train = tfds.as_numpy(federated_train_data[0])
fed_df = pd.DataFrame(fed_train)
fed_train_X = fed_df['x']
fed_train_Y = fed_df['y']

fed_test = tfds.as_numpy(federated_test_data[0])
fed_df = pd.DataFrame(fed_test)
fed_test_X = fed_df['x']
fed_test_Y = fed_df['y']

indices = df.cat_id.values

# Length of the vocabulary in chars
vocab_size = int(np.max(indices) + 1)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 256

# List of numerical column names
numerical_column_names = ['clock_sin', 'clock_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_day_sin',
                          'week_day_cos']

# Number of different places
number_of_places = indices.max() + 1

NUM_EPOCHS = 4
BATCH_SIZE = 16
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 5


# Create a model
def create_keras_model(number_of_places, batch_size):
    # Shortcut to the layers package
    layers = tf.keras.layers

    # List of numeric feature columns to pass to the DenseLayer
    numeric_feature_columns = []

    # Handling numerical columns
    for header in numerical_column_names:
        # Append all the numerical columns defined into the list
        numeric_feature_columns.append(feature_column.numeric_column(header, shape=N - 1))

    # Now we need to define an input dictionary.
    # Where the keys are the column names
    # This is a model with multiple inputs, so we need to declare and input layer for each feature
    feature_inputs = {
        'clock_sin': tf.keras.Input((N - 1,), batch_size=batch_size, name='clock_sin'),
        'clock_cos': tf.keras.Input((N - 1,), batch_size=batch_size, name='clock_cos'),
        'day_sin': tf.keras.Input((N - 1,), batch_size=batch_size, name='day_sin'),
        'day_cos': tf.keras.Input((N - 1,), batch_size=batch_size, name='day_cos'),
        'month_sin': tf.keras.Input((N - 1,), batch_size=batch_size, name='month_sin'),
        'month_cos': tf.keras.Input((N - 1,), batch_size=batch_size, name='month_cos'),
        'week_day_sin': tf.keras.Input((N - 1,), batch_size=batch_size, name='week_day_sin'),
        'week_day_cos': tf.keras.Input((N - 1,), batch_size=batch_size, name='week_day_cos'),
    }

    # We cannot use an array of features as always because we have sequences and we cannot match the shape otherwise
    # We have to do one by one
    clock_sin = feature_column.numeric_column("clock_sin", shape=(N - 1))
    clock_sin_feature = layers.DenseFeatures(clock_sin)(feature_inputs)

    clock_cos = feature_column.numeric_column("clock_cos", shape=(N - 1))
    clock_cos_feature = layers.DenseFeatures(clock_cos)(feature_inputs)

    day_sin = feature_column.numeric_column("day_sin", shape=(N - 1))
    day_sin_feature = layers.DenseFeatures(day_sin)(feature_inputs)

    day_cos = feature_column.numeric_column("day_cos", shape=(N - 1))
    day_cos_feature = layers.DenseFeatures(day_cos)(feature_inputs)

    month_sin = feature_column.numeric_column("month_sin", shape=(N - 1))
    month_sin_feature = layers.DenseFeatures(month_sin)(feature_inputs)

    month_cos = feature_column.numeric_column("month_cos", shape=(N - 1))
    month_cos_feature = layers.DenseFeatures(month_cos)(feature_inputs)

    week_day_sin = feature_column.numeric_column("week_day_sin", shape=(N - 1))
    week_day_sin_feature = layers.DenseFeatures(week_day_sin)(feature_inputs)

    week_day_cos = feature_column.numeric_column("week_day_cos", shape=(N - 1))
    week_day_cos_feature = layers.DenseFeatures(week_day_cos)(feature_inputs)

    # We have also to add a dimension to then concatenate
    clock_sin_feature = tf.expand_dims(clock_sin_feature, -1)
    clock_cos_feature = tf.expand_dims(clock_cos_feature, -1)
    day_sin_feature = tf.expand_dims(day_sin_feature, -1)
    day_cos_feature = tf.expand_dims(day_cos_feature, -1)
    month_sin_feature = tf.expand_dims(month_sin_feature, -1)
    month_cos_feature = tf.expand_dims(month_cos_feature, -1)
    week_day_sin_feature = tf.expand_dims(week_day_sin_feature, -1)
    week_day_cos_feature = tf.expand_dims(week_day_cos_feature, -1)

    # Declare the dictionary for the places sequence as before
    sequence_input = {
        'category': tf.keras.Input((N - 1,), batch_size=batch_size, dtype=tf.dtypes.int32, name='category')
        # add batch_size=batch_size in case of stateful GRU
    }

    # Handling the categorical feature sequence using one-hot
    category_one_hot = feature_column.sequence_categorical_column_with_vocabulary_list(
        'category', [i for i in range(number_of_places)])

    # Embed the one-hot encoding
    category_embed = feature_column.embedding_column(category_one_hot, embedding_dim)

    # With an input sequence we can't use the DenseFeature layer, we need to use the SequenceFeatures
    sequence_features, sequence_length = tf.keras.experimental.SequenceFeatures(category_embed)(sequence_input)

    input_sequence = layers.Concatenate(axis=2)(
        [sequence_features, clock_sin_feature, clock_cos_feature, day_sin_feature, day_cos_feature, month_sin_feature,
         month_cos_feature, week_day_sin_feature, week_day_cos_feature])

    # Rnn
    recurrent = layers.GRU(rnn_units,
                           batch_size=batch_size,  # in case of stateful
                           dropout=0.3,
                           return_sequences=True,
                           stateful=True,
                           recurrent_initializer='glorot_uniform')(input_sequence)

    # Last layer with an output for each places
    dense_1 = layers.Dense(number_of_places)(recurrent)

    # Softmax output layer
    output = layers.Softmax()(dense_1)

    # To return the Model, we need to define it's inputs and outputs
    # In out case, we need to list all the input layers we have defined
    inputs = list(feature_inputs.values()) + list(sequence_input.values())

    # Return the Model
    return tf.keras.Model(inputs=inputs, outputs=output)


# Load model and data (MobileNetV2, CIFAR-10)
keras_model = create_keras_model(
    number_of_places,
    batch_size=BATCH_SIZE
)
keras_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)



# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return keras_model.get_weights()

    def fit(self, parameters, config):
        keras_model.set_weights(parameters)
        keras_model.fit(federated_train_data['x'], federated_train_data['y'], epochs=1, batch_size=32)
        return keras_model.get_weights(), len(federated_train_data['x']), {}

    def evaluate(self, parameters, config):
        keras_model.set_weights(parameters)
        loss, accuracy = keras_model.evaluate(federated_test_data['x'], federated_test_data['y'])
        return loss, len(federated_test_data['x']), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
