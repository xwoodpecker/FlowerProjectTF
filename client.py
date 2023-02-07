import os

import flwr as fl
import tensorflow as tf
from tensorflow import feature_column
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tqdm import tqdm
import collections
import sys

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# read the dataset from Drive
df = pd.read_csv("../HumanMobilityPredictionMA/4square/processed_transformed_old.csv")

columns_titles = ["user_id","cat_id","clock_sin","clock_cos","day_sin","day_cos","month_sin","month_cos","week_day_sin","week_day_cos"]

columns_names = ["cat_id","clock_sin","clock_cos","day_sin","day_cos","month_sin","month_cos","week_day_sin","week_day_cos"]

df=df.reindex(columns=columns_titles)

count = df.user_id.value_counts()

idx = count.loc[count.index[:5]].index  # count >= 1000
df = df.loc[df.user_id.isin(idx)]

# Select the clients
sample_clients = [293, 185, 354, 315, 84]
if len(sys.argv) > 1:
    index = int(sys.argv[1])
else:
    index = 0

client_id = sample_clients[index]

user_df = df.loc[df.user_id == client_id].copy()

df_train, df_test = train_test_split(user_df, test_size=0.2, shuffle=False)
#train, val = train_test_split(train, test_size=0.2, shuffle=False)

# Split the data into chunks of N+1
N = 17

list_train = [df_train[i:i+N] for i in range(0, df_train.shape[0], N)]
list_test = [df_test[i:i+N] for i in range(0, df_test.shape[0], N)]


def create_dict(values, N):

    c_data = collections.OrderedDict()

    # If the last dataframe of the list is not complete # todo: test this "diff" -> does it even work as  intended???
    if len(values[-1]) < N:
      diff = 1
    else:
      diff = 0

    if len(values) > 0:
      # Create the dictionary to create a clientData
      for header in columns_names:
        c_data[header] = [values[i][header].values for i in range(0, len(values)-diff)]
      dataset = c_data

    return dataset

train_dict = create_dict(list_train, N)
test_dict = create_dict(list_test, N)

train_data = tf.data.Dataset.from_tensor_slices(train_dict)
test_data = tf.data.Dataset.from_tensor_slices(test_dict)

print('example  --->    lul')
print(next(iter(train_data)))

# preprocess dataset to tf format
def preprocess(dataset, N):

  def batch_format_fn(element):

    x=collections.OrderedDict()

    for name in columns_names:
      x[name]=tf.reshape(element[name][:, :-1], [-1, N-1])

    y=tf.reshape(element[columns_names[0]][:, 1:], [-1, N-1])

    return collections.OrderedDict(x=x, y=y)

  #return dataset.repeat(4).batch(16, drop_remainder=True).map(batch_format_fn).prefetch(5)
  return dataset.repeat(4).batch(16, drop_remainder=True).map(batch_format_fn)

batch_size = 16

train_ds = preprocess(train_data, N)
sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(train_ds)))

print('sample_batch   --->   ', sample_batch['x']['cat_id'].shape)

# Create the training dataset
print('o2 ---------- XD')
print(train_ds)

# Create the test dataset
test_ds = preprocess(test_data, N)

# All the different places in the dataset
categories = df.cat_id

# Length of the vocabulary of places
vocab_size = categories.nunique()

print(vocab_size)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 256

# List of numerical column names
numerical_column_names = ['clock_sin', 'clock_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                          'week_day_sin', 'week_day_cos']


# Create a model
def create_keras_model(vocab_size, batch_size):
    # Shortcut to the layers package
    l = tf.keras.layers

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

    # We cannot use anarray of features as always because we have sequences and we cannot match the shape otherwise
    # We have to do one by one
    clock_sin = feature_column.numeric_column("clock_sin", shape=(N - 1))
    clock_sin_feature = l.DenseFeatures(clock_sin)(feature_inputs)

    clock_cos = feature_column.numeric_column("clock_cos", shape=(N - 1))
    clock_cos_feature = l.DenseFeatures(clock_cos)(feature_inputs)

    day_sin = feature_column.numeric_column("day_sin", shape=(N - 1))
    day_sin_feature = l.DenseFeatures(day_sin)(feature_inputs)

    day_cos = feature_column.numeric_column("day_cos", shape=(N - 1))
    day_cos_feature = l.DenseFeatures(day_cos)(feature_inputs)

    month_sin = feature_column.numeric_column("month_sin", shape=(N - 1))
    month_sin_feature = l.DenseFeatures(month_sin)(feature_inputs)

    month_cos = feature_column.numeric_column("month_cos", shape=(N - 1))
    month_cos_feature = l.DenseFeatures(month_cos)(feature_inputs)

    week_day_sin = feature_column.numeric_column("week_day_sin", shape=(N - 1))
    week_day_sin_feature = l.DenseFeatures(week_day_sin)(feature_inputs)

    week_day_cos = feature_column.numeric_column("week_day_cos", shape=(N - 1))
    week_day_cos_feature = l.DenseFeatures(week_day_cos)(feature_inputs)

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
        'cat_id': tf.keras.Input((N - 1,), batch_size=batch_size, dtype=tf.dtypes.int32, name='cat_id')
        # add batch_size=batch_size in case of stateful GRU
    }

    # Handling the categorical feature sequence using one-hot
    category_one_hot = feature_column.sequence_categorical_column_with_vocabulary_list(
        'cat_id', [i for i in range(vocab_size)])

    # indicator column for one-hot encoding
    indicator_input = feature_column.indicator_column(category_one_hot)

    # With an input sequence we can't use the DenseFeature layer, we need to use the SequenceFeatures
    sequence_features, sequence_length = tf.keras.experimental.SequenceFeatures(indicator_input)(sequence_input)

    input_sequence = l.Concatenate(axis=2)(
        [sequence_features, clock_sin_feature, clock_cos_feature, day_sin_feature, day_cos_feature, month_sin_feature,
         month_cos_feature, week_day_sin_feature, week_day_cos_feature])


    #AAA
    # Rnn
    #recurrent = l.GRU(rnn_units,
    #                  batch_size=batch_size,  # in case of stateful
    #                  dropout=0.3,
    #                  return_sequences=True,
    #                  stateful=True,
    #                  recurrent_initializer='glorot_uniform')(input_sequence)

    # Last layer with an output for each places
    #dense_1 = layers.Dense(vocab_size)(recurrent)

    # Softmax output layer
    #output = l.Softmax()(dense_1)
    #AAA


    #BBB
    # Just a hidden layer with 128 neurons
    dense = l.Dense(128, kernel_initializer='he_uniform', activation='relu')(input_sequence)

    # Uncomment to use one more layer
    dense_1 = l.Dense(128, kernel_initializer='he_uniform', activation='relu')(dense)

    # Last layer with an output for each location
    dense_2 = layers.Dense(vocab_size)(dense_1)

    # Softmax layer
    output = l.Softmax()(dense_2)
    #BBB

    # To return the Model, we need to define it's inputs and outputs
    # In out case, we need to list all the input layers we have defined
    inputs = list(feature_inputs.values()) + list(sequence_input.values())
    print('----> LENGTHHHHH  ', len(inputs))

    # Return the Model
    return tf.keras.Model(inputs=inputs, outputs=output)


# Create the model
model = create_keras_model(vocab_size=vocab_size, batch_size=16)

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
        model.fit(train_ds, epochs=1, batch_size=16)
        return model.get_weights(), len(train_ds), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_ds)
        return loss, len(test_ds), {"accuracy": accuracy}



# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())