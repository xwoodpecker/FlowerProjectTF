import collections
import functools
import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

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

    input_sequence = l.Concatenate(axis=2)([sequence_features] + num_features)

    # Rnn
    recurrent = l.GRU(256,  # rnn_units -> todo: hyperparameter
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

    return tf.keras.Model(inputs=inputs, outputs=output)


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


NUM_EPOCHS = 4
BATCH_SIZE = 16
# SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 5
n = 17

df = pd.read_csv("../HumanMobilityPredictionMA/4square/processed_transformed_old.csv")

# the number of different categories defines the vocabulary size
categories = df.cat_id
vocab_size = categories.nunique()


user_df = df.loc[df.user_id == 293].copy()
print(user_df['cat_id'])

df_train, df_test = train_test_split(user_df, test_size=0.2, shuffle=False)
df_train, df_val = train_test_split(df_train, test_size=0.2, shuffle=False)

user_ids = df_train.user_id.unique()
print('User count ', user_ids.size)

columns_names = df_train.columns.values  # todo this is wrong now, column order has been changed
columns_names = np.delete(columns_names, np.where(columns_names == 'user_id'))
numerical_column_names = ['clock_sin', 'clock_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'week_day_sin',
                          'week_day_cos']

list_train = [df_train[i:i+n] for i in range(0, df_train.shape[0], n)] #todo: sequences can be used like "floating windows"
list_test = [df_test[i:i+n] for i in range(0, df_test.shape[0], n)]

train_dict = create_dict(list_train, n)
test_dict = create_dict(list_test, n)

train_data = tf.data.Dataset.from_tensor_slices(train_dict)
test_data = tf.data.Dataset.from_tensor_slices(test_dict)

train_ds = preprocess(train_data, n)
test_ds = preprocess(test_data, n)

print('Training...')

model = create_keras_model(vocab_size, n, batch_size=BATCH_SIZE)

adam = tf.keras.optimizers.Adam(lr=0.002)

# Compile the model with optimizer, loss and metrics
model.compile(optimizer=adam,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_ds, epochs=1, batch_size=32)

loss, accuracy = model.evaluate(test_ds)
