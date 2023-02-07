import os

import flwr as fl
import tensorflow as tf
from tensorflow import feature_column
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# read the dataset from Drive
df = pd.read_csv("../HumanMobilityPredictionMA/4square/processed_transformed_old.csv")
df.drop(['user_id'], axis=1, inplace=True)

train, test = train_test_split(df, test_size=0.2, shuffle=False)


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=False, batch_size=32):
    dataframe = df.copy()
    labels = dataframe.pop('cat_id')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes

# Create the training dataset
train_ds = df_to_dataset(train, batch_size=batch_size)

# Create the test dataset
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

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
def create_model():
    # List of numeric feature columns to pass to the DenseLayer
    feature_columns = []
    # Shortcut to the layers package
    l = tf.keras.layers

    # Handling numerical columns
    for header in numerical_column_names:
        # Append all the numerical columns defined into the list
        feature_columns.append(feature_column.numeric_column(header))

    # Now we need to define an input dictionary.
    # Keras will receive as input just the 'x', which is a dictionary as well, \
    # Where the keys are the column names
    # This is a model with multiple inputs, so we need to declare and input layer for each feature
    feature_inputs = {
        'clock_sin': tf.keras.Input((1,), name='clock_sin'),
        'clock_cos': tf.keras.Input((1,), name='clock_cos'),
        'day_sin': tf.keras.Input((1,), name='day_sin'),
        'day_cos': tf.keras.Input((1,), name='day_cos'),
        'month_sin': tf.keras.Input((1,), name='month_sin'),
        'month_cos': tf.keras.Input((1,), name='month_cos'),
        'week_day_sin': tf.keras.Input((1,), name='week_day_sin'),
        'week_day_cos': tf.keras.Input((1,), name='week_day_cos'),
    }
    # We declare two DenseFeature layers, one for the numeric columns which do not require\
    # Any training, and one for the Embedding. It is easier to do it like this
    numerical_features = l.DenseFeatures(feature_columns)(feature_inputs)

    # Just a hidden layer with 128 neurons
    dense = l.Dense(128, kernel_initializer='he_uniform', activation='relu')(numerical_features)

    # Uncomment to use one more layer
    dense_1 = l.Dense(128, kernel_initializer='he_uniform', activation='relu')(dense)

    # Last layer with an output for each location
    dense_2 = layers.Dense(vocab_size)(dense_1)

    # Softmax layer
    output = l.Softmax()(dense_2)

    # To return the Model, we need to define it's inputs and outputs
    # In out case, we need to list all the input layers we have defined
    inputs = list(feature_inputs.values())

    # Return the Model
    return tf.keras.Model(inputs=inputs, outputs=output)


# Define the batch_size and split the dataset
batch_size = 64
train_ds = df_to_dataset(train, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Create the model
model = create_model()

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
        model.fit(train_ds, epochs=1, batch_size=32)
        return model.get_weights(), len(train_ds), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_ds)
        return loss, len(test_ds), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
