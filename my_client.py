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


# these functions are copied from a newer version and is needed to run the code below:
def timeseries_dataset_from_array(
        data,
        targets,
        sequence_length,
        sequence_stride=1,
        sampling_rate=1,
        batch_size=128,
        shuffle=False,
        seed=None,
        start_index=None,
        end_index=None,
):
    if start_index:
        if start_index < 0:
            raise ValueError(
                "`start_index` must be 0 or greater. Received: "
                f"start_index={start_index}"
            )
        if start_index >= len(data):
            raise ValueError(
                "`start_index` must be lower than the length of the "
                f"data. Received: start_index={start_index}, for data "
                f"of length {len(data)}"
            )
    if end_index:
        if start_index and end_index <= start_index:
            raise ValueError(
                "`end_index` must be higher than `start_index`. "
                f"Received: start_index={start_index}, and "
                f"end_index={end_index} "
            )
        if end_index >= len(data):
            raise ValueError(
                "`end_index` must be lower than the length of the "
                f"data. Received: end_index={end_index}, for data of "
                f"length {len(data)}"
            )
        if end_index <= 0:
            raise ValueError(
                "`end_index` must be higher than 0. "
                f"Received: end_index={end_index}"
            )

    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(
            "`sampling_rate` must be higher than 0. Received: "
            f"sampling_rate={sampling_rate}"
        )
    if sampling_rate >= len(data):
        raise ValueError(
            "`sampling_rate` must be lower than the length of the "
            f"data. Received: sampling_rate={sampling_rate}, for data "
            f"of length {len(data)}"
        )
    if sequence_stride <= 0:
        raise ValueError(
            "`sequence_stride` must be higher than 0. Received: "
            f"sequence_stride={sequence_stride}"
        )
    if sequence_stride >= len(data):
        raise ValueError(
            "`sequence_stride` must be lower than the length of the "
            f"data. Received: sequence_stride={sequence_stride}, for "
            f"data of length {len(data)}"
        )

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory
    # usage).
    num_seqs = end_index - start_index - (sequence_length * sampling_rate) + 1
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = "int32"
    else:
        index_dtype = "int64"

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)
    ).map(
        lambda i, positions: tf.range(
            positions[i],
            positions[i] + sequence_length * sampling_rate,
            sampling_rate,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)
        ).map(
            lambda i, positions: positions[i],
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index
        )
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset


def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


# read the dataset from Drive
df = pd.read_csv("../HumanMobilityPredictionMA/4square/processed_transformed_locations.csv")

locations = df.location_id
vocab_size = locations.nunique()

# Select the clients
sample_clients = [293, 185, 354, 315, 84]
if len(sys.argv) > 1:
    index = int(sys.argv[1])
else:
    index = 0

client_id = sample_clients[index]

user_df = df.loc[df.user_id == client_id].copy()
user_df.drop(['user_id'], axis=1, inplace=True)

n = len(user_df)
train_df = user_df[0:int(n * 0.7)]
val_df = user_df[int(n * 0.7):int(n * 0.9)]
test_df = user_df[int(n * 0.9):]


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


w = WindowGenerator(input_width=16, label_width=1, shift=1,
                    label_columns=['location_id'])


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=8, )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


model = tf.keras.Sequential()
model.add(tf.keras.layers.GRU(128, return_sequences=True, input_shape=(16, 9), activation='relu'))
model.add(tf.keras.layers.GRU(64, input_shape=(16, 9), activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(vocab_size))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.002))


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
        model.fit(w.train, epochs=4, validation_data=w.val)
        #model.fit(w.train, epochs=1)
        return model.get_weights(), len(w.train), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        model.set_weights(parameters)
        loss, sparse_categorical_accuracy = model.evaluate(w.test)
        return loss, len(w.test), {"sparse_categorical_accuracy": sparse_categorical_accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client(index))
