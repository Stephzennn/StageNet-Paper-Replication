import numpy as np
import argparse
import os
#import imp
import importlib as imp
import re
import keras
import sys
sys.path.append(os.path.abspath('C:/Users/Estif/Desktop/machine_problems/TOP/OMSCS_BIG_DATA_FOR_HEALTHCARE/FinalProject/StageNet-Paper-Replication/mimic3benchmark'))
print("Current Working Directory:", os.getcwd())
'''
from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils
'''
from readers import DecompensationReader
sys.path.append(os.path.abspath('C:/Users/Estif/Desktop/machine_problems/TOP/OMSCS_BIG_DATA_FOR_HEALTHCARE/FinalProject/StageNet-Paper-Replication/mimic3models'))
print("Current Working Directory:", os.getcwd())

from decompensation import utils


#from preprocessing import  Discretizer, Normalizer

import preprocessing
import metrics
import keras_utils
import common_utils

#======================================================================

import numpy as np
import platform
import pickle
import json
import os


class Discretizer:
    def __init__(self, timestep=0.8, store_masks=True, impute_strategy='zero', start_time='zero',
                 config_path=os.path.join(os.path.dirname(__file__), 'resources/discretizer_config.json')):

        with open(config_path) as f:
            config = json.load(f)
            self._id_to_channel = config['id_to_channel']
            self._channel_to_id = dict(zip(self._id_to_channel, range(len(self._id_to_channel))))
            self._is_categorical_channel = config['is_categorical_channel']
            self._possible_values = config['possible_values']
            self._normal_values = config['normal_values']

        self._header = ["Hours"] + self._id_to_channel
        self._timestep = timestep
        self._store_masks = store_masks
        self._start_time = start_time
        self._impute_strategy = impute_strategy

        # for statistics
        self._done_count = 0
        self._empty_bins_sum = 0
        self._unused_data_sum = 0

    def transform(self, X, header=None, end=None):
        if header is None:
            header = self._header
        assert header[0] == "Hours"
        eps = 1e-6

        N_channels = len(self._id_to_channel)
        ts = [float(row[0]) for row in X]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1] + eps

        if self._start_time == 'relative':
            first_time = ts[0]
        elif self._start_time == 'zero':
            first_time = 0
        else:
            raise ValueError("start_time is invalid")

        if end is None:
            max_hours = max(ts) - first_time
        else:
            max_hours = end - first_time

        N_bins = int(max_hours / self._timestep + 1.0 - eps)

        cur_len = 0
        begin_pos = [0 for i in range(N_channels)]
        end_pos = [0 for i in range(N_channels)]
        for i in range(N_channels):
            channel = self._id_to_channel[i]
            begin_pos[i] = cur_len
            if self._is_categorical_channel[channel]:
                end_pos[i] = begin_pos[i] + len(self._possible_values[channel])
            else:
                end_pos[i] = begin_pos[i] + 1
            cur_len = end_pos[i]

        data = np.zeros(shape=(N_bins, cur_len), dtype=float)
        mask = np.zeros(shape=(N_bins, N_channels), dtype=int)
        original_value = [["" for j in range(N_channels)] for i in range(N_bins)]
        total_data = 0
        unused_data = 0

        def write(data, bin_id, channel, value, begin_pos):
            channel_id = self._channel_to_id[channel]
            if self._is_categorical_channel[channel]:
                category_id = self._possible_values[channel].index(value)
                N_values = len(self._possible_values[channel])
                one_hot = np.zeros((N_values,))
                one_hot[category_id] = 1
                for pos in range(N_values):
                    data[bin_id, begin_pos[channel_id] + pos] = one_hot[pos]
            else:
                data[bin_id, begin_pos[channel_id]] = float(value)

        for row in X:
            t = float(row[0]) - first_time
            if t > max_hours + eps:
                continue
            bin_id = int(t / self._timestep - eps)
            assert 0 <= bin_id < N_bins

            for j in range(1, len(row)):
                if row[j] == "":
                    continue
                channel = header[j]
                channel_id = self._channel_to_id[channel]

                total_data += 1
                if mask[bin_id][channel_id] == 1:
                    unused_data += 1
                mask[bin_id][channel_id] = 1

                write(data, bin_id, channel, row[j], begin_pos)
                original_value[bin_id][channel_id] = row[j]

        # impute missing values

        if self._impute_strategy not in ['zero', 'normal_value', 'previous', 'next']:
            raise ValueError("impute strategy is invalid")

        if self._impute_strategy in ['normal_value', 'previous']:
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if self._impute_strategy == 'normal_value':
                        imputed_value = self._normal_values[channel]
                    if self._impute_strategy == 'previous':
                        if len(prev_values[channel_id]) == 0:
                            imputed_value = self._normal_values[channel]
                        else:
                            imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        if self._impute_strategy == 'next':
            prev_values = [[] for i in range(len(self._id_to_channel))]
            for bin_id in range(N_bins-1, -1, -1):
                for channel in self._id_to_channel:
                    channel_id = self._channel_to_id[channel]
                    if mask[bin_id][channel_id] == 1:
                        prev_values[channel_id].append(original_value[bin_id][channel_id])
                        continue
                    if len(prev_values[channel_id]) == 0:
                        imputed_value = self._normal_values[channel]
                    else:
                        imputed_value = prev_values[channel_id][-1]
                    write(data, bin_id, channel, imputed_value, begin_pos)

        empty_bins = np.sum([1 - min(1, np.sum(mask[i, :])) for i in range(N_bins)])
        self._done_count += 1
        self._empty_bins_sum += empty_bins / (N_bins + eps)
        self._unused_data_sum += unused_data / (total_data + eps)

        if self._store_masks:
            data = np.hstack([data, mask.astype(np.float32)])

        # create new header
        new_header = []
        for channel in self._id_to_channel:
            if self._is_categorical_channel[channel]:
                values = self._possible_values[channel]
                for value in values:
                    new_header.append(channel + "->" + value)
            else:
                new_header.append(channel)

        if self._store_masks:
            for i in range(len(self._id_to_channel)):
                channel = self._id_to_channel[i]
                new_header.append("mask->" + channel)

        new_header = ",".join(new_header)

        return (data, new_header)

    def print_statistics(self):
        print("statistics of discretizer:")
        print("\tconverted {} examples".format(self._done_count))
        print("\taverage unused data = {:.2f} percent".format(100.0 * self._unused_data_sum / self._done_count))
        print("\taverage empty  bins = {:.2f} percent".format(100.0 * self._empty_bins_sum / self._done_count))


class Normalizer:
    def __init__(self, fields=None):
        self._means = None
        self._stds = None
        self._fields = None
        if fields is not None:
            self._fields = [col for col in fields]

        self._sum_x = None
        self._sum_sq_x = None
        self._count = 0

    def _feed_data(self, x):
        x = np.array(x)
        self._count += x.shape[0]
        if self._sum_x is None:
            self._sum_x = np.sum(x, axis=0)
            self._sum_sq_x = np.sum(x**2, axis=0)
        else:
            self._sum_x += np.sum(x, axis=0)
            self._sum_sq_x += np.sum(x**2, axis=0)

    def _save_params(self, save_file_path):
        eps = 1e-7
        with open(save_file_path, "wb") as save_file:
            N = self._count
            self._means = 1.0 / N * self._sum_x
            self._stds = np.sqrt(1.0/(N - 1) * (self._sum_sq_x - 2.0 * self._sum_x * self._means + N * self._means**2))
            self._stds[self._stds < eps] = eps
            pickle.dump(obj={'means': self._means,
                             'stds': self._stds},
                        file=save_file,
                        protocol=2)

    def load_params(self, load_file_path):
        with open(load_file_path, "rb") as load_file:
            if platform.python_version()[0] == '2':
                dct = pickle.load(load_file)
            else:
                dct = pickle.load(load_file, encoding='latin1')
            self._means = dct['means']
            self._stds = dct['stds']

    def transform(self, X):
        if self._fields is None:
            fields = range(X.shape[1])
        else:
            fields = self._fields
        ret = 1.0 * X
        for col in fields:
            ret[:, col] = (X[:, col] - self._means[col]) / self._stds[col]
        return ret

#=======================================================================
#from keras import ModelCheckpoint, CSVLogger

from keras._tf_keras.keras.callbacks import CSVLogger , ModelCheckpoint

import tensorflow as tf
import importlib
#from keras.callbacks import ModelCheckpoint, CSVLogger




parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/decompensation/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.set_defaults(deep_supervision=False)
args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30

# Build readers, discretizers, normalizers
if args.deep_supervision:
    train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                               listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                               small_part=args.small_part)
    val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                             listfile=os.path.join(args.data, 'val_listfile.csv'),
                                                             small_part=args.small_part)
else:
    train_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'train_listfile.csv'))
    val_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

if args.deep_supervision:
    discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
else:
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'decomp_ts{}.input_str-previous.n1e5.start_time-zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'decomp'


# Build the model
print("==> using model {}".format(args.network))
model_module = imp.machinery.SourceFileLoader(os.path.basename(args.network), args.network)
module = model_module.load_module()
model = module.Network(**args_dict)
suffix = "{}.bs{}{}{}.ts{}".format("" if not args.deep_supervision else ".dsup",
                                   args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)


# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'learning_rate': args.lr,
                               'beta_1': args.beta_1}}

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
model.compile(optimizer=optimizer_config,
              loss='binary_crossentropy')
model.summary()

# Load model weights
n_trained_chunks = 0
if args.load_state != "":
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*chunk([0-9]+).*", args.load_state).group(1))
# Load data and prepare generators

def train_generator():
    for batch in train_data_gen:
        X, y = batch
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        yield X, y

def val_generator():
    for batch in val_data_gen:
        X, y = batch
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        yield X, y
        
# Load data and prepare generators
if args.deep_supervision:
    train_data_gen = utils.BatchGenDeepSupervision(
        train_data_loader, discretizer, normalizer, args.batch_size, shuffle=True)
    val_data_gen = utils.BatchGenDeepSupervision(
        val_data_loader, discretizer, normalizer, args.batch_size, shuffle=False)
else:
    # Set number of batches in one epoch
    train_nbatches = 2000
    val_nbatches = 1000
    if args.small_part:
        train_nbatches = 40
        val_nbatches = 40
    train_data_gen = utils.BatchGen(
        train_reader, discretizer, normalizer, args.batch_size, train_nbatches, True)
    val_data_gen = utils.BatchGen(
        val_reader, discretizer, normalizer, args.batch_size, val_nbatches, False)

def train_generator():
    for batch in train_data_gen:
        X, y = batch
        X = np.reshape(X, (X.shape[0], 1, 76))  # Adjust the shape as needed
        y = np.reshape(y, (y.shape[0], 1))      # Ensure y has the shape (batch_size, 1)
        yield X, y

def val_generator():
    for batch in val_data_gen:
        X, y = batch
        X = np.reshape(X, (X.shape[0], 1, 76))  # Adjust the shape as needed
        y = np.reshape(y, (y.shape[0], 1))      # Ensure y has the shape (batch_size, 1)
        yield X, y

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator(),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, 76), dtype=tf.float32),  # Adjusted shape
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)       # Adjusted shape for labels
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator(),
    output_signature=(
        tf.TensorSpec(shape=(None, 1, 76), dtype=tf.float32),  # Adjusted shape
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)       # Adjusted shape for labels
    )
)

if args.mode == 'train':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss:.4f}.keras')

    metrics_callback = keras_utils.DecompensationMetrics(
        train_data_gen=train_data_gen,
        val_data_gen=val_data_gen,
        deep_supervision=args.deep_supervision,
        batch_size=args.batch_size,
        verbose=args.verbose)
    
    # Make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    saver = ModelCheckpoint(
        path, verbose=1, save_freq='epoch', save_best_only=True, monitor='val_loss', mode='min')

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'), append=True, separator=';')

    print("==> training")
    model.fit(
        train_dataset,
        steps_per_epoch=train_data_gen.steps,
        validation_data=val_dataset,
        validation_steps=val_data_gen.steps,
        epochs=n_trained_chunks + args.epochs,
        initial_epoch=n_trained_chunks,
        callbacks=[metrics_callback, saver, csv_logger],
        verbose=args.verbose)


elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_data_gen
    del val_data_gen

    names = []
    ts = []
    labels = []
    predictions = []

    if args.deep_supervision:
        del train_data_loader
        del val_data_loader
        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'test'),
                                                                  listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                                  small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                      normalizer, args.batch_size,
                                                      shuffle=False, return_names=True)

        for i in range(test_data_gen.steps):
            print("\tdone {}/{}".format(i, test_data_gen.steps), end='\r')
            ret = next(test_data_gen)
            (x, y) = ret["data"]
            cur_names = np.array(ret["names"]).repeat(x[0].shape[1], axis=-1)
            cur_ts = ret["ts"]
            for single_ts in cur_ts:
                ts += single_ts

            pred = model.predict(x, batch_size=args.batch_size)
            for m, t, p, name in zip(x[1].flatten(), y.flatten(), pred.flatten(), cur_names.flatten()):
                if np.equal(m, 1):
                    labels.append(t)
                    predictions.append(p)
                    names.append(name)
        print('\n')
    else:
        del train_reader
        del val_reader
        test_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'test'),
                                           listfile=os.path.join(args.data, 'test_listfile.csv'))

        test_data_gen = utils.BatchGen(test_reader, discretizer,
                                       normalizer, args.batch_size,
                                       None, shuffle=False, return_names=True)  # put steps = None for a full test

        for i in range(test_data_gen.steps):
            print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
            ret = next(test_data_gen)
            x, y = ret["data"]
            cur_names = ret["names"]
            cur_ts = ret["ts"]

            x = np.array(x)
            pred = model.predict_on_batch(x)[:, 0]
            predictions += list(pred)
            labels += list(y)
            names += list(cur_names)
            ts += list(cur_ts)

    metrics.print_metrics_binary(labels, predictions)
    path = os.path.join(args.output_dir, 'test_predictions', os.path.basename(args.load_state)) + '.csv'
    utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")