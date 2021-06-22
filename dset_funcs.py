# ###############################################
# Functions for dataset processing and generation
# ###############################################

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float64')

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Used to generate the records
def serialize_example(Xf, Xp, Y):
    # After creating each array save it by doing:
    # with tf.io.TFRecordWriter(f'data/{label}.tfrecord') as writer:
    #     for ii in range(num):
    #         serialized = serialize_example(combs_in[ii], p_in[ii], evol_out[ii])
    #         writer.write(serialized)
    feature = {
        'Xf': _float_feature(Xf.flatten()),
        'Xp': _float_feature(Xp.flatten()),
        'Y':  _float_feature(Y.flatten()),
    }
    example    = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()

    return serialized

# Used to generate the records
def serialize_example_with_weights(Xf, Xp, Y, W):
    feature = {
        'Xf': _float_feature(Xf.flatten()),
        'Xp': _float_feature(Xp.flatten()),
        'Y':  _float_feature(Y.flatten()),
        'W':  _float_feature(W.flatten()),
    }
    example    = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()

    return serialized

# Parse data
def proto_wrapper(branch_sensors, dim_y, dim_out):
    def parse_proto(example_proto):
        features = {
            'Xf': tf.io.FixedLenFeature([branch_sensors], tf.float32),
            'Xp': tf.io.FixedLenFeature([dim_y], tf.float32),
            'Y':  tf.io.FixedLenFeature([], tf.float32),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return (parsed_features['Xf'], parsed_features['Xp']), parsed_features['Y']
    return parse_proto

# Parse data
def proto_wrapper_with_weights(branch_sensors, dim_y, dim_out):
    def parse_proto(example_proto):
        features = {
            'Xf': tf.io.FixedLenFeature([branch_sensors], tf.float32),
            'Xp': tf.io.FixedLenFeature([dim_y], tf.float32),
            'Y':  tf.io.FixedLenFeature([dim_out], tf.float32),
            'W':  tf.io.FixedLenFeature([], tf.float32),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return ((parsed_features['Xf'], parsed_features['Xp']),
                 parsed_features['Y'],
                 parsed_features['W'])
    return parse_proto

# Load dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
def load_dataset(filepaths, branch_sensors, dim_y, dim_out, batch_size,
                 use_weights=True,
                 preads=1,
                 shuffle_buffer=0):
    # Read records
    dataset = tf.data.TFRecordDataset(filepaths,
                                      num_parallel_reads=preads)

    # Disable order
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = dataset.with_options(ignore_order)

    # Parse proto
    if use_weights:
        dataset = dataset.map(proto_wrapper_with_weights(branch_sensors, dim_y, dim_out),
                              num_parallel_calls=AUTOTUNE)
    else:
        dataset = dataset.map(proto_wrapper(branch_sensors, dim_y, dim_out),
                              num_parallel_calls=AUTOTUNE)

    # Shuffle, prefetch and batch
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset
