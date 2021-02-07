import os
import sys
import logging
import numpy as np
import tensorflow as tf
from .example_parser import _parse_function
from .example_parser import _parse_function_detect
from .example_parser import _parse_function_old
from .example_parser import _parse_function_bp
AUTOTUNE = tf.data.experimental.AUTOTUNE
logging.basicConfig(level=logging.INFO,
        format='%(levelname)s : %(asctime)s : %(message)s')

def _yield_batch(parsed_dataset, batch_size,
         data_length, trc_channel, label_channel):
    parsed_iterator = parsed_dataset.as_numpy_iterator()
    for ds in parsed_iterator:
        trc = tf.reshape(ds['trc_data'], 
            (batch_size, data_length, trc_channel))
        label = tf.reshape(ds['label_data'], 
            (batch_size, data_length, label_channel))
        idx = ds['idx']
        yield trc, label, idx


def tfrecord_dataset(file_list, repeat=-1, batch_size=None,
            trc_chn=3, label_chn=3, 
            data_length=2001, shuffle_buffer_size=300):
    if batch_size == None:
        raise ValueError("Must specify value of `batch_size`")
    else:
        dataset = tf.data.TFRecordDataset(file_list,
             num_parallel_reads=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size,
             reshuffle_each_iteration=True)
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        parsed_dataset = dataset.map(
                lambda x:_parse_function(x, 
            data_length=data_length, trc_chn=trc_chn,
            label_chn=label_chn))
        generator = _yield_batch(parsed_dataset, batch_size, 
            data_length, trc_chn, label_chn)
        return generator

def _yield_batch_old(parsed_dataset, batch_size, data_length,
         trc_channel, label_channel):
    parsed_iterator = parsed_dataset.as_numpy_iterator()
    for ds in parsed_iterator:
        trc = tf.reshape(ds['trc_data'],
            (batch_size, data_length, trc_channel))
        label = tf.reshape(ds['label_data'],
             (batch_size, data_length, label_channel))
        yield trc, label


def tfrecord_dataset_old(file_list, repeat=-1, batch_size=None,
                         trc_chn=3, label_chn=3, data_length=2001,
                          shuffle_buffer_size=300):
    if batch_size == None:
        raise ValueError("Must specify value of `batch_size`")
    else:
        dataset = tf.data.TFRecordDataset(file_list, 
            num_parallel_reads=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, 
                    reshuffle_each_iteration=True)
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        parsed_dataset = dataset.map(
                lambda x:_parse_function_old(x,
                     data_length=data_length, 
                     trc_chn=trc_chn, label_chn=label_chn))
        generator = _yield_batch_old(parsed_dataset, 
                batch_size, data_length, trc_chn, label_chn)
        return generator


def TFRdataset_bp(file_list,
            batch_size=10, data_length=2001,
            repeat=None, trc_dim=6, label_dim=1, trc_chn=3, label_chn=3):
    dataset = tf.data.TFRecordDataset(file_list,
                                num_parallel_reads=AUTOTUNE)
    if repeat:
        dataset = dataset.repeat(repeat)
    dataset = dataset.shuffle(buffer_size=100, 
                                reshuffle_each_iteration=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    parsed_dataset = dataset.map(
            lambda x:_parse_function_bp(x, 
            data_length=data_length, trc_dim=trc_dim, label_dim=label_dim,
            trc_chn=trc_chn, label_chn=label_chn, batch_size=batch_size)
        )
    return parsed_dataset

def _yield_batch_detect(parsed_dataset, batch_size,
         data_length, trc_channel, label_channel, mask_chn):
    parsed_iterator = parsed_dataset.as_numpy_iterator()
    for ds in parsed_iterator:
        trc = tf.reshape(ds['trc_data'],
             (batch_size, data_length, trc_channel))
        label = tf.reshape(ds['label_data'],
             (batch_size, data_length, label_channel))
        idx = ds['idx']
        yield trc, label, idx

def tfrecord_dataset_detect(file_list, repeat=-1, batch_size=None, 
                        trc_chn=3, label_chn=3, mask_chn=2,
                        data_length=2001, shuffle_buffer_size=300):
    if batch_size == None:
        raise ValueError("Must specify value of `batch_size`")
    else:
        dataset = tf.data.TFRecordDataset(file_list, 
                    num_parallel_reads=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, 
                    reshuffle_each_iteration=True)
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        parsed_dataset = dataset.map(
                lambda x:_parse_function_detect(x,
                 data_length=data_length, trc_chn=trc_chn, 
                 label_chn=label_chn, mask_chn=2, batch_size=batch_size))
        return parsed_dataset