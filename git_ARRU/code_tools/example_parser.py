import os
import numpy as np
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

## order 1. write tfrecord from obspy traces
def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

## order 2. read tfrecord files

def write_TFRecord(trc_3C, label_psn, idx, outfile):
    '''
    1. Create feature dictionary to be ready for setting up tf.train.Example object
        tf.train.Example can only accept 1-d data
    2. Create example protocol using tf.train.Example
    3. Write TFRecord object
    '''
    feature = {
        'trc_data': _float_feature(value=trc_3C.flatten() ),
        'label_data': _float_feature(value=label_psn.flatten()),
        'idx':_bytes_feature(value=idx.encode('utf-8'))
        }

    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature)) 

    out = tf.io.TFRecordWriter(outfile)
    out.write(example_proto.SerializeToString())

def _parse_function(record, data_length=2001, trc_chn=3, label_chn=3):
    flatten_size_trc = data_length*trc_chn
    flatten_size_label = data_length*label_chn
    feature = {
        "trc_data": tf.io.FixedLenFeature([flatten_size_trc], tf.float32),
        "label_data": tf.io.FixedLenFeature([flatten_size_label], tf.float32),
        "idx": tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_example(record, feature)


def _parse_function_old(record, data_length=2001, trc_chn=3, label_chn=3):
    flatten_size_trc = data_length*trc_chn
    flatten_size_label = data_length*label_chn
    feature = {
        "trc_data": tf.io.FixedLenFeature([flatten_size_trc], tf.float32),
        "label_data": tf.io.FixedLenFeature([flatten_size_label], tf.float32)
    }
    return tf.io.parse_example(record, feature)

def write_TFRecord_bp(trc_3C, label_psn, idx, envelope_len, outfile):
    '''
    1. Create feature dictionary to be ready for setting up tf.train.Example object
        tf.train.Example can only accept 1-d data
    2. Create example protocol using tf.train.Example
    3. Write TFRecord object
    '''
    feature = {
        'trc_data': _float_feature(value=trc_3C.flatten() ),
        'label_data': _float_feature(value=label_psn.flatten()),
        'idx':_bytes_feature(value=idx.encode('utf-8')),
        'envelope_len': _float_feature(value=envelope_len )
        }

    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature)) 

    out = tf.io.TFRecordWriter(outfile)
    out.write(example_proto.SerializeToString())

def _parse_function_bp(record, data_length=2001,
                        trc_dim=6, label_dim=1,
                        trc_chn=3, label_chn=3, batch_size=10):
    flatten_size_trc = data_length*trc_chn*trc_dim
    flatten_size_label = data_length*label_chn*label_dim
    feature = {
        "trc_data": tf.io.FixedLenFeature([flatten_size_trc], 
                    tf.float32),
        "label_data": tf.io.FixedLenFeature([flatten_size_label],
                    tf.float32),
        "idx": tf.io.FixedLenFeature([], tf.string),
        "envelope_len": tf.io.VarLenFeature(tf.float32)                    
    }
    record = tf.io.parse_example(record, feature)
    record['trc_data'] = tf.reshape(record['trc_data'], 
                    (batch_size, trc_dim, data_length, trc_chn))
    record['label_data'] = tf.reshape(record['label_data'],
                    (batch_size, label_dim, data_length, label_chn))
    
    return record['trc_data'], record['label_data'], \
            record['idx'], record['envelope_len']
    #return tf.io.parse_example(record, feature)

def write_TFRecord_detect(trc_3C, label_psn, mask, idx, outfile):
    '''
    1. Create feature dictionary to be ready for setting up
        tf.train.Example object
        tf.train.Example can only accept 1-d data
    2. Create example protocol using tf.train.Example
    3. Write TFRecord object
    '''
    feature = {
        'trc_data': _float_feature(value=trc_3C.flatten() ),
        'label_data': _float_feature(value=label_psn.flatten()),
        'mask': _float_feature(value=mask.flatten()),
        'idx':_bytes_feature(value=idx.encode('utf-8'))
        }

    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature)) 

    out = tf.io.TFRecordWriter(outfile)
    out.write(example_proto.SerializeToString())

def _parse_function_detect(record, data_length=2001, 
        trc_chn=3, label_chn=3, mask_chn=2, batch_size=10):
    flatten_size_trc = data_length*trc_chn
    flatten_size_label = data_length*label_chn
    flatten_size_mask = data_length*mask_chn
    feature = {
        "trc_data": tf.io.FixedLenFeature([flatten_size_trc], tf.float32),
        "label_data": tf.io.FixedLenFeature([flatten_size_label], tf.float32),
        "mask": tf.io.FixedLenFeature([flatten_size_mask], tf.float32),
        "idx": tf.io.FixedLenFeature([], tf.string)
    }

    record = tf.io.parse_example(record, feature)
    record['trc_data'] = tf.reshape(record['trc_data'], 
                    (batch_size, data_length, trc_chn))
    record['label_data'] = tf.reshape(record['label_data'],
                    (batch_size, data_length, label_chn))
    record['mask'] = tf.reshape(record['mask'],
                    (batch_size, data_length, mask_chn))

    return record['trc_data'], record['label_data'],\
            record['mask'], record['idx']
    #return tf.io.parse_example(record, feature)