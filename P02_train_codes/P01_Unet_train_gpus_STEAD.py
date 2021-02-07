import os
import sys
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorflow_addons.optimizers import RectifiedAdam
from code_tools.model.build_model import unets
from code_tools.model.save_model import save_model_v2
from code_tools.model.opt_model_gpus import distributed_train_step
from code_tools.model.opt_model_gpus import  distributed_val_step
from code_tools.data_io import tfrecord_dataset_detect

AUTOTUNE = tf.data.experimental.AUTOTUNE
logging.basicConfig(level=logging.INFO,
                format='%(levelname)s : %(asctime)s : %(message)s')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
##############################3 basic information
basepath = './'
datapath = os.path.join(basepath, '../input_TFRecord_EQT_20s')
outdir = os.path.join(basepath, 's02_trained_model',
            f'ARRU_detect/EQT_detect_20s')
if not os.path.exists(outdir): os.makedirs(outdir)
train_wf = glob(os.path.join(datapath, 'train/*.tfrecord'))#[:50000]
val_wf = glob(os.path.join(datapath, 'val/*.tfrecord'))#[:5000]
train_list = np.random.permutation(train_wf)
val_list = np.random.permutation(val_wf)

# model training parameters
frame = unets(input_size=(2001, 3))
train_epoch = 300
data_length = 2001
not_improve_patience = 20
resume_training = False
save_model_per_epoch = False

# multi-devices strategy
batch_size_per_replica = 20
strategy = tf.distribute.MirroredStrategy()
n_device = strategy.num_replicas_in_sync
global_batch_size = batch_size_per_replica * n_device

# training/validation steps per epoch
resume_training = False
save_model_per_epoch = False
n_train_data = len(train_list)
n_val_data = len(val_list)
train_steps_per_epoch = int(np.floor(n_train_data/global_batch_size))
val_steps_per_epoch = int(np.floor(n_val_data/global_batch_size))

############################### model training
## Initiliaze model 
#train_loss_avg = tf.keras.metrics.Mean(name='train_loss')
#val_loss_avg = tf.keras.metrics.Mean(name='val_loss')

with strategy.scope():
    opt = RectifiedAdam(lr=1e-4, min_lr=1e-6, 
            warmup_proportion=0.1, total_steps=1785900)
    loss_estimator = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    val_acc = tf.keras.metrics.CategoricalAccuracy(name='val_acc')

    if resume_training:
        model = frame.build_attR2unet(
                    pretrained_weights=os.path.join(
                        outdir, resume_training, 'train.hdf5'))

        train_loss_epoch = list(np.load(os.path.join(
            outdir, resume_training, 'train_loss.npy')))
        val_loss_epoch = list(np.load(os.path.join(
            outdir, resume_training, 'val_loss.npy')))
        train_acc_epoch = list(np.load(os.path.join
        (outdir, resume_training, 'train_acc.npy')))
        val_acc_epoch = list(np.load(os.path.join(
            outdir, resume_training, 'val_acc.npy')))
        track_loss = val_loss_epoch[-1]
        st_epoch_idx = len(val_acc_epoch)
    else:
        #model = frame.build_unet() 
        #model = frame.build_R2unet()
        model = frame.build_attR2unet()

        train_loss_epoch = []; val_loss_epoch = []
        train_acc_epoch = []; val_acc_epoch = []
        track_loss = np.inf
        st_epoch_idx = 1
    # Loss reduction and scaling is done automatically 
    # in keras model.compile and model.fit
    model.compile(optimizer = opt, loss=loss_estimator,
        metrics=['accuracy', 'loss'])

    # Training loop
    for i in range(1, train_epoch+1):
        logging.info(f"Training epoch : {i}/{train_epoch}")
        if save_model_per_epoch:
            outdir_epoch = os.path.join(outdir, f'epoch_{i:04}')
            if not os.path.exists(outdir_epoch):
                os.makedirs(outdir_epoch)
            save_dir = outdir_epoch
        else:
            save_dir = outdir
        # make and shuffle input data list
        train_list = np.random.permutation(train_list)
        val_list = np.random.permutation(val_list)

        # make data generator
        dis_train = iter(strategy.experimental_distribute_dataset(
            tfrecord_dataset_detect(train_list,
                 batch_size=global_batch_size, data_length=2001)))
        dis_val = iter(strategy.experimental_distribute_dataset(
            tfrecord_dataset_detect(val_list,
                 batch_size=global_batch_size, data_length=2001)))

        total_train_loss = 0
        train_num_batches = 0
        for train_step in range(train_steps_per_epoch):
            # update model weights per step
            dis_train_trc_in, dis_train_label_in, _, _ = next(dis_train)
            mean_train_batch_loss, per_replica_losses = distributed_train_step(
                strategy,
                (model, train_acc, opt, global_batch_size,
                dis_train_trc_in, dis_train_label_in,
                loss_estimator),
            )
            #stop
            #train_loss = tf.reduce_sum(mean_train_batch_loss)/\
            #    (global_batch_size*strategy.num_replicas_in_sync)
            train_loss = tf.reduce_mean(mean_train_batch_loss)/\
                strategy.num_replicas_in_sync

            total_train_loss += train_loss.numpy()
            train_num_batches += 1
            # progress bar
            progbar = tf.keras.utils.Progbar(
                train_steps_per_epoch, interval=0.1,
                stateful_metrics=['step'])
            progbar.update(train_step+1)
            #progbar.update(train_step+1, 
            #    values=(train_step+1
            #        ('steps',  train_step)
            #    ))
        train_loss_f = total_train_loss / train_num_batches

        total_val_loss = 0
        val_num_batches = 0
        for val_step in range(val_steps_per_epoch):
            # estimate validation dataset
            dis_val_trc_in, dis_val_label_in, _, _ = next(dis_val)
            mean_val_batch_loss = distributed_val_step(
                strategy,
                (model, global_batch_size, val_acc,
                dis_val_trc_in, dis_val_label_in,
                loss_estimator)
            )
            #val_loss = tf.reduce_sum(mean_val_batch_loss)/\
            #    (global_batch_size*strategy.num_replicas_in_sync)
            val_loss = tf.reduce_mean(mean_val_batch_loss)/\
                strategy.num_replicas_in_sync
            total_val_loss += val_loss.numpy()
            val_num_batches += 1
        val_loss_f = total_val_loss / val_num_batches
        #stop

        progbar.update(train_step+1,
            values=(
                ('train_loss',  train_loss_f),
                ('val_loss', val_loss_f),
                ('train_acc',  train_acc.result().numpy()),
                ('val_acc', val_acc.result().numpy())
            ))

        train_loss_epoch.append(train_loss_f)
        train_acc_epoch.append(train_acc.result().numpy())
        val_loss_epoch.append(val_loss_f)
        val_acc_epoch.append(val_acc.result().numpy())

        # trace model improvements
        track_item = val_loss_f
        if  track_item < track_loss:
            message = save_model_v2(
                model, save_dir, train_loss_epoch,
                val_loss_epoch, train_acc_epoch, val_acc_epoch)

            logging.info("val_loss improved from "
                f"{track_loss:.6f} to {track_item:.6f}, {message}\n")
            track_loss = track_item
            ct_not_improve = 0
        else:
            ct_not_improve += 1
            logging.info(f"val_loss {track_item:.6f} "
                f"did not improve from {track_loss:.6f}")
            logging.info(f'Saved to {save_dir}\n')
            message = save_model_v2(
                model, save_dir,
                train_loss_epoch, val_loss_epoch,
                train_acc_epoch, val_acc_epoch)

        if ct_not_improve == not_improve_patience:
            logging.info("Performance has not improved for "
                f"{ct_not_improve} epochs, stop training.")
            logging.info(f'Do you want to keep training? yes or no')
            decision = input()
            if decision == 'yes':
                ct_not_improve = 0
                pass
            elif decision == 'no':
                break
        #train_loss_avg.reset_states()
        train_acc.reset_states()
        #val_loss_avg.reset_states()
        val_acc.reset_states()

