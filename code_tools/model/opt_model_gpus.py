import tensorflow as tf
'''
Sources:
https://www.tensorflow.org/guide/distributed_training
https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/

# Encapsulating the forward and backward passs of data using
# tf.GradientTape for updating model weights.

model = build_unet_v1()
opt = Lookahead(RectifiedAdam(learning_rate=1e-3), sync_period=5, slow_step_size=0.5)
loss_estimator = tf.keras.losses.BinaryCrossentropy()
acc_estimator = tf.keras.metrics.BinaryAccuracy()
'''

@tf.function
def distributed_train_step(strategy, train_args):
    def train_step_gpus(model, train_acc, opt, global_batch_size, 
            train_trc, train_label, loss_estimator, training=True):
        # compute loss for gradient descent
        with tf.GradientTape() as tape:
            # make predictions and estimate loss
            train_pred = model(train_trc, training=True)
            per_replica_losses = loss_estimator(train_label, train_pred)
            grad_loss = tf.reduce_sum(per_replica_losses)/global_batch_size

        # calculate the gradients and update the weights
        grad = tape.gradient(grad_loss, model.trainable_variables)
        opt.apply_gradients(zip(grad, model.trainable_variables))
        train_acc.update_state(train_label, train_pred)
        return per_replica_losses

    per_replica_losses = strategy.run(
                            train_step_gpus, args=train_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
                    per_replica_losses, axis=None)                               

    #train_loss_avg.update_state(mean_loss)
    return mean_batch_loss, per_replica_losses

@tf.function
def distributed_val_step(strategy, val_args):
    def val_step_gpus(model, global_batch_size,
            val_acc, val_trc, val_label, loss_estimator):
        # estimate validation data 
        val_pred = model(val_trc, training=False)
        per_replica_losses = loss_estimator(val_label, val_pred)
        val_acc.update_state(val_label, val_pred)
        return per_replica_losses
    per_replica_losses = strategy.run(val_step_gpus, args=val_args)
    mean_batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, 
                    per_replica_losses, axis=None)
    return mean_batch_loss

if __name__=='__main__':
    pass
