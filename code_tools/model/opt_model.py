import tensorflow as tf
'''
Sources:
https://www.pyimagesearch.com/2020/03/23/using-tensorflow-and-gradienttape-to-train-a-keras-model/

# Encapsulating the forward and backward passs of data using
# tf.GradientTape for updating model weights.

'''

@tf.function
def train_step(model, loss_estimator, acc_estimator, 
                opt, train_trc, train_label, training=True):
    # compute loss for gradient descent
    with tf.GradientTape() as tape:
        # make predictions and estimate loss
        train_pred = model(train_trc, training=training)
        train_loss = loss_estimator(train_label, train_pred)
        train_acc =  acc_estimator(train_label, train_pred)
    # calculate the gradients and update the weights
    grads = tape.gradient(train_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return train_loss, train_acc

@tf.function
def val_step(model, loss_estimator, acc_estimator,
             val_trc, val_label, training=False):
    # make predictions and estimate loss
    val_pred = model(val_trc, training=training)
    val_loss = loss_estimator(val_label, val_pred)
    val_acc = acc_estimator(val_label, val_pred)
    # calculate the gradients and update the weight
    return val_loss, val_acc


if __name__=='__main__':
    pass