#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Physics-Informed DeepONet class
# Written by Patricio Clark Di Leoni at Johns Hopkins University
# June 2021

import os
import copy
import numpy as np
import tensorflow as tf
from   tensorflow import keras
import time

tf.keras.backend.set_floatx('float32')

class DeepONet:
    """
    General Physics-Informed DeepONet class

    The class creates a keras.Model with a branch and trunk networks.

    Can work as a regular DeepONet too. The custom training loop calculates the
    residuals of the supplied PDEs in the same way as PINNs. Checkpoints and
    output are handled there and no keras callbacks are used.

    The code assumes the data is given in batched tf.data.Dataset formats. The
    lambda_data parameter from PINNs is now absorbed into the weights, and
    lambda_phys is fixed to 1. A global baseline lambda_data can be supplied
    outside of the weights. The Adam-like lambda balancing algorithm is
    implementing and balances the data part.

    Parameters
    ----------

    m : int
        Number of sensors (second dimension of branch network's input)
    dim_y : int
        Dimension of y (trunk network's input)
    depth_branch : int
        depth of branch network
    depth_trunk : int
        depth of branch network
    p : int
        Width of branch and trunk networks
    dim_out : int
        Dimension of output. Must be a true divisor of p. Default is 1. 
    aux_model : tf.keras.Model [optional]
        Option to add an auxiliary model as output. If None a dummy constant
        output is added to the DeepONet. Default is None.
    aux_coords : list [optional]
        If using an aux_model, specify which variables of the trunk input are used for it.
    dest : str [optional]
        Path for output files.
    activation : str [optional]
        Activation function to be used. Default is 'relu'.
    feature_expansion: func or None [optional]
        If not None, then the trunk inputs are feature expanded using the
        function provided. Default is None.
    optimizer : keras.optimizer instance [optional]
        Optimizer to be used in the gradient descent. Default is Adam with
        fixed learning rate equal to 1e-3.
    norm_in : float or array [optional]
        If a number or an array of size din is supplied, the first layer of the
        network normalizes the inputs uniformly between -1 and 1. Default is
        False.
    norm_out : float or array [optional]
        If a number or an array of size dim_out is supplied, the layer layer of the
        network normalizes the outputs using z-score. Default is
        False.
    norm_out_type : str [optional]
        Type of output normalization to use. Default is 'z-score'.
    save_freq : int [optional]
        Save model frequency. Default is 1.
    restore : bool [optional]
        If True, it checks if a checkpoint exists in dest. If a checkpoint
        exists it restores the modelfrom there. Default is True.
    """
    # Initialize the class
    def __init__(self,
                 m,
                 dim_y,
                 depth_branch,
                 depth_trunk,
                 p,
                 dim_out=1,
                 aux_model=None,
                 aux_coords=None,
                 dest='./',
                 regularizer=None,
                 p_drop=0.0,
                 activation='relu',
                 slope_recovery=False,
                 feature_expansion=None,
                 optimizer=keras.optimizers.Adam(lr=1e-3),
                 norm_in=False,
                 norm_out=False,
                 norm_out_type='z-score',
                 save_freq=1,
                 restore=True):

        # Numbers and dimensions
        self.m            = m
        self.dim_y        = dim_y
        self.dim_out      = dim_out
        self.depth_branch = depth_branch
        self.depth_trunk  = depth_trunk
        self.width        = p

        # Extras
        self.dest        = dest
        self.regu        = regularizer
        self.norm_in     = norm_in
        self.norm_out    = norm_out
        self.optimizer   = optimizer
        self.save_freq   = save_freq
        self.activation  = activation

        # Activation function
        if activation=='tanh':
            self.act_fn = keras.activations.tanh
            self.kinit  = 'glorot_normal'
        elif activation=='relu':
            self.act_fn = keras.activations.relu
            self.kinit  = 'he_normal'
        elif activation=='elu':
            self.act_fn = keras.activations.elu
            self.kinit  = 'glorot_normal'
        elif activation=='selu':
            self.act_fn = keras.activations.selu
            self.kinit  = 'lecun_normal'

        # Inputs definition
        funct = keras.layers.Input(m,     name='funct')
        point = keras.layers.Input(dim_y, name='point')

        # Normalize input
        if norm_in:
            fmin   = norm_in[0][0]
            fmax   = norm_in[0][1]
            pmin   = norm_in[1][0]
            pmax   = norm_in[1][1]
            norm_f   = lambda x: 2*(x-fmin)/(fmax-fmin) - 1
            norm_p   = lambda x: 2*(x-pmin)/(pmax-pmin) - 1
            hid_b = keras.layers.Lambda(norm_f)(funct)
            hid_t = keras.layers.Lambda(norm_p)(point)
        else:
            hid_b = funct
            hid_t = point

        # Auxiliary network
        if aux_model is not None:
            aux_coords = [hid_t[:,ii:ii+1] for ii in aux_coords]
            aux_coords = keras.layers.concatenate(aux_coords)
            aux_out    = aux_model(aux_coords, training=False)[0]
        else:
            cte     = keras.layers.Lambda(lambda x: 0*x[:,0:1]+1)(hid_t)
            aux_out = keras.layers.Dense(1, use_bias=False)(cte)

        # Expand time
        if feature_expansion is not None:
            hid_t = keras.layers.Lambda(feature_expansion)(hid_t)

        # Branch network
        for ii in range(self.depth_branch-1):
            hid_b = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       kernel_initializer=self.kinit,
                                       activation=self.act_fn)(hid_b)
            if p_drop:
                hid_b = keras.layers.Dropout(p_drop)(hid_b)
        hid_b = keras.layers.Dense(self.width,
                                   kernel_initializer=self.kinit,
                                   kernel_regularizer=self.regu)(hid_b)

        # Trunk network
        for ii in range(self.depth_trunk):
            hid_t = keras.layers.Dense(self.width,
                                       kernel_regularizer=self.regu,
                                       kernel_initializer=self.kinit,
                                       activation=self.act_fn)(hid_t)
            if p_drop and ii<self.depth_trunk-1:
                hid_t = keras.layers.Dropout(p_drop)(hid_t)

        # Output definition
        if dim_out>1:
            hid_b = keras.layers.Reshape((dim_out, p//dim_out))(hid_b)
            hid_t = keras.layers.Reshape((dim_out, p//dim_out))(hid_t)
        output = keras.layers.Multiply()([hid_b, hid_t])
        output = tf.reduce_sum(output, axis=2)
        output = BiasLayer()(output)

        # Normalize output
        if norm_out:
            if norm_out_type=='z_score':
                mm = norm_out[0]
                sg = norm_out[1]
                out_norm = lambda x: sg*x + mm 
            elif norm_out_type=='min_max':
                ymin = norm_out[0]
                ymax = norm_out[1]
                out_norm = lambda x: 0.5*(x+1)*(ymax-ymin) + ymin
            output = keras.layers.Lambda(out_norm)(output)

        # Create model
        model = keras.Model(inputs=[funct, point], outputs=[output, aux_out])
        self.model = model
        self.num_trainable_vars = np.sum([np.prod(v.shape)
                                          for v in self.model.trainable_variables])
        self.num_trainable_vars = tf.cast(self.num_trainable_vars, tf.float32)

        # Parameter for dynamic balance
        self.bal_data = tf.Variable(1.0, name='bal_data')

        # Create save checkpoints, managers and callbacks
        self.ckpt    = tf.train.Checkpoint(step=tf.Variable(0),
                                           model=self.model,
                                           bal_data=self.bal_data,
                                           optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  self.dest + '/ckpt',
                                                  max_to_keep=5)
        if restore:
            self.ckpt.restore(self.manager.latest_checkpoint)

    def train(self,
              train_dataset,
              pde,
              baseline_lambda=1.0,
              alpha=0.0,
              enforce_physics=True,
              epochs=10,
              verbose=False,
              print_freq=1,
              valid_freq=0,
              early_stopping=False,
              val_threshold=np.inf,
              data_mask=None,
              valid_dataset=None,
              save_freq=1):
        """
        Train function

        Loss functions are written to output.dat

        Parameters
        ----------

        train_dataset : tf.data.Dataset
            Training data. Shoud be of the form X, Y, W with X=(X_branch,
            X_trunk). W plays the role of loss function sample weights and
            lambda_data in the PI scheme.
        pde : function
            Function specifying the equations of the problem. Takes as a
            DeepONet class instance and coords as inputs.  The output of the
            function must be a list containing all equations.
        baseline_lambda : float [optional]
            Global baseline lambda_data parameter. Default is 1.0.
        alpha : float [optional]
            If non-zero, performs adaptive balance of the physics and data part
            of the loss functions. See comment above for reference. Cannot be
            set if "ntk_balance" is also set. Default is zero.
        enforce_physics : bool [optional]
            Turn physics on/off. Default is True.
        epochs : int [optional]
            Number of epochs to train. Default is 10.
        verbose : bool [optional]
            Verbose output or not. Default is False.
        print_freq : int [optional]
            Print status frequency. Default is 1.
        early_stopping : bool [optional]
            If True only saves the model Checkpoint when the validation is
            decreasing. Default is False.
        data_mask : list [optional]
            Determine which output fields to use in the data constraint. Must have
            shape (dout,) with either True or False in each element. Default is
            all True.
        valid_dataset : tf.data.Dataset
            Validation data. Same description as training_data, but does not
            use a data_mask.
        save_freq : int [optional]
            Save model frequency. Default is 1.
        """

        # Check data_mask
        if data_mask is None:
            data_mask = [True for _ in range(self.dim_out)]

        # Cast balance
        bal_data = tf.constant(self.bal_data.numpy(), dtype='float32')

        # Run epochs
        ep0 = int(self.ckpt.step)
        best_val = np.inf
        for ep in range(ep0, ep0+epochs):

            # Loop through batches
            for X, Y, W in train_dataset:
                (loss_data,
                 loss_phys,
                 bal_data) = self.training_step(X, Y, W,
                                                pde,
                                                baseline_lambda,
                                                data_mask,
                                                bal_data,
                                                alpha,
                                                enforce_physics)

            # Get validation
            if valid_dataset is not None:
                valid = self.validation(valid_dataset, baseline_lambda)
            else:
                valid = loss_data

            # Print status
            if ep%print_freq==0:
                status = [loss_data.numpy(), loss_phys.numpy()]

                if valid_dataset is not None:
                    status.append(valid.numpy())

                if alpha>0.0:
                    status.append(bal_data.numpy())

                self.print_status(ep, status, verbose=verbose)

            # Save progress
            self.ckpt.step.assign_add(1)
            self.ckpt.bal_data.assign(bal_data.numpy())
            if ep%save_freq==0 and valid.numpy()<best_val:
                self.manager.save()
                
                if early_stopping and valid.numpy()<val_threshold:
                    best_val = valid.numpy()
            

    @tf.function
    def training_step(self, X, Y, W, pde,
                      l0, data_mask, bal_data, alpha, enforce_physics):
        with tf.GradientTape(persistent=True) as tape:
            # Data part
            Y_p = self.model(X, training=True)[0]
            aux = [tf.reduce_mean(
                   W*l0*tf.square(Y[:,ii]-Y_p[:,ii]))
                   for ii in range(self.dim_out)
                   if data_mask[ii]]
            loss_data = tf.add_n(aux)/self.dim_out

            # Physics part
            if enforce_physics:
                equations = pde(self.model, X)
                loss_eqs  = [tf.reduce_mean(tf.square(eq))
                                 for eq in equations]
                loss_phys = tf.add_n(loss_eqs)
            else:
                loss_phys = 1.0

        # Calculate gradients of data part
        gradients_data = tape.gradient(loss_data,
                    self.model.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Calculate gradients of physics part
        if enforce_physics:
            gradients_phys = tape.gradient(loss_phys,
                        self.model.trainable_variables,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # alpha-based dynamic balance
        if alpha>0.0 and enforce_physics:
            mean_grad_data = get_mean_grad(gradients_data, self.num_trainable_vars)
            mean_grad_phys = get_mean_grad(gradients_phys, self.num_trainable_vars)
            lhat = mean_grad_phys/mean_grad_data
            bal_data = (1.0-alpha)*bal_data + alpha*lhat

        # Apply gradients
        if enforce_physics:
            gradients = [bal_data*g_data + g_phys
                         for g_data, g_phys in zip(gradients_data, gradients_phys)]
        else:
            gradients = gradients_data
        self.optimizer.apply_gradients(zip(gradients,
                    self.model.trainable_variables))

        return loss_data, loss_phys, bal_data

    @tf.function
    def validation(self, valid_dataset, l0):
        jj  = 0.0
        acc = 0.0
        for X, Y, W in valid_dataset:
            Y_p = self.model(X, training=True)[0]
            aux = [tf.reduce_mean(W*l0*tf.square(Y[:,ii]-Y_p[:,ii]))
                   for ii in range(self.dim_out)]
            acc += tf.add_n(aux)/self.dim_out
            jj  += 1.0

        return acc/jj

    def print_status(self, ep, status, verbose=False):
        """ Print status function """

        # Loss functions
        output_file = open(self.dest + 'output.dat', 'a')
        print(ep, *status, file=output_file)
        output_file.close()

        if verbose:
            print(ep, *status)

def get_mini_batch(X1, X2, Y, W, idx_arr, batch_size):
    idxs = np.random.choice(idx_arr, batch_size)
    return X1[idxs], X2[idxs], Y[idxs], W[idxs]

class BiasLayer(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

@tf.function
def MSE_loss(y_true, y_pred, weights):
    return tf.reduce_mean(tf.math.square(y_true - y_pred))

@tf.function
def get_mean_grad(grads, n):
    ''' Get the mean of the absolute values of the gradient '''
    sum_over_layers = [tf.reduce_sum(tf.abs(gr)) for gr in grads]
    total_sum       = tf.add_n(sum_over_layers)
    return total_sum/n
