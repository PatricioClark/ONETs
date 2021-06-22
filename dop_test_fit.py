# DeepONet test

# This test uses Keras' fit method to train and is better suited for pipelines
# using tfrecords and datasets, and parallelization

import time
import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf
from tensorflow import keras

from onet import DeepONet

import matplotlib
import matplotlib.font_manager
matplotlib.rc('font',**{'size':20,
             'family':'serif',
             'serif':["Computer Modern Roman"]})
matplotlib.rc('text',usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{bm}')

def get_data(ell, m, num):
    try:
        X1_train = np.load('X1_train.npy')
        X2_train = np.load('X2_train.npy')
        Y_train = np.load('Y_train.npy')
        X1_test  = np.load('X1_test.npy')
        X2_test  = np.load('X2_test.npy')
        Y_test  = np.load('Y_test.npy')
    except:
        t0 = time.time()
        X1_train, X2_train, Y_train = generate_data(ell, m, num_training)
        print('Time for generation:', time.time()-t0)
        X1_test,  X2_test,  Y_test  = generate_data(ell, m, num_testing)
        np.save('X1_train.npy', X1_train)
        np.save('X2_train.npy', X2_train)
        np.save('Y_train.npy', Y_train)
        np.save('X1_test.npy',  X1_test)
        np.save('X2_test.npy',  X2_test)
        np.save('Y_test.npy',  Y_test)
    W_train = np.ones(np.shape(Y_train))
    W_test = np.ones(np.shape(Y_train))
    return (X1_train, X2_train, Y_train, W_train,
            X1_test,  X2_test,  Y_test,  W_test)

def generate_data(ell, m, num):
    # Specify Gaussian Process
    kernel = RBF(length_scale=ell)
    gp = GaussianProcessRegressor(kernel=kernel)

    # Create sensors
    sensors = np.linspace(0, 1, num=m)[:, None]

    # Create u's
    u_samples = gp.sample_y(sensors, num, None).T

    # Create y's
    y_samples = np.random.rand(num)[:, None]

    # Get G(u)(y)
    u_funcs = [interpolate.interp1d(sensors[:,0],
                                    sample,
                                    kind='cubic',
                                    copy=False,
                                    assume_sorted=True)
                                    for sample in u_samples]

    solutions = [solve_ivp(lambda y,s: f(y), [0, yf[0]], [0], method="RK45").y[0,-1:]
                 for f, yf in zip(u_funcs, y_samples)]
    solutions = np.array(solutions)

    return u_samples, y_samples, solutions

# Parameters
ell = 0.2
m   = 100
num_training = 10000
num_testing  = 10000
epochs       = 50000
bsize        = 10000
train        = True

(Xf_train,
 Xp_train,
 Y_train,
 W_train,
 Xf_test,
 Xp_test,
 Y_test,
 W_test)= get_data(ell, m, num_testing)

# Initialize and compile model
donet = DeepONet(m=m, dim_y=1, depth_branch=2, depth_trunk=2, p=40)
donet.model.compile(optimizer=donet.optimizer, loss='mse')

# Train
if train:
    donet.model.fit((Xf_train, Xp_train), Y_train,
            verbose=0,
            epochs=epochs,
            batch_size=bsize,
            initial_epoch=donet.ckpt.step.numpy(),
            callbacks    = [donet.ckpt_cb, donet.logger],
            validation_data=((Xf_test, Xp_test), Y_test))

# Test examples
lett = {10:'b', 15:'a', 20:'d'}
for ii in [10,15,20]:
    sensors = np.linspace(0, 1, num=m)[:, None]
    ys      = sensors

    # kernel = RBF(length_scale=ell)
    # gp = GaussianProcessRegressor(kernel=kernel)
    # u = gp.sample_y(sensors, 1, None).T
    u = Xf_test[ii].reshape(1,-1)
    u_func = interpolate.interp1d(sensors[:,0],
                                    u,
                                    copy=False,
                                    assume_sorted=True)
    f = lambda y,s: u_func(y)

    solutions = solve_ivp(f, [0, 1], [0], method="RK45", t_eval=ys[:,0]).y[0,:]
    solutions = np.array(solutions)

    dy = np.diff(ys[:,0])[10]
    deriv = np.gradient(solutions, dy)

    us = np.concatenate([u for _ in range(m)])
    pred = donet.model((us, ys))

    # Y_pred = donet.model((Xf_test, Xp_test))

    plt.figure(ii)
    plt.clf()
    plt.plot(sensors, u[0], ':',  label='$f$')
    # plt.plot(sensors, solutions, label='s')
    # plt.plot(sensors, pred, label='Onet')
    # plt.plot(sensors, deriv, label='deriv')
    # plt.plot(Xp_test[ii], Y_test[ii], 'ro')
    # plt.plot(Xp_test[ii], Y_pred[ii], 'go')
    plt.plot(sensors, solutions, label='$G^\dagger(f)(\zeta)$')
    plt.plot(sensors, pred, '--', label='$G(f)(\zeta)$')
    plt.ylabel('$f$, $G^\dagger(f), G(f)$')
    plt.xlabel('$\zeta$')
    plt.legend()
    ax=plt.gca()
    if ii==10:
        plt.text(0.05, 0.9, f'(b)', transform = ax.transAxes)
        # print(ii, np.sqrt(np.mean((pred[:,0]-solutions)**2)/np.mean(solutions**2)))
        # print(ii, np.mean(np.abs((pred[
    elif ii==20:
        plt.text(0.05, 0.9, f'(c)', transform = ax.transAxes)
        # print(ii, np.sqrt(np.mean((pred[:,0]-solutions)**2)/np.mean(solutions**2)))
    plt.tight_layout()
    plt.savefig(f'antiderivative_example_{ii:02}')
    # plt.close()

plt.figure(2)
plt.clf()
ax=plt.gca()
ep, loss, val = np.loadtxt('output.dat', skiprows=1, unpack=True)
plt.semilogy(ep, loss, label='Training')
plt.semilogy(ep, val, '--', label='Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.text(0.06, 0.9, '(a)', transform = ax.transAxes)
plt.tight_layout()
plt.savefig('loss_example')

plt.draw()
plt.show()
