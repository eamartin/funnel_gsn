import cPickle as pickle
import itertools

import numpy as np
from pylearn2.corruption import *
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError as MSR
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster, SGD
import theano.tensor as T

from funnel_gsn.cost import FunnelGSNCost
from funnel_gsn.distribution import FunnelDistribution, generate_points

def rlu(x):
    return T.maximum(0, x)

def train():
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.25

    MAX_EPOCHS = 500
    BATCHES_PER_EPOCH = 100
    BATCH_SIZE = 1000

    dataset = FunnelDistribution()
    cost = FunnelGSNCost([(0, 1.0, MSR())], walkback=1)

    gc = GaussianCorruptor(0.75)
    dc = DropoutCorruptor(.5)
    gsn = GSN.new([10, 200, 10],
                  [None, "tanh", "tanh"], # activation
                  [None] * 3, # pre corruption
                  [None] * 3, # post corruption
                  [None] * 3, # layer samplers
                  tied=False)
    gsn._bias_switch = False

    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=cost,
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE,
              monitoring_batches=100,
              monitoring_dataset=dataset)

    trainer = Train(dataset, gsn, algorithm=alg, save_path="funnel_gsn.pkl",
                    extensions=[MonitorBasedLRAdjuster()],
                    save_freq=50)

    trainer.main_loop()
    print "done training"

def reconstruct():
    """ test to see how well point is reconstructed"""
    gsn = pickle.load(open('funnel_gsn.pkl'))

    x = generate_points(1)
    r_x = gsn.get_samples([(0, x)], symbolic=False)[0][0]

    print 'Original ', x
    print 'Reconstructed ', r_x

def sample():
    gsn = pickle.load(open('funnel_gsn.pkl'))
    seed = np.random.rand(1, 10).astype('float32')

    samples = gsn.get_samples([(0, seed)], symbolic=False, walkback=10000)

    # dimensions: 0: time step, 1: idx into mb, 2: component of vector
    samples = np.array(list(itertools.chain(*samples)))

    samples = samples[:, 0, :]

    print 'Mean ', samples.mean()
    print 'Width std ', samples[:, 0].std()
    print 'Width max ', samples[:, 0].max()
    print 'Width min ', samples[:, 0].min()
    print 'Width med ', np.median(samples[:, 0])
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    train()
    reconstruct()
    sample()
