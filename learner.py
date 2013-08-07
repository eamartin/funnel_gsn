import numpy as np

from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError as MSR
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster, SGD

from funnel_gsn.cost import FunnelGSNCost
from funnel_gsn.distribution import FunnelDistribution

def train():
    GAUSSIAN_NOISE = 2.0

    LEARNING_RATE = 0.0001
    MOMENTUM = 0.0

    MAX_EPOCHS = 500
    BATCHES_PER_EPOCH = 100
    BATCH_SIZE = 1000

    dataset = FunnelDistribution()

    layers = [10, 12]
    corruptor = GaussianCorruptor(GAUSSIAN_NOISE)

    cost = FunnelGSNCost([(0, 1.0, MSR())])

    gsn = GSN.new_ae(layers, vis_corruptor=corruptor,
                     hidden_pre_corruptor=None, hidden_post_corruptor=None,
                     visible_act=None, hidden_act="tanh",
                     visible_sampler=lambda: None, tied=False)

    alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=cost,
              termination_criterion=EpochCounter(MAX_EPOCHS),
              batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE,
              monitoring_batches=10, monitoring_dataset=dataset)

    trainer = Train(dataset, gsn, algorithm=alg, save_path="funnel_gsn.pkl",
                    extensions=[MonitorBasedLRAdjuster()], save_freq=10)

    trainer.main_loop()
    print "done training"

def sample():
    import cPickle as pickle
    from funnel_gsn.distribution import generate_points

    gsn = pickle.load(open('funnel_gsn.pkl'))
    seed = np.random.rand(1, 10)

    samples = gsn.get_samples([(0, seed)], symbolic=False, walkback=100)

    from IPython import embed; embed()

if __name__ == '__main__':
    train()
    sample()
