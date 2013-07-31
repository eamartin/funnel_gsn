from pylearn2.corruption import GaussianCorruptor
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError as MSR
from pylearn2.costs.gsn import GSNCost
from pylearn2.models.gsn import GSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD

from funnel_gsn.distribution import FunnelDistribution

HIDDEN_SIZE = 60
GAUSSIAN_NOISE = 3.0

LEARNING_RATE = 0.001
MOMENTUM = 0.5

MAX_EPOCHS = 100
BATCHES_PER_EPOCH = 100
BATCH_SIZE = 100

dataset = FunnelDistribution()

layers = [10, HIDDEN_SIZE]
corruptor = GaussianCorruptor(GAUSSIAN_NOISE)

cost = GSNCost([(0, 1.0, MSR())])

gsn = GSN.new_ae(layers, vis_corruptor=corruptor,
                 hidden_pre_corruptor=corruptor, hidden_post_corruptor=corruptor,
                 visible_act=None)

alg = SGD(LEARNING_RATE, init_momentum=MOMENTUM, cost=cost,
          termination_criterion=EpochCounter(MAX_EPOCHS),
          batches_per_iter=BATCHES_PER_EPOCH, batch_size=BATCH_SIZE)

trainer = Train(dataset, gsn, algorithm=alg, save_path="funnel_gsn.pkl",
                save_freq=10)
trainer.main_loop()
print "done training"
