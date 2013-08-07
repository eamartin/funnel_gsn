import numpy as np
from pylearn2.datasets.dataset import Dataset

def generate_points(num_points=1000):
    y = np.random.normal(scale=3.0, size=(num_points, 1))
    x = np.random.normal(scale=np.exp(y / 2.0), size=(num_points, 9))

    return np.hstack((y, x)).astype('float32')


class FunnelIterator(object):
    stochastic = False

    def __init__(self, batch_size, num_batches, **kwargs):
        assert all([batch_size > 0, num_batches > 0])
        self.batch_size = batch_size
        self.remaining_batches = num_batches
        self.num_examples = batch_size * num_batches

    def __iter__(self):
        return self

    def next(self):
        if self.remaining_batches == 0:
            raise StopIteration
        else:
            self.remaining_batches -= 1
            return (generate_points(num_points=self.batch_size),)


class FunnelDistribution(Dataset):
    def iterator(self, batch_size=100, num_batches=100, **kwargs):
        return FunnelIterator(batch_size, num_batches)

    def has_targets(self):
        return False
