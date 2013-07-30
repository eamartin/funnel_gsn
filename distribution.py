import numpy as np
from pylearn2.datasets.dataset import Dataset

def generate_points(num_points=1000):
    y = np.random.normal(scale=3.0, size=(num_points, 1))
    x = np.random.normal(scale=np.exp(y / 2.0), size=(num_points, 9))

    return np.hstack((y, x))

class FunnelDistribution(Dataset):
    def iterator(self, batch_size=100, num_batches=100, **kwargs):
        # FIXME: use rng
        assert all([batch_size > 0, num_batches > 0])

        for _ in xrange(num_batches):
            yield generate_points(batch_size)

    def has_targets(self):
        return False
