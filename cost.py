import itertools
import math

import theano.tensor as T
from pylearn2.costs.gsn import GSNCost

class FunnelGSNCost(GSNCost):
    def get_monitoring_channels(self, model, data, **kwargs):
        chans = super(FunnelGSNCost, self) \
            .get_monitoring_channels(model, data, **kwargs)

        output = self._get_samples_from_model(model, data)

        # axes: 0: time step, 1: item in minibatch, 2: sample component
        samples = T.stack(*list(itertools.chain(*output)))

        # only want first component of each sample
        # axis 0: time step, axis 1: item in in minibatch
        samples = samples[:, :, 0]

        # sigma^2 = 9.0

        chans['x_std'] = T.std(data[0][:, 0])

        likelihood = T.exp(-T.sqr(samples) / 18.0) / T.sqrt(18.0 * math.pi)
        #chans['y_ll'] = T.sum(T.log(likelihood))
        chans['y_mean'] = T.mean(samples)
        chans['y_std'] = T.std(samples)

        return chans


