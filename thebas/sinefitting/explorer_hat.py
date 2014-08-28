# coding=utf-8
"""SANDBOX TO DELETE"""
from thebas.sinefitting.models import instantiate_model
import numpy as np
from thebas.sinefitting.samplers import sample

if __name__ == '__main__':
    # Control stochastics
    np.random.seed(0)

    def varsexp1():
        group_id, group_data, model_factory, model, sample_params = instantiate_model()
        print model
        var = model[0]
        print var.value
        print var.parents
        print var.__name__
        for var in model:
            if 'y_' in var.__name__:
                print '-' * 40
                print var.__name__, var.value
                for varname, p in var.parents.iteritems():
                    print varname, p.value, p.__size__

    sample(iters=80000, burn=20000)
