# coding=utf-8
import os.path as op
from itertools import product
import inspect

import numpy as np

from thebas.sinefitting import PB_PROJECTS
from thebas.sinefitting.models import instantiate_model
from thebas.sinefitting.results import MCMCRunManager


def sample(pbproject='dcn',
           freq=2.,
           genotype_id='VT37804_TNTE',
           model_id='gpa_t1',
           seed=0,
           num_chains=4,  # used via "eval"
           iters=80000,   # used via "eval"
           burn=60000,    # used via "eval"
           progress_bar=False):

    if isinstance(pbproject, basestring):
        if pbproject not in PB_PROJECTS.keys():
            raise Exception('The perturbation-bias project must be one of {%s}' %
                            ', '.join(sorted(PB_PROJECTS.keys())))
        pbproject = PB_PROJECTS[pbproject]

    print('Using model %s for genotype %s, frequency %g' % (model_id, genotype_id, freq))

    # Control rng (pymc uses numpy global)
    if seed is not None:
        np.random.seed(seed)

    # Instantiate the model
    group_id, group_data, model_factory, model, sample_params = \
        instantiate_model(pbproject=pbproject, freq=freq, genotype=genotype_id, model_id=model_id)

    # Run-id and dest dir
    name = 'model=%s__%s__seed=%r' % (model_id, group_id, seed)
    root_dir = op.join(pbproject.mcmc_dir, name)

    # Instantiate the result manager
    result = MCMCRunManager(root_dir=root_dir)
    result.save_model_txt(''.join(inspect.getsourcelines(model_factory)[0]))
    result.save_data(group_data)

    # Fill in sampling parameters not fixed by the model specification
    for sample_param in ('iters', 'burn', 'num_chains'):
        if sample_param not in sample_params:
            sample_params[sample_param] = eval(sample_param)

    # Run MCMC
    result.sample(model,
                  progress_bar=progress_bar,
                  **sample_params)


def cl(PROJECTS=sorted(PB_PROJECTS.keys()),
       MODELS=('gpa333',),
       FREQS=(4.0, 8.0, 0.5, 1.0, 2.0, 16.0, 32.0, 40.0),  # angular frequencies (rad/s)
       log_dir='~'):
    """Generates command lines to launch the script with different parameters."""
    for project in PROJECTS:
        pbproject = PB_PROJECTS[project]
        for model, af, genotype in product(MODELS, FREQS, pbproject.genotypes()):
            expid = '%s__%s__%s__%g' % (project, model, genotype, af)
            print('python2 -u thebas/sinefitting/samplers.py sample '
                  '--pbproject %s '
                  '--freq %g '
                  '--genotype-id %s '
                  '--model-id %s '
                  '&>%s/%s.log' %
                  (project, af, genotype, model, log_dir, expid))


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([cl, sample])
    parser.dispatch()
