# coding=utf-8
import os.path as op
from itertools import product
import inspect
from thebas.sinefitting import DEFAULT_MCMC_RESULTS_DIR
from thebas.sinefitting.models import MODEL_FACTORIES
from thebas.sinefitting.perturbation_experiment_data import perturbation_data_to_records
from thebas.sinefitting.results import MCMCRunManager


def sample(root_dir=None,
           freq=2.,
           genotype_id='VT37804_TNTE',
           model_id='gpa_t1',
           num_chains=4,  # used via "eval"
           iters=80000,   # used via "eval"
           burn=60000,    # used via "eval"
           progress_bar=False):

    print('Using model %s for genotype %s, frequency %g' % (model_id, genotype_id, freq))

    # Pick the specified model
    if not model_id in MODEL_FACTORIES:
        raise Exception('Do not know a model with id %s' % model_id)
    model_factory = MODEL_FACTORIES[model_id]

    # Read and select the data
    group_id = 'freq=%g__genotype=%s' % (freq, genotype_id)
    group_data = perturbation_data_to_records()
    group_data = group_data[(group_data['freq'] == freq) &  # select the group data
                            (group_data['genotype'] == genotype_id)]

    # Instantiate the model
    model, sample_params = model_factory(group_id, group_data)

    # Run-id and dest dir
    name = 'model=%s__%s' % (model_id, group_id)
    root_dir = op.join(DEFAULT_MCMC_RESULTS_DIR, name) if root_dir is None else op.join(root_dir, name)

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


def cl(MODELS=('gpa3', 'gpa_t1', 'gpa_t1_slice', 'gpa_t2_slice', 'gpad_t1_slice'),
       GROUPS=('VT37804_TNTE', 'VT37804_TNTin'),   # DCD-silenced, control)
       FREQS=(.5, 1., 2., 4., 8., 16., 32., 40.),  # angular frequencies (rad/s)
       log_dir='~'):
    """Generates command lines to launch the script with different parameters."""
    for model, af, genotype in product(MODELS, FREQS, GROUPS):
        expid = '%s__%s__%g' % (model, genotype, af)
        print('python2 -u thebas/sinefitting/samplers.py sample '
              '--freq %g '
              '--genotype-id %s '
              '--model-id %s '
              '&>%s/%s.log' %
              (af, genotype, model, log_dir, expid))


if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([cl, sample])
    parser.dispatch()