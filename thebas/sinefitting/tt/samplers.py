# coding=utf-8
import os.path as op
from itertools import product
import inspect
from thebas.sinefitting import DEFAULT_MCMC_RESULTS_DIR
from thebas.sinefitting.models import MODEL_FACTORIES
from thebas.sinefitting.perturbation_experiment import perturbation_data_to_records
from thebas.sinefitting.results import MCMCRunManager


def sample(root_dir=None,
           mapstart=False,
           freq=2., genotype_id='VT37804_TNTE', model_id='gpa3',
           num_chains=4, iters=80000, burn=20000, progress_bar=False):
    print('Using model %s for genotype %s, frequency %g' % (model_id, genotype_id, freq))

    # Pick the specified model
    if not model_id in MODEL_FACTORIES:
        raise Exception('Do not know a model with id %s' % model_id)
    model_factory = MODEL_FACTORIES[model_id]

    # Read and select the data
    group_id = 'freq=%g__genotype=%s' % (freq, genotype_id)
    group_data = perturbation_data_to_records()  # low-pass smoothed, Lisa's filter for silence-removal
    group_data = group_data[(group_data.freq == freq) & (group_data.group == genotype_id)]  # select the group data

    # Instantiate the model
    model_dict = model_factory(group_id, group_data)

    # Run-id and dest dir
    name = 'model=%s__%s' % (model_id, group_id)
    if root_dir is None:
        root_dir = op.join(DEFAULT_MCMC_RESULTS_DIR, name)
    else:
        root_dir = op.join(root_dir, name)

    # Instantiate the result manager
    result = MCMCRunManager(root_dir=root_dir)
    result.save_model_txt(''.join(inspect.getsourcelines(model_factory)[0]))
    result.save_data(group_data)

    # Run MCMC
    result.run_sampler(model_dict, iters=iters, burn=burn,
                       num_chains=num_chains, progress_bar=progress_bar, mapstart=mapstart)

# sample(freq=4., iters=20000, burn=5000)
# exit(69)


def cl(MODELS=('gpa3nomap',),
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

sample(freq=16.)

if __name__ == '__main__':
    import argh
    parser = argh.ArghParser()
    parser.add_commands([cl, sample])
    parser.dispatch()