# coding=utf-8
"""Code for sinewave-perturbation-response bayesian (and others) data analysis."""
import os
import os.path as op


def matplotlib_without_x(force=False):
    import os
    if force or os.getenv('DISPLAY') is None:
        import matplotlib
        matplotlib.use('agg')
        from matplotlib import pyplot as plt
        plt.ioff()
matplotlib_without_x()

# One dir to contain'em all
PERTURBATION_BIAS_ROOT = op.join(op.expanduser('~'), 'data-analysis', 'closed_loop_perturbations')
if not op.isdir(PERTURBATION_BIAS_ROOT):
    PERTURBATION_BIAS_ROOT = '/mnt/strawscience/santi/tethered-bayesian'


class PBProject(object):

    def __init__(self, root=PERTURBATION_BIAS_ROOT, name='HS'):
        super(PBProject, self).__init__()
        self.root = op.join(root, name)
        # Where the original (and munged) data will be...
        self.data_root = op.join(self.root, 'data')
        # Lazy loading of genotypes
        self._genotypes = None
        # Where we will store MCMC traces
        self.mcmc_dir = op.join(self.root, 'MCMC')
        # Where some of the generated plots will be
        self.plots_dir = op.join(self.root, 'plots')

    def genotypes(self):
        # We assume that we have a directory per genotype, an hdf5 per fly in each directory
        # No more directories should be present...
        if self._genotypes is None:
            self._genotypes = sorted([dirname for dirname in os.listdir(self.data_root)
                                      if op.isdir(op.join(self.data_root, dirname))])
        return self._genotypes

    def flyhdf5(self, genotype='VT37804_TNTE', flyid='2012-12-18-16-04-06'):
        return op.join(self.data_root, genotype, '%s.hdf5' % flyid)

    def is_control(self, genotype):
        return 'TNTin' in genotype

    def all_genotypes_dirs(self):
        return [op.join(self.data_root, genotype) for genotype in self.genotypes()]


# DCN
DCN_PROJECT = PBProject(name='DCN')
DCN_TEST_HDF5 = DCN_PROJECT.flyhdf5('VT37804_TNTE', '2012-12-18-16-04-06')

# HS
HS_PROJECT = PBProject(name='HS')
HS_TEST_HDF5 = DCN_PROJECT.flyhdf5('VT58487_tshirtgal80_TNTE', '2013-04-11-16-31-36')


# label -> project
PB_PROJECTS = {
    'hs': HS_PROJECT,
    'dcn': DCN_PROJECT,
}