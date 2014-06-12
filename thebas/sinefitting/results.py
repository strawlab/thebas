# coding=utf-8
"""Results storage, postprocessing and summaries."""
import os.path as op
from pprint import pprint
import weakref
import time

import joblib
import pymc
from pymc.Matplot import plot, summary_plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from thebas.misc import ensure_dir


class MCMCRunManager(object):
    """Manages a single MCMC result in disk."""
    def __init__(self, root_dir, name=None, backend='pickle'):
        super(MCMCRunManager, self).__init__()

        # root-dir and name
        if name is None:
            self.name = op.basename(root_dir)
        else:
            root_dir = op.join(root_dir, name)
        self.root_dir = ensure_dir(root_dir)

        # Models storage
        self.model_dir = ensure_dir(op.join(root_dir, 'model'))
        self.model_pickle = op.join(self.model_dir, '%s.pickle' % self.name)
        self.model_txt = op.join(self.model_dir, '%s.py' % self.name)
        self.model_dot = op.join(self.model_dir, '%s.dot' % self.name)
        self.model_png = op.join(self.model_dir, '%s.png' % self.name)

        # Traces storage
        self.db_dir = ensure_dir(op.join(self.root_dir, 'db'))
        self.backend = backend
        self.db_file = op.join(self.db_dir, '%s.pymc.%s' % (self.name, self.backend))
        self._db = None

        # Plots storage
        self.plots_dir = ensure_dir(op.join(self.root_dir, 'plots'))

        # Stats storage
        self.stats_dir = ensure_dir(op.join(self.root_dir, 'stats'))

        # Data storage - only if we really want to keep provenance clear
        self.data_dir = ensure_dir(op.join(self.root_dir, 'data'))
        self.data_pickle = op.join(self.data_dir, 'data.pickle')

        # Done file
        self.done_file = op.join(self.root_dir, 'DONE')

    def is_done(self):
        return op.isfile(self.done_file)

    def save_model_txt(self, txt):
        with open(self.model_txt, 'w') as writer:
            writer.write(txt)

    def load_model_txt(self):
        try:
            with open(self.model_txt) as reader:
                return reader.read()
        except:
            return None

    def save_data(self, data):
        data.to_pickle(self.data_pickle)

    def load_data(self):
        try:
            return pd.read_pickle(self.data_pickle)
        except:
            return None

    def save_pymc_model_dict(self, model_dict):
        try:
            import dill
            with open(self.model_pickle, 'w') as writer:
                dill.dump(model_dict, writer, dill.HIGHEST_PROTOCOL)
        except:
            print 'Move to PyMC3 and hope that theano allows serialization...'
            raise  # No way to pickle these nasty fortrans, does PyMC allow to serialize models like this?

    def pymc_db(self):
        # WARNING: weakrefed cache
        traces = self._db() if self._db is not None else None
        if traces is None:
            backend = getattr(pymc.database, self.backend)
            traces = backend.load(self.db_file)
            self._db = weakref.ref(traces)
        return traces

    def traces(self, varname):
        """Returns a numpy array with the traces for the variable, one squeezed row per chain."""
        # TODO: Q&D to eliminate delays on reading, rethink
        cache_file = op.join(self.db_dir, '%s.%s' % (varname, 'pickle'))
        if not op.isfile(cache_file):
            # We could instead combine all the chains into a long one with chain=None
            # See e.g. https://github.com/pymc-devs/pymc/issues/144
            traces = np.array([self.pymc_db().trace(varname, chain=chain)[:].squeeze()
                               for chain in xrange(self.num_chains())])
            joblib.dump(traces, cache_file, compress=3)
            return traces
        return joblib.load(cache_file)

    def pymctraces(self, varname):
        traces = self.pymc_db()._traces
        print traces
        return traces[varname]

    def varnames(self):
        # TODO: Q&D to eliminate delays on reading, rethink
        cache_file = op.join(self.db_dir, 'tracenames.pickled')
        if not op.isfile(cache_file):
            trace_names = self.pymc_db().trace_names
            joblib.dump(trace_names, cache_file, compress=3)
            return trace_names
        return joblib.load(cache_file)

    def num_chains(self):
        return len(self.varnames())

    def run_sampler(self, model,
                    iters=80000, burn=20000, num_chains=4,
                    force=False,
                    doplot=True, showplots=False, progress_bar=False,
                    mapstart=False):

        print('MCMC for %s' % self.name)

        if self.is_done():
            if not force:
                print('\tAlready done, skipping...')
                return self.db_file
            else:
                print('\tWARNING: recomputing, there might be spurious files from previous runs...')
        # Let's graph the model
        graph = pymc.graph.dag(pymc.Model(model),
                               name=self.name,
                               path=self.model_dir)
        graph.write_png(op.join(self.model_dir, self.name + '.png'))

        start = time.time()

        if mapstart:
            # See http://stronginference.com/post/burn-in-and-other-mcmc-folklore
            # BUT WARNING, WOULD THIS MAKE MULTIPLE CHAIN START BE OVERLY CORRELATED?
            try:
                from pymc import MAP
                print('\tFinding MAP estimates...')
                M = MAP(model)
                M.fit()
                model = M.variables
                print('\tMAP estimates found...')
            except Exception, e:
                print('\tMAP Failed...', str(e))

        M = pymc.MCMC(model, db=self.backend, dbname=self.db_file, name=self.name)

        for chain in xrange(num_chains):
            print('\tChain %d of %d' % (chain + 1, num_chains))
            M.sample(iter=iters, burn=burn, progress_bar=progress_bar)
            try:
                if doplot:  # Summaries for the chain
                    plot(M, suffix='__' + self.name + '__chain=%d' % chain, path=self.plots_dir, verbose=0)
                    summary_plot(M, name='summary' + self.name + '__chain=%d' % chain, path=self.plots_dir + '/')
                    # TODO: report no op.join (+'/') bug to pymc people
                if showplots:
                    plt.show()
                chain_stats = M.stats(chain=chain)
                with open(op.join(self.stats_dir, 'stats__chain=%d' % chain), 'w') as writer:
                    pprint(chain_stats, writer)
            except Exception, e:
                print('\tError plotting or summarizing')
                print(str(e))

        with open(self.done_file, 'w') as writer:
            writer.write('Taken %.2f seconds' % (time.time() - start))

        return self.db_file

#
# TODO: look at hdf5ea backend if we were to get traces of big data vectors
#
# TODO: disable pytables naturalnamewarning
# https://www.mail-archive.com/pytables-users@lists.sourceforge.net/msg01130.html
#
# TODO: all the database backends in pymc are disappointing:
#   - pickle is pickle (all in memory, slow, and pickled serialization has the usual mantainability problems)
#   - sqlite and text are oververbose
#   - ram is in ram
#   - hdf5 and hdf5a fail too easily
#  We would need something that do not keep the traces in memory and is easy on space.
#  In the meantime, we can just use pickle (we could monkeypatch to use joblib).
#  We then cache traces using joblib when requested.
#  Writing an efficient backend just using plain datafiles should be simple.
#
# TODO: easily restart chains (instead of sampling new chains)
#
# TODO: check that MAP has worked (no nans and the like)
#