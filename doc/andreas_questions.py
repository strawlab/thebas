# Hi Santi, I am trying to understand the concept of thebas
# But it's kind of tough, because there's sooo much bookkeeping in it.
#
# So now I am trying to understand the import pieces. I boiled thebas
# down to what I think is the bare minimum, and maybe you can quickly confirm
# if I am right


# XXX QUESTION 1:
#  So we model that each infividual fly's amplitude and phase distributions
#  sample their parameters from these group distributions?
#
group_phase_kappa = pymc.Uniform("phase_kappa", lower=1e-9, upper=10)
group_phase_mu = pymc.CircVonMises("phase_mu", mu=0, kappa=1.0)
group_amp_alpha = pymc.Normal("amp_alpha", value=1.0, mu=1.0, tau=0.01)
group_amp_beta = pymc.Uniform("amp_beta", value=25, lower=1.0, upper=50.0)


def fly_model(fly):

    perturbation_freq = fly['freq']       # single value
    time = fly['wba_t']                   # wingbeat time
    signal = fly['wba']                   # delta wingbeat amp
    flyid = 'fly=%s' % str(fly['flyid'])  # a unique id for each fly


    # XXX QUESTION 2:
    #  These are the distributions from which the individual fly samples
    #  its amplitude and phase?
    #
    phase = pymc.CircVonMises('phase_' + flyid,
                              mu=group_phase_mu,
                              kappa=group_phase_kappa)
    amplitude = pymc.HalfCauchy('amplitude_' + flyid,
                                alpha=group_amplitude_alpha,
                                beta=group_amplitude_beta)


    # XXX QUESTION 3:
    #  This is the ideal response of the fly. Sampling from phase and amplitude
    #  distributions given above?
    #
    @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
    def modeled_signal(amplitude=amplitude, phase=phase):
        return amplitude * np.sin(2*np.pi * perturbation_freq * time + phase)


    # XXX QUESTION 4:
    #  y represents the likelyhood that given sampled parameters used in model_signal
    #  what is the chance, that signal is measured?
    #
    sigma = pymc.Uniform('sigma_' + flyid, 1e-9, 10.)
    # --- likelihood (each observation is modeled as a Gaussian, all share their sd)
    y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

    # --- fly model
    return [y, phase, amplitude, sigma]

# Collect all model parameters in model
model = [group_phase_mu, group_phase_kappa, group_amp_alpha, group_amp_beta]
for fly in flies:
    model += fly_model(fly)

# Sample the model
M = pymc.MCMC(model, db='pickle', dbname='blah.pickle', name='mytest')
M.sample(iter=80000, burn=20000, progress_bar=False)
