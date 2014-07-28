# coding=utf-8
"""Bayesian Models to fit the data for the perturbation experiment."""
import pymc
import numpy as np


def perturbation_signal(times, amplitude=5, phase=0, mean_val=0, freq=0.5, freq_is_angular=True):
    """Computes the perturbation signal for certain parameters.
    The perturbation signal stimuli was a sinewave using the default parameters of this function.
    """
    if freq_is_angular:
        return amplitude * np.sin(freq * times + phase) + mean_val
    return amplitude * np.sin(2 * np.pi * freq * times + phase) + mean_val


def sanity_checks(data):
    # Check the data corresponds to just one frequency
    present_freqs = data.freq.unique()
    if len(present_freqs) > 1:
        raise Exception('Each group must correspond to only one frequency')
    # Check that the data corresponds to just one genotype
    present_groups = data.group.unique()
    if len(present_groups) > 1:
        raise Exception('Each group must correspond to only one genotype')
    # Check that all the flys have different ids
    if len(data['fly'].unique()) < len(data['fly']):
        raise Exception('There are duplicated fly ids')


def group_phase_model1(group_id, group_data, min_num_obs=10):

    sanity_checks(group_data)

    # group phase (~VonMises(group_phase_mu, group_phase_kappa))
    group_phase_kappa = pymc.Uniform(group_id + '_phase_kappa', lower=0, upper=10.0)  # kappa ~ Uniform(0, 10)
    group_phase_mu = pymc.CircVonMises(group_id + '_phase_mu', mu=0, kappa=0)  # mu~VonMises(0, 0) (uniform circular)
    group_phase = pymc.CircVonMises(group_id + '_phase', group_phase_mu, group_phase_kappa)

    def fly_model(fly):

        freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        max_amplitude = np.max(np.abs(signal))  # Any usefulness normalizing the range to [0, 1]?
        flyid = fly['fly']

        #--- priors
        phase_kappa = pymc.Uniform(flyid + '_kappa', 0, 10.0)    # hyperparameter for the phase (kappa ~ Uniform(0, 10))
        phase = pymc.CircVonMises(flyid + '_phase',              # mu shrinked to the group phase
                                  mu=group_phase,
                                  kappa=phase_kappa)             # phase ~ VonMises(mu_group, kappa_group)
        # Uninformative for the signal's DC, amplitude and noise's sd...
        mean_val = pymc.Uniform(flyid + '_mean_val', -max_amplitude, max_amplitude)  # This should probably be Normal
        amplitude = pymc.Uniform(flyid + '_amplitude', 0, max_amplitude)             # This should probably be Normal
                                                                                     # (but support only on >=0)
        sigma = pymc.Uniform(flyid + '_noise_sigma', 0, max_amplitude)  # not so clear the role here...

        # We should also get a parent frequency instead of fixing it...
        # Anything to do about closed loop?

        @pymc.deterministic(plot=False, name=flyid + '_modeled_signal')  # graph plots get clearer if we name this var
        def modeled_signal(times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):  # We just "fix" frequency
            return perturbation_signal(times=times, amplitude=amplitude, phase=phase, mean_val=mean_val, freq=freq)

        #--- likelihood
        # Always centered at the value of the real modelled signal... careful.
        y = pymc.Normal(flyid + '_y', mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        #--- model
        return [y, amplitude, phase, sigma, mean_val]

    model = [group_phase, group_phase_mu, group_phase_kappa]

    for _, fly in group_data.iterrows():
        if len(fly['wba']) > min_num_obs:
            model += fly_model(fly)

    return model


def group_phase_model2(group_id, group_data, min_num_obs=10):

    sanity_checks(group_data)

    # group phase (~VonMises(group_phase_mu, group_phase_kappa))
    group_phase_kappa = pymc.Uniform(group_id + '_phase_kappa', lower=0, upper=10.0)  # kappa ~ Uniform(0, 10)
    group_phase_mu = pymc.CircVonMises(group_id + '_phase_mu', mu=0, kappa=0)  # mu~VonMises(0, 0) (uniform circular)
    group_phase = pymc.CircVonMises(group_id + '_phase', group_phase_mu, group_phase_kappa)

    # group frequency
    group_frequency_tau = pymc.Uniform(group_id + '_frequency_tau', lower=0, upper=1000)
    bias_freq = group_data.iterrows().next()[1].freq
    group_frequency = pymc.Normal(group_id + '_frequency', mu=bias_freq, tau=group_frequency_tau)
    # Dodgy, has support on negatives

    def fly_model(fly):

        time = fly['wba_t']
        signal = fly['wba']
        max_amplitude = np.max(np.abs(signal))  # Any usefulness normalizing the range to [0, 1]?
        flyid = fly['fly']

        #--- priors
        phase_kappa = pymc.Uniform(flyid + '_kappa', 0, 10.0)    # hyperparameter for the phase (kappa ~ Uniform(0, 10))
        phase = pymc.CircVonMises(flyid + '_phase',              # mu shrinked to the group phase
                                  mu=group_phase,
                                  kappa=phase_kappa)             # phase ~ VonMises(mu_group, kappa_group)
        # Uninformative for the signal's DC, amplitude and noise's sd...
        mean_val = pymc.Uniform(flyid + '_mean_val', -max_amplitude, max_amplitude)  # This should probably be Normal
        amplitude = pymc.Uniform(flyid + '_amplitude', 0, max_amplitude)             # This should probably be Normal
                                                                                     # (but support only on >=0)
        sigma = pymc.Uniform(flyid + '_noise_sigma', 0, max_amplitude)  # not so clear the role here...
        # Frequency!
        frequency_tau = pymc.Uniform(flyid + '_frequency_tau', lower=0, upper=1000)
        frequency = pymc.Normal(flyid + '_frequency', mu=group_frequency, tau=frequency_tau)

        # We should also get a parent frequency instead of fixing it...
        # Anythong to do about closed loop?

        @pymc.deterministic(plot=False, name=flyid + '_modeled_signal')  # graph plots get clearer if we name this var
        def modeled_signal(times=time, amplitude=amplitude, frequency=frequency, phase=phase, mean_val=mean_val):
            return perturbation_signal(times=times,
                                       amplitude=amplitude, phase=phase, mean_val=mean_val, freq=frequency)

        #--- likelihood
        # Always centered at the value of the real modelled signal... careful.
        y = pymc.Normal(flyid + '_y', mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        #--- model
        return [y, amplitude, phase, sigma, mean_val, frequency_tau, frequency]

    model = [group_phase, group_phase_mu, group_phase_kappa, group_frequency_tau, group_frequency]

    for _, fly in group_data.iterrows():
        if len(fly['wba']) > min_num_obs:
            model += fly_model(fly)

    return model


def group_phase_amplitude_model3(group_id, group_data, min_num_obs=10):

    sanity_checks(group_data)

    # group phase
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=1E-9, upper=100)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=0)
    # TT remove the Vm from the group
    # group_phase = pymc.CircVonMises('phase_' + group_id, group_phase_mu, group_phase_kappa)
    # group amplitude
    max_amplitude = np.max([np.max(np.abs(fly['wba'])) for _, fly in group_data.iterrows()])
    group_amplitude = pymc.Uniform('amplitude_' + group_id, lower=0, upper=max_amplitude)
                                                                    # just max_amplitude is ok

    def fly_model(fly):

        freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        max_amplitude = np.max(np.abs(signal))  # Any usefulness normalizing the range to [0, 1]?
        flyid = fly['fly']

        #--- priors
        # phase_kappa = pymc.Uniform('kappa_' + flyid, 1E-9, 10.0)    # hyperparameter for the phase (kappa ~ Uniform(0, 10))
        phase = pymc.CircVonMises('phase_' + flyid,                   # mu shrinked to the group phase
                                  mu=group_phase_mu,
                                  kappa=group_phase_kappa)             # phase ~ VonMises(mu_group, kappa_group)
        # Uninformative for the signal's DC, amplitude and noise's sd...
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        amplitude = pymc.Uniform('amplitude_' + flyid, lower=0, upper=2*group_amplitude)
        sigma = pymc.Uniform('sigma_' + flyid, 0, 10)  # No sense max amplitude
                                                       # If it hits 10, increase it

        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):  # We just "fix" frequency
            return perturbation_signal(times=times, amplitude=amplitude, phase=phase, mean_val=mean_val, freq=freq)

        #--- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        #--- model
        return [y, amplitude, phase, sigma, mean_val]

    model = [group_phase_mu, group_phase_kappa, group_amplitude]

    for _, fly in group_data.iterrows():
        if len(fly['wba']) > min_num_obs:
            model += fly_model(fly)

    return model


def group_phase_amplitude_model4(group_id, group_data, min_num_obs=10):

    sanity_checks(group_data)

    # group phase
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=1E-9, upper=100)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=0)
    group_phase = pymc.CircVonMises('phase_' + group_id, group_phase_mu, group_phase_kappa)
    # group amplitude
    # pymc.rweibull(alpha, beta, N)
    group_amplitude_a = pymc.Uniform('amplitudeA_' + group_id, lower=0, upper=10, value=5,
                                     doc='Weibull alpha (shape) parameter')
    group_amplitude_b = pymc.Uniform('amplitudeB_' + group_id, lower=0, upper=10, value=5,
                                     doc='Weibull beta (scale) parameter')
    group_amplitude = pymc.Weibull('amplitude_' + group_id, alpha=group_amplitude_a, beta=group_amplitude_b)

    def fly_model(fly):

        freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        max_amplitude = np.max(np.abs(signal))  # Any usefulness normalizing the range to [0, 1]?
        flyid = fly['fly']

        #--- priors
        phase_kappa = pymc.Uniform('kappa_' + flyid, 0, 10.0)    # hyperparameter for the phase (kappa ~ Uniform(0, 10))
        phase = pymc.CircVonMises('phase_' + flyid,              # mu shrinked to the group phase
                                  mu=group_phase,
                                  kappa=phase_kappa)             # phase ~ VonMises(mu_group, kappa_group)
        # Uninformative for the signal's DC, amplitude and noise's sd...
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        amplitude_tau = pymc.Uniform('amplitudeTau_' + flyid, lower=1E-9, upper=10)
        amplitude = pymc.Normal('amplitude_' + flyid, mu=group_amplitude, tau=amplitude_tau)
        sigma = pymc.Uniform('sigma_' + flyid, 0, max_amplitude)

        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):  # We just "fix" frequency
            return perturbation_signal(times=times, amplitude=amplitude, phase=phase, mean_val=mean_val, freq=freq)

        #--- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        #--- model
        return [y, amplitude_tau, amplitude, phase, sigma, mean_val, amplitude]

    model = [group_phase, group_phase_mu, group_phase_kappa, group_amplitude_a, group_amplitude_b, group_amplitude]

    for _, fly in group_data.iterrows():
        if len(fly['wba']) > min_num_obs:
            model += fly_model(fly)

    return model


MODEL_FACTORIES = {
    'gp1': group_phase_model1,
    'gp2': group_phase_model2,
    'gpa3': group_phase_amplitude_model3,
    'gpa4': group_phase_amplitude_model4,
    'gpa3nomap': group_phase_amplitude_model3,
}

#
# TODO: Amplitude be a normal with mean that of the data and wide std (several times that of the data)
#
# TODO: make more convenient and robust the specification of models
#       ala kabuki, for example make systematic the attribution of names to the variables
#
# TODO: make variable names start by the random variable, then the id...
#       that would make spotting what is what much easier
#
# TODO: use daft to plot the graphical models prettier (still needs laying-ou efforts)
#       http://daft-pgm.org/
#
# TODO: model also the frequency
#
# TODO: non-negative distributions on PyMC
# sc_nonnegative_distributions = ['bernoulli', 'beta', 'betabin', 'binomial', 'chi2', 'exponential',
#                                 'exponweib', 'gamma', 'half_cauchy', 'half_normal',
#                                 'hypergeometric', 'inverse_gamma', 'lognormal',
#                                 'weibull']
#
# TODO: We are not using correctly the lognormal. We are getting log estimates.
# http://doingbayesiandataanalysis.blogspot.co.at/2013_04_01_archive.html
#
# TODO: start by MAP estimation in results
#
# TODO: use data driven values for, for example, the means
#
# TODO: check if by passing an array to the observed variables we get many observations or just one
#
# TODO: check for VonMises parameters
#
# TODO: use a better prior for the amplitude
#
