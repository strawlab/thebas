# coding=utf-8
"""Bayesian Models to fit by the data acquired in the perturbation experiment."""
import numpy as np
import pymc.distributions


def perturbation_signal(times, amplitude=5., phase=0., mean_val=0., freq=0.5, freq_is_angular=True):
    """Computes the perturbation signal for certain parameters.
    In Lisa's original experiments, the perturbation signal stimuli was a sinewave using
    the default parameters of this function (but for frequency).
    """
    if freq_is_angular:
        return amplitude * np.sin(freq * times + phase) + mean_val
    return amplitude * np.sin(2 * np.pi * freq * times + phase) + mean_val


def sanity_checks(data, min_num_obs=10, min_num_flies=2):
    """
    Checks the data for a group, making sure that:
      - There is only one frequency present
      - There is only one genotype present
      - The fly ids are unique
      - All flies have enough data (number of observations)

    Parameters
    ----------
    data: pandas DataFrame
        The data of the group (and nothing else)

    min_num_obs: int, default 10
        The minimum number of observations we need to have for a fly to be modelled

    min_num_flies: int, default 2
        The minimum number of flies we need to have after flies without enough observations have been removed

    Returns
    -------
    A new DataFrame where the flies with too little data have been removed.
    """
    # Check the data corresponds to just one frequency
    present_freqs = data['freq'].unique()
    if len(present_freqs) > 1:
        raise Exception('Each group must correspond to only one frequency')
    # Check that the data corresponds to just one genotype
    present_groups = data['genotype'].unique()
    if len(present_groups) > 1:
        raise Exception('Each group must correspond to only one genotype')
    # Check that all the flys have different ids
    if len(data['flyid'].unique()) < len(data['flyid']):
        raise Exception('There are duplicated fly ids')
    # Remove flies with not enough observations
    data = data[data['wba'].apply(len) >= min_num_obs]
    if len(data) < min_num_flies:
        raise Exception('We have less than %d flies with enough data' % min_num_flies)
    return data

#############################
# Models
#############################
#
# To add a new model:
#
#   - copy and paste example function "gpa_t1"
#     modify at will (I recommend simplifying or removing the docstring)
#     For provenance tracking purposes, I also recommend really copying and pasting
#     (as opposed to retrieving by calling an existing factory and slightly modifying it)
#
#   - add it to the dictionary "MODEL_FACTORIES"
#
#############################


def gpa_t1(group_id, group_data, min_num_obs=10, SMALL=1E-9):
    """Group phase and group amplitude, model t1.

    Model highlights (details on the function body):
      group hyperpriors on phase and amplitude
        amplitude ~ HalfCauchy(alpha_group, beta_group)
        phase ~ VonMises(mu_group, kappa_group)
      an stochastic (with strong prior) for omega

    Parameters
    ----------
    group_id: string
        A string identifying the group (usually the genotype)

    group_data: pandas DataFrame
        A DataFrame containing the measured response, stimulus information and others
        (see perturbation_experiment_data)

    min_num_obs: int, default=10
        The minimum number of observations a fly must have to be considered

    Returns
    -------
    A tuple (model, sample_params)
      - model is the list of variables defining a pymc2 model
      - sample_params is a dictionary of options to pass to MCMCRunManager.sample
        for example: {'mapstart': True}

    """

    # check and clean data
    group_data = sanity_checks(group_data, min_num_obs=min_num_obs)

    # --- group hyperpriors...
    group_id = 'group="%s"' % group_id
    # group phase - parameters for Von Mises
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=SMALL, upper=10)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=1.)
    # group amplitude - parameters for Half Cauchy
    mean_group_amplitude = np.mean([np.max(np.abs(fly['wba'])) for _, fly in group_data.iterrows()])
    group_amplitude_alpha = pymc.Normal('amplitudeAlpha_' + group_id,
                                        value=mean_group_amplitude,
                                        mu=mean_group_amplitude,
                                        tau=1/100.)
    group_amplitude_beta = pymc.Uniform('amplitudeBeta_' + group_id,
                                        value=25,
                                        lower=1.,
                                        upper=50.)  # should probably use a stronger prior

    # --- each individual fly...
    def fly_model(fly):

        perturbation_freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        max_amplitude = np.max(np.abs(signal))
        flyid = 'fly=%s' % str(fly['flyid'])

        # --- priors

        # phase ~ VonMises(mu_group, kappa_group)
        phase = pymc.CircVonMises('phase_' + flyid,
                                  mu=group_phase_mu,
                                  kappa=group_phase_kappa)
        # amplitude ~ HalfCauchy(group_amplitude_alpha, group_amplitude_beta)
        amplitude = pymc.HalfCauchy('amplitude_' + flyid,
                                    # value=mean_group_amplitude,
                                    # alpha=mean_group_amplitude,
                                    # beta=25  # beta=25 is a common choice
                                    alpha=group_amplitude_alpha,
                                    beta=group_amplitude_beta)

        # uninformative DC
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        # frequency narrowly distributed around the frequency of the perturbation
        freq = pymc.Normal('freq_' + flyid, mu=perturbation_freq, tau=100.)
        # freq = perturbation_freq
        # uninformative sd for the Normal likelihood
        sigma = pymc.Uniform('sigma_' + flyid, SMALL, 10.)

        # the modeled signal generation process
        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(freq=freq, times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):
            return perturbation_signal(times=times,
                                       amplitude=amplitude,
                                       phase=phase,
                                       mean_val=mean_val,
                                       freq=freq)

        # --- likelihood (each observation is modeled as a Gaussian, all share their sd)
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        # --- fly model
        return [y, phase, amplitude, mean_val, sigma]  # freq,

    # --- put all together
    model = [group_phase_mu, group_phase_kappa, group_amplitude_alpha, group_amplitude_beta]
    for _, fly in group_data.iterrows():
        model += fly_model(fly)
    return model, {}


def gpa_t1_slice(group_id, group_data, min_num_obs=10, SMALL=1E-9):
    """Group phase and group amplitude, model t1.

    Model highlights (details on the function body):
      group hyperpriors on phase and amplitude
        amplitude ~ HalfCauchy(alpha_group, beta_group)
        phase ~ VonMises(mu_group, kappa_group)
      slice sampling for the group hyperpriors
      an stochastic (with strong prior) for omega
    """
    # check and clean data
    group_data = sanity_checks(group_data, min_num_obs=min_num_obs)

    # --- group hyperpriors...
    group_id = 'group="%s"' % group_id
    # group phase - parameters for Von Mises
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=SMALL, upper=10)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=1.)
    # group amplitude - parameters for Half Cauchy
    mean_group_amplitude = np.mean([np.max(np.abs(fly['wba'])) for _, fly in group_data.iterrows()])
    group_amplitude_alpha = pymc.Normal('amplitudeAlpha_' + group_id,
                                        value=mean_group_amplitude,
                                        mu=mean_group_amplitude,
                                        tau=1/100.)
    group_amplitude_beta = pymc.Uniform('amplitudeBeta_' + group_id,
                                        value=25,
                                        lower=1.,
                                        upper=50.)  # should probably use a stronger prior
    # group_amplitude_alpha = pymc.HalfCauchy('amplitudeAlpha_' + group_id, alpha=mean_group_amplitude, beta=25.)
    # group_amplitude_beta = pymc.Uniform('amplitudeBeta_' + group_id, lower=SMALL, upper=100.)

    # --- each individual fly...
    def fly_model(fly):

        perturbation_freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        max_amplitude = np.max(np.abs(signal))
        flyid = 'fly=%s' % str(fly['flyid'])

        # --- priors

        # phase ~ VonMises(mu_group, kappa_group)
        phase = pymc.CircVonMises('phase_' + flyid,
                                  mu=group_phase_mu,
                                  kappa=group_phase_kappa)
        # amplitude ~ HalfCauchy(group_amplitude_alpha, group_amplitude_beta)
        amplitude = pymc.HalfCauchy('amplitude_' + flyid,
                                    alpha=group_amplitude_alpha,
                                    beta=group_amplitude_beta)
        # uninformative DC
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        # frequency narrowly distributed around the frequency of the perturbation
        freq = pymc.Normal('freq_' + flyid, mu=perturbation_freq, tau=100.)
        # uninformative noise for the Normal likelihood
        sigma = pymc.Uniform('sigma_' + flyid, SMALL, 10.)

        # the modeled signal generation process
        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(freq=freq, times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):
            return perturbation_signal(times=times,
                                       amplitude=amplitude,
                                       phase=phase,
                                       mean_val=mean_val,
                                       freq=freq)

        # --- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        # --- fly model
        return [y, phase, amplitude, freq, mean_val, sigma]

    # --- put all together
    model = [group_phase_mu, group_phase_kappa, group_amplitude_alpha, group_amplitude_beta]
    for _, fly in group_data.iterrows():
        model += fly_model(fly)

    return model, {'mapstart': False,
                   'step_methods': [(group_phase_kappa, pymc.Slicer, {}),
                                    (group_phase_mu, pymc.Slicer, {}),
                                    (group_amplitude_alpha, pymc.Slicer, {}),
                                    (group_amplitude_beta, pymc.Slicer, {})]}  # N.B. Slicer requires pymc >=2.3.2


def gpa_t2_slice(group_id, group_data, min_num_obs=10, SMALL=1E-9):
    """Group phase and group amplitude, model t2.

    Model highlights (details on the function body):
      group hyperpriors on phase and amplitude
        amplitude ~ HalfCauchy(alpha_group, beta_group)
        phase ~ VonMises(mu_group, kappa_group)
      slice sampling for the group hyperpriors
      an stochastic (with strong prior) for omega
    """
    # check and clean data
    group_data = sanity_checks(group_data, min_num_obs=min_num_obs)

    # --- group hyperpriors...
    group_id = 'group="%s"' % group_id
    # group phase - parameters for Von Mises
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=SMALL, upper=10)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=1.)
    # group amplitude - parameters for Half Cauchy
    mean_group_amplitude = np.mean([np.max(np.abs(fly['wba'])) for _, fly in group_data.iterrows()])
    group_amplitude_alpha = pymc.HalfCauchy('amplitudeAlpha_' + group_id, alpha=mean_group_amplitude, beta=10.)
    group_amplitude_beta = pymc.Uniform('amplitudeBeta_' + group_id, lower=SMALL, upper=100.)
    # should probably use a stronger prior

    # --- each individual fly...
    def fly_model(fly):

        perturbation_freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        max_amplitude = np.max(np.abs(signal))
        flyid = 'fly=%s' % str(fly['flyid'])

        # --- priors

        # phase ~ VonMises(mu_group, kappa_group)
        phase = pymc.CircVonMises('phase_' + flyid,
                                  mu=group_phase_mu,
                                  kappa=group_phase_kappa)
        # amplitude ~ HalfCauchy(group_amplitude_alpha, group_amplitude_beta)
        amplitude = pymc.HalfCauchy('amplitude_' + flyid,
                                    alpha=group_amplitude_alpha,
                                    beta=group_amplitude_beta)
        # uninformative DC
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        # frequency narrowly distributed around the frequency of the perturbation
        freq = pymc.Normal('freq_' + flyid, mu=perturbation_freq, tau=100.)
        # uninformative noise for the Normal likelihood
        sigma = pymc.Uniform('sigma_' + flyid, 1E-9, 10.)

        # the modeled signal generation process
        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(freq=perturbation_freq,
                           times=time,
                           amplitude=amplitude,
                           phase=phase,
                           mean_val=mean_val):
            return perturbation_signal(times=times,
                                       amplitude=amplitude,
                                       phase=phase,
                                       mean_val=mean_val,
                                       freq=freq)

        # --- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        # --- fly model
        return [y, phase, amplitude, freq, mean_val, sigma]

    # --- put all together
    model = [group_phase_mu, group_phase_kappa, group_amplitude_alpha, group_amplitude_beta]
    for _, fly in group_data.iterrows():
        model += fly_model(fly)

    return model, {'mapstart': False,
                   'step_methods': [(group_phase_kappa, pymc.Slicer, {}),
                                    (group_phase_mu, pymc.Slicer, {}),
                                    (group_amplitude_alpha, pymc.Slicer, {}),
                                    (group_amplitude_beta, pymc.Slicer, {})]}  # N.B. Slicer requires pymc >=2.3.2


def gpad_t1_slice(group_id, group_data, min_num_obs=10, SMALL=1E-9):
    """Group phase, group amplitude, group DC model t1.

    Model highlights (details on the function body):
      group hyperpriors on phase, amplitude and DC
        amplitude ~ HalfCauchy(alpha_group, beta_group)
        phase ~ VonMises(mu_group, kappa_group)
        DC ~ Normal(mu_group, tau_group)
      slice sampling for the group hyperpriors
      an stochastic (with strong prior) for omega
    """
    #
    # FIXME: this model never worked... revisit if necessary
    #        it seeks to shrink DC too (seems that one genotype is more biased right than the other...)
    #

    # check and clean data
    group_data = sanity_checks(group_data, min_num_obs=min_num_obs)

    # --- group hyperpriors...
    group_id = 'group="%s"' % group_id
    # group phase - parameters for Von Mises
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=SMALL, upper=10)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=1.)
    # group amplitude - parameters for Half Cauchy
    mean_group_amplitude = np.mean([np.max(np.abs(fly['wba'])) for _, fly in group_data.iterrows()])
    group_amplitude_alpha = pymc.HalfCauchy('amplitudeAlpha_' + group_id, alpha=mean_group_amplitude, beta=10.)
    group_amplitude_beta = pymc.Uniform('amplitudeBeta_' + group_id, lower=SMALL, upper=1000.)
    # group DC - uninformative priors and no data-initialised
    group_dc_mu = pymc.Normal('dcMu_' + group_id, mu=0, tau=10)
    group_dc_std = pymc.Uniform('dcStd_' + group_id, lower=SMALL, upper=100)

    # --- each individual fly...
    def fly_model(fly):

        perturbation_freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        flyid = 'fly=%s' % str(fly['flyid'])

        # --- priors

        # phase ~ VonMises(mu_group, kappa_group)
        phase = pymc.CircVonMises('phase_' + flyid,
                                  mu=group_phase_mu,
                                  kappa=group_phase_kappa)
        # amplitude ~ HalfCauchy(group_amplitude_alpha, group_amplitude_beta)
        amplitude = pymc.HalfCauchy('amplitude_' + flyid,
                                    alpha=group_amplitude_alpha,
                                    beta=group_amplitude_beta)
        # uninformative DC
        mean_val = pymc.Normal('DC_' + flyid, mu=group_dc_mu, tau=1./group_dc_std**2)
        # frequency narrowly distributed around the frequency of the perturbation
        freq = pymc.Normal('freq_' + flyid, mu=perturbation_freq, tau=100.)
        # uninformative noise for the Normal likelihood
        sigma = pymc.Uniform('sigma_' + flyid, SMALL, 10.)

        # the modeled signal generation process
        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(freq=perturbation_freq,
                           times=time,
                           amplitude=amplitude,
                           phase=phase,
                           mean_val=mean_val):
            return perturbation_signal(times=times,
                                       amplitude=amplitude,
                                       phase=phase,
                                       mean_val=mean_val,
                                       freq=freq)

        # --- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        # --- fly model
        return [y, phase, amplitude, freq, mean_val, sigma]

    # --- put all together
    model = [group_phase_mu, group_phase_kappa, group_amplitude_alpha, group_amplitude_beta]
    for _, fly in group_data.iterrows():
        model += fly_model(fly)

    return model, {'mapstart': False,
                   'step_methods': [(group_phase_kappa, pymc.Slicer, {}),
                                    (group_phase_mu, pymc.Slicer, {}),
                                    (group_amplitude_alpha, pymc.Slicer, {}),
                                    (group_amplitude_beta, pymc.Slicer, {}),
                                    (group_dc_mu, pymc.Slicer, {}),
                                    (group_dc_std, pymc.Slicer, {})]}  # N.B. Slicer requires pymc >=2.3.2

#########################
# Older models
#########################


def gpa3(group_id, group_data, min_num_obs=10):

    # check and clean data
    group_data = sanity_checks(group_data, min_num_obs=min_num_obs)

    # --- group hyperpriors...
    group_id = 'group="%s"' % group_id
    # group phase
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=1E-9, upper=100)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=0)
    group_phase = pymc.CircVonMises('phase_' + group_id, group_phase_mu, group_phase_kappa)
    # group amplitude
    max_amplitude = np.max([np.max(np.abs(fly.wba)) for _, fly in group_data.iterrows()])
    group_amplitude = pymc.Uniform('amplitude_' + group_id, lower=1E-6, upper=max_amplitude)
    # just max_amplitude is not ok

    def fly_model(fly):

        perturbation_freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        flyid = 'fly=%s' % str(fly['flyid'])

        # --- priors
        phase_kappa = pymc.Uniform('kappa_' + flyid, 0, 10.0)    # hyperparameter for the phase (kappa ~ Uniform(0, 10))
        phase = pymc.CircVonMises('phase_' + flyid,              # mu shrinked to the group phase
                                  mu=group_phase,
                                  kappa=phase_kappa)             # phase ~ VonMises(mu_group, kappa_group)
        # Uninformative for the signal's DC, amplitude and noise's sd...
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        amplitude = pymc.Uniform('amplitude_' + flyid, lower=0, upper=2 * group_amplitude)
        sigma = pymc.Uniform('sigma_' + flyid, 0, max_amplitude)

        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):  # We just "fix" frequency
            return perturbation_signal(times=times,
                                       amplitude=amplitude,
                                       phase=phase,
                                       mean_val=mean_val,
                                       freq=perturbation_freq)

        # --- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        # --- fly model
        return [y, amplitude, phase, sigma, mean_val]

    # --- put all together
    model = [group_phase, group_phase_mu, group_phase_kappa, group_amplitude]
    for _, fly in group_data.iterrows():
        model += fly_model(fly)
    return model, {}


def gpa3hc1(group_id, group_data, min_num_obs=10):
    """Like GPA3 but using a Half-Cauchy instead of a uniform amplitude."""

    # check and clean data
    group_data = sanity_checks(group_data, min_num_obs=min_num_obs)

    # --- group hyperpriors...
    group_id = 'group="%s"' % group_id
    # group phase
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=1E-9, upper=100)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=0)
    group_phase = pymc.CircVonMises('phase_' + group_id, group_phase_mu, group_phase_kappa)
    # group amplitude
    max_amplitude = np.max([np.max(np.abs(fly.wba)) for _, fly in group_data.iterrows()])
    max_median_amplitude = np.max([np.median(np.abs(fly.wba)) for _, fly in group_data.iterrows()])
    group_amplitude = pymc.HalfCauchy('amplitude_' + group_id,
                                      alpha=max_median_amplitude,
                                      beta=25.)  # beta=25 is a common choice for HC hyperpriors
    # just max_amplitude is not ok

    def fly_model(fly):

        perturbation_freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        flyid = 'fly=%s' % str(fly['flyid'])

        # --- priors
        phase_kappa = pymc.Uniform('kappa_' + flyid, 0, 10.0)    # hyperparameter for the phase (kappa ~ Uniform(0, 10))
        phase = pymc.CircVonMises('phase_' + flyid,              # mu shrinked to the group phase
                                  mu=group_phase,
                                  kappa=phase_kappa)             # phase ~ VonMises(mu_group, kappa_group)
        # Uninformative for the signal's DC, amplitude and noise's sd...
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        amplitude = pymc.Uniform('amplitude_' + flyid, lower=0, upper=2 * group_amplitude)
        sigma = pymc.Uniform('sigma_' + flyid, 0, max_amplitude)

        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):  # We just "fix" frequency
            return perturbation_signal(times=times,
                                       amplitude=amplitude,
                                       phase=phase,
                                       mean_val=mean_val,
                                       freq=perturbation_freq)

        # --- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        # --- fly model
        return [y, amplitude, phase, sigma, mean_val]

    # --- put all together
    model = [group_phase, group_phase_mu, group_phase_kappa, group_amplitude]
    for _, fly in group_data.iterrows():
        model += fly_model(fly)
    return model, {}


def gpa3hc2(group_id, group_data, min_num_obs=10):
    """Like gpa3hc1 but using a Half-Cauchy instead of a uniform amplitude also for individual flies."""

    # check and clean data
    group_data = sanity_checks(group_data, min_num_obs=min_num_obs)

    # --- group hyperpriors...
    group_id = 'group="%s"' % group_id
    # group phase
    group_phase_kappa = pymc.Uniform('phaseKappa_' + group_id, lower=1E-9, upper=100)
    group_phase_mu = pymc.CircVonMises('phaseMu_' + group_id, mu=0, kappa=0)
    group_phase = pymc.CircVonMises('phase_' + group_id, group_phase_mu, group_phase_kappa)
    # group amplitude
    max_amplitude = np.max([np.max(np.abs(fly.wba)) for _, fly in group_data.iterrows()])
    max_median_amplitude = np.max([np.median(np.abs(fly.wba)) for _, fly in group_data.iterrows()])
    group_amplitude = pymc.HalfCauchy('amplitude_' + group_id,
                                      alpha=max_median_amplitude,
                                      beta=25.)  # beta=25 is a common choice for HC hyperpriors

    def fly_model(fly):

        perturbation_freq = fly['freq']
        time = fly['wba_t']
        signal = fly['wba']
        flyid = 'fly=%s' % str(fly['flyid'])

        # --- priors
        phase_kappa = pymc.Uniform('kappa_' + flyid, 0, 10.0)    # hyperparameter for the phase (kappa ~ Uniform(0, 10))
        phase = pymc.CircVonMises('phase_' + flyid,              # mu shrinked to the group phase
                                  mu=group_phase,
                                  kappa=phase_kappa)             # phase ~ VonMises(mu_group, kappa_group)
        # Uninformative for the signal's DC, amplitude and noise's sd...
        mean_val = pymc.Uniform('DC_' + flyid, -max_amplitude, max_amplitude)
        amplitude = pymc.HalfCauchy('amplitude_' + flyid, alpha=group_amplitude, beta=25)
        sigma = pymc.Uniform('sigma_' + flyid, 0, max_amplitude)

        @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
        def modeled_signal(times=time, amplitude=amplitude, phase=phase, mean_val=mean_val):  # We just "fix" frequency
            return perturbation_signal(times=times,
                                       amplitude=amplitude,
                                       phase=phase,
                                       mean_val=mean_val,
                                       freq=perturbation_freq)

        # --- likelihood
        y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)

        # --- fly model
        return [y, amplitude, phase, sigma, mean_val]

    # --- put all together
    model = [group_phase, group_phase_mu, group_phase_kappa, group_amplitude]
    for _, fly in group_data.iterrows():
        model += fly_model(fly)
    return model, {}


MODEL_FACTORIES = {model.__name__: model for model in [
    gpa_t1,
    gpa_t1_slice,
    gpa_t2_slice,
    gpad_t1_slice,
    gpa3,
    gpa3hc1,
    gpa3hc2,
]}


def instantiate_model(pbproject,
                      freq=2.,
                      genotype='VT37804_TNTE',
                      model_id='gpa_t1'):
    """
    Instantiates a model for the given data coordinates,

    Parameters
    ----------
    pbproject: PBProject instance
      At the moment, this will be HS_PROJECT or DCN_PROJECT

    freq : float, default 2
      The frequency of the perturbation

    genotype_id:
      The genotype of the flies

    model_id:
      A model id (from MODEL_FACTORIES)

    Returns
    -------
    group_id (a string identifying the data group), group_data (the data for the group),
    model_factory (the function that instantiates the model), model (the instantiated model),
    sample_params (a dictionary param_name -> param_value to configure the sampler).
    """

    # We will need to access the data
    from thebas.sinefitting.data import perturbation_data_to_records

    # Pick the specified model
    if model_id not in MODEL_FACTORIES:
        raise Exception('Do not know a model with id %s' % model_id)
    model_factory = MODEL_FACTORIES[model_id]

    # Read and select the data
    group_id = 'freq=%g__genotype=%s' % (freq, genotype)
    group_data = perturbation_data_to_records(pbproject=pbproject)
    group_data = group_data[(group_data['freq'] == freq) &  # select the group data
                            (group_data['genotype'] == genotype)]

    # Instantiate the model
    model, sample_params = model_factory(group_id, group_data)

    return group_id, group_data, model_factory, model, sample_params


###############################################
#
# Thomas wisdom, tips and suggestions
#
#   - Do not add too many levels to the hierarchy; for example, shrink directly mu and kappa to mu_group and kappa_group
#     (instead of sampling from a group Von-Mises and then make that VM inform the flies).
#
#   - Usually the most difficult challenge is to fit the top-level (group) variables.
#     He recommended using slice sampling for these (or in general, try other samplers with misbehaving variables).
#
#   - Beware of disconnected priors or variables that are informed by little data in the top levels.
#     That is a surprisingly common error.
#
#   - "More than 100 observations should be good" - Thomas rule of thumb; ability to handle small datasets is
#     why he moved to Bayesian methods in the first instance.
#
#   - He suggested also to pool data and build a giant model, maybe with experiments in an intermediate level.
#     Even if then he seemed less convinced after explaining that is not trivial or even reasonable to do,
#     it is worth give it a second thought.
#
#   - He suggested to not fix things like omega
#     (maybe just allow a normal with small variance prior to control that value too)
#
#   - For the amplitude distributions, he suggested to try with other positive distros (he mentioned half-t, Gamma,
#     half-Normal, half-Cauchy); I already tried some in the past without success (he was not surprised that Gamma
#     was problematic and that is often his experience too). Also, for these, it is good to try priors different than
#     uniform for spread-parameters. He warned with a couple of plots-in-the-envelope about how easy it is for chains
#     over little variance variables to get stuck for long time regions sampling zero-or-constant values.
#
#   - He suggested not to use data-scale related bounds for priors on variables controlling variance
#     (he said "use 10, if you hit the limit at sampling use 20"). He referred to a couple of back-of-the-envelope
#     plots exemplifying on least-squares regression and extending his example to fitting a sine function.
#     Recommended reading this from his blog:
#       - http://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/
#       - http://twiecki.github.io/blog/2013/08/12/bayesian-glms-2/  (broken link)
#         http://nbviewer.ipython.org/github/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/
#         GLM-blog-robust.ipynb
#       - http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
#
# Remember:
#   - write a followup email thanking him, updating with results, and maybe with an invitation to visit us.
#   - allow him access to thebas (and hell rename to tebas)
#
# Most of these things were already in my comments but now it is clearer how to implement them.
# Also try a group DC (as desired by Lisa, to find group biases on flying to the right or left)
# And also look at the angles (ga), we were commenting something about a few months ago
#
###############################################
#
# FIXME: maybe here is best to specify samplers, MAP initialisation etc
#        so that provenance has just one place...
#
###############################################
#
# TODO: implement sine-wave pdfs
#   http://books.google.ie/books?id=kKf9PmmazzMC&pg=PA61&lpg=PA61&dq=probability+density+function+for+sine+wave&source=bl&ots=Ifv09awVE0&sig=8lY76M9iDN86n3jJxcYXLF5wfXk&hl=en&sa=X&ei=mOT-U-7jKNLOaLukgNAP&sqi=2&ved=0CCYQ6AEwAQ#v=onepage&q=probability%20density%20function%20for%20sine%20wave&f=false
#   http://atif-razzaq.blogspot.co.at/2011/02/probability-density-function-pdf-of.html
#
###############################################
#
# DONE
# DONE As soon as we add a group hyperprior to the HalfCauchy amplitudes,
# DONE (or in general a non-negative group hyperprior to any amplitude prior)
# DONE we get ZeroProb exceptions for some flies
# DONE (similar to what was happening before with the other non-negative distros I was trying...)
# DONE   - maybe we need to clean the data
# DONE   - maybe we need to allow negative amplitudes
# DONE   - maybe the models are still not sensible
# DONE   - ...
# DONE
# Seems related to this regression in pymc 2.3.3 (the one provided in stock anaconda at the moment)
#   https://github.com/pymc-devs/pymc/issues/588
#   https://github.com/pymc-devs/pymc/issues/589
# Compiling from master seems to fix this weeks-taking annoyance (running right now)
#
###############################################
