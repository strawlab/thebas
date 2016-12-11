# Hi Santi, I am trying to understand the concept of thebas
# But it's kind of tough, because there's sooo much bookkeeping in it.
#
# So now I am trying to understand the import pieces. I boiled thebas
# down to what I think is the bare minimum, and maybe you can quickly confirm
# if I am right

# ANSWER: Hi Andreas,
#
# These are hierarchical Bayesian models, where the parameter values for individual flies
# are informed by (shrinked to) the priors and the data from the group they pertain to.
# You have distilled very well the essence of the task all the way to the MCMC simulation,
# so I'm pretty sure you understand quite well all this stuff. I would make sure to
# understand both pymc2 basics and the why and the how of Bayesian inference.
# Also the bookkeeping is very important. I suggest we have a chat about it if you
# think it is necessary.
#
# So let's think about the Bayesian + MCMC big picture:
#   (One of many nice writeups: http://twiecki.github.io/blog/2015/11/10/mcmc-sampling/)
#
# What we want to know is *the distribution of theta*, the "posterior".
#   - theta: the parameters that govern the distribution of what we observe (the "data").
#   - here: theta = (group_phase_mu, group_phase_kappa, ..., fly3_phase_mu...)
# Estimate the distribution of theta?
# To inform us with the data, we apply the Bayes theorem directly:
#   p(theta|data) is proportional to p(data|theta) * p(theta)
# Or even more in words to make splicit names for each of these 3 probabilities:
#   posterior is proportional to likelihood * prior
#
# We run a simulation where we repeately sample values from theta, and stay
# more time evaluating more likely values of theta than evaluating less likely
# values for theta (where likely is defined by the data and the prior). If
# all goes well, at the end we will have spent on each "region of theta" time
# proportional to the density of that region in real posterior of theta.
# Although the model is defined on its own, we need to put it in the context
# of inference, the MCMC simulation we will use to find the posterior, to fully
# understand the code.
#
# On each iteration (of a vanilla MCMC chain), we sample a new value for theta
# based on the previous value for theta, compute the likelihood of the data
# given that new value, weight it by the given prior probability of theta
# (right hand side of our Bayes theorem equation) and "accept" the new value
# if the weighted likelihood is better than that for the previous value or if,
# by chance proportional to how good it fits the data and our prior beliefs,
# we decide to accept it.
#


# XXX QUESTION 1:
#  So we model that each infividual fly's amplitude and phase distributions
#  sample their parameters from these group distributions?
#
group_phase_kappa = pymc.Uniform("phase_kappa", lower=1e-9, upper=10)
group_phase_mu = pymc.CircVonMises("phase_mu", mu=0, kappa=1.0)
group_amp_alpha = pymc.Normal("amp_alpha", value=1.0, mu=1.0, tau=0.01)
group_amp_beta = pymc.Uniform("amp_beta", value=25, lower=1.0, upper=50.0)

# ANSWER:
#
# No. Remember, this is Bayesian analysis. We do not sample directly from these distributions.
# The sampler will decide how to move in the parameter space, devoting more time in areas
# of high density.
# These are the priors. They encode how we think these parameters are distributed (for the fly groups).
# But they could (should in an ideal world) be "overpowered" by they data if they are incorrect.
# In this case, these priors are quite non-committal (also called "flat", "uninformative"),
# meaning we originally assign a fair amount of credibility to a wide range of parameter values.
# We will reasign that credibility given the data we observe during the simulation.
# The posterior distribution (what we are looking for to know) might not look at
# all like these priors. For example, the phase kappa posterior might have 99% of its values on a
# narrow band around 5 if there is where the real value distribute has a mode, while here
# our prior is a unirmly distributed between 0 and 10.
# At the end of the simulation we will inspect the posterior values for each parameter
# (e.g. "phase_kappa") to see how these parameters seem to be distributed. Of course,
# in this case the group parameters are, by far, what interest us the most from the whole
# posterior distribution of theta.
#
# In fact, I think that these priors are only used for:
#   - sampling the initial value of the parameters in the simulation
#   - compute the prior probability of theta to weight against the likelihood
#


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
    # ANSWER: Again, these are the priors and can be overpowered.
    # Notice how the means priors are distributed after ("shrinked to") the group priors.
    # Also, note that there are other kind of models in thebas where the group variables
    # are modelled differently. How to model the group "hyper-priors" was one of the most
    # important questions we had fun answering.

    # XXX QUESTION 3:
    #  This is the ideal response of the fly. Sampling from phase and amplitude
    #  distributions given above?
    #
    @pymc.deterministic(plot=False, name='modeledSignal_' + flyid)
    def modeled_signal(amplitude=amplitude, phase=phase):
        return amplitude * np.sin(2*np.pi * perturbation_freq * time + phase)
    # ANSWER: Yes and no.
    # This is how we model the "mean" of the response (see question 4).
    # Note this is a pymc2 "deterministic": its values are fully determined by
    # its parents (in this case, the fly amplitude and phase values).
    # In other words, no random number generator is directly involved on each iteration
    # of the simulation when sampling its value, but the values that
    # the parents parameters take at that iteration are plugged directly into the equation.
    # Note that here there is a, somebody would call nasty, trick to include the time into the
    # equation: we use a closure where "time" is the times we read on top of the function.
    # They are fixed in our experiments, and while this python wonder is non very standard
    # practice in pymc, it gets the job done...

    # XXX QUESTION 4:
    #  y represents the likelyhood that given sampled parameters used in model_signal
    #  what is the chance, that signal is measured?
    #
    sigma = pymc.Uniform('sigma_' + flyid, 1e-9, 10.)
    # --- likelihood (each observation is modeled as a Gaussian, all share their sd)
    y = pymc.Normal('y_' + flyid, mu=modeled_signal, tau=1.0/sigma**2, value=signal, observed=True)
    # ANSWER: We model each fly's response as a Gaussian with a mean given
    # by the "ideal response" defined above; the parameters of the "ideal response"
    # will be the values sampled from the current iteration of the MCMC simulation.
    # Note that this is a multivariate Gaussian. Its dimensionality is equal
    # to the number of times we have sampled the signal (which in turn, in
    # Lisa's experimental setup, depends on the frequency of the stimulus; remember,
    # we fit one model per group and stimulus frequency).
    # As you see, its standard deviation prior is, again, very non-commital.
    # Note that since this is where we provide access to the data in the model,
    # we assign its "value" to our observations and mark it as "observed".
    #
    # In this line we are plugging in the data to compute its likelihood
    # given "the current sampled value of theta" and weight it by
    # the priors. You can imagine on each iteration the sampler computing
    # the pdf of the data given the current value of theta here.

    # --- fly model
    return [y, phase, amplitude, sigma]

# Collect all model parameters in model
model = [group_phase_mu, group_phase_kappa, group_amp_alpha, group_amp_beta]
for fly in flies:
    model += fly_model(fly)

# Sample the model
M = pymc.MCMC(model, db='pickle', dbname='blah.pickle', name='mytest')
M.sample(iter=80000, burn=20000, progress_bar=False)

#
# ANSWER: So, to repeat and summarize, on "the current iteration" of our simulation:
# Given the previously accepted "theta_last" and "p_last(last_theta|data)".
#
#   1- The sampler samples (proposes) a value for theta_current given theta_last.
#      It usually moves somehow in the parameter space, usually sampling from some
#      (normal) distribution centered at theta_last. Does not care about the priors here!
#
#     1.1- It starts by sampling in the higher level of our hierarchy (the group),
#      moves down in the tree keeping sampling on all non-deterministic parameters
#      given priors and hyperpriors.
#
#     1.2- Down the model tree we get to our deterministic variable,
#      where we plug "the ideal" (sine like) response model. There
#      we compute for all the times we observed the values of the sine
#      given the partial value of theta_current, and will use them as the mean
#      of our multivariate Gaussian which will compute the likelihood.
#
#     1.3- The mean of our multivariate Gaussian, at the very bottom of
#      our model tree, is set to the result in (1.2). Also a standard deviation
#      is randomly sampled given the last value and therefore
#      theta_current is now fully defined. We can now proceed to compute
#      the likelihood of the data given the current value of theta.
#
#   2- Compute the likelihood of the data given current theta.
#      This means computing the pdf of the data given the multivariate Gaussian at the
#      bottom of the model. Weight it by the prior of theta.
#          p_current = p(data | theta_current) * p(theta_current)
#
#   3- Accept current theta proposal?
#      We need to decide if we accept theta, therefore adding it to the trace
#      and using it to generate the new theta proposal in the next iteration.
#      For this the MCMC sampler will usually compute the probability of acceptance
#      like this:
#        p_acceptance = p_current / p_previous
#      We accept the new value of theta  with probability p_acceptance:
#        - If accepted, we will collect theta_current and use it when generating
#          a new proposal for theta in the next iteration.
#        - If not accepted, we will collect and keep using theta_last.
#
