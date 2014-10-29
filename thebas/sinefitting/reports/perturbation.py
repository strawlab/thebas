# coding=utf-8
"""Building reports out of the initial data and the MCMC results for the sinewave perturbation data experiment."""

import numpy as np
import matplotlib.pyplot as plt
from thebas.sinefitting import HS_PROJECT, DCN_PROJECT

from thebas.sinefitting.data import all_biases


# ----- Plots


def all_biases_plot(pbproject=HS_PROJECT, transpose=True):
    # Original bias had amplitude 5
    from matplotlib import rc, rcParams
    rc('font', family='serif', size=16)
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\boldmath']
    bias, bias_t = all_biases(pbproject).values()[0]
    bias_t -= bias_t[0]
    if not transpose:
        plt.plot(bias_t, bias, color='g')
        plt.ylabel('bias (rad/s)', fontsize=20)
        plt.xlabel('time (s)', fontsize=20)
        plt.xlim((-5, 205))
        plt.axhline(linewidth=4, color='g')
    else:
        plt.plot(bias, bias_t, color='g')
        plt.xlabel('bias (rad/s)', fontsize=20)
        plt.ylabel('time (s)', fontsize=20)
        plt.ylim((-5, 205))
        plt.axvline(linewidth=4, color='g')
    plt.title('bias velocity over time')
    plt.show()
    # Plot the bias magnitude of change in time...
    # dbias_t = bias_t[0:len(bias_t)-1] - bias_t[0]
    # dbias = bias[1:] - bias[0:len(bias)-1]
    dbias_t = bias_t
    dbias = np.gradient(bias)
    if not transpose:
        plt.plot(dbias_t, dbias, color='g')
        plt.axhline(linewidth=4, color='g')
        plt.ylabel('bias ($rad/s^2$)', fontsize=20)
        plt.xlabel('time (s)', fontsize=20)
        plt.xlim((-5, 205))
    else:
        plt.plot(dbias, dbias_t, color='g')
        plt.axvline(linewidth=4, color='g')
        plt.xlabel('bias ($rad/s^2$)', fontsize=20)
        plt.ylabel('time (s)', fontsize=20)
        plt.ylim((-5, 205))
    plt.title('bias acceleration over time')
    plt.show()


def data_sizes_plot(freqs=None):
    from matplotlib import rc, rcParams
    rc('font', family='serif', size=16)
    rc('text', usetex=True)
    rcParams['text.latex.preamble'] = [r'\boldmath']
    if freqs is None:
        freqs = np.array((0.5, 1, 2, 4, 8, 16, 32, 40))
    seconds = 2 * 8 * np.pi / freqs
    num_obs = seconds * 100
    width = .35
    ind = np.arange(len(freqs))
    plt.bar(ind, num_obs, width=0.35)
    plt.xticks(ind + width / 2, freqs)
    plt.ylabel('number of observations', fontsize=20)
    plt.xlabel('$\omega$ (rad/s)', fontsize=20)
    plt.title('Higher frequencies get less observations')
    plt.show()
    return num_obs

if __name__ == '__main__':
    all_biases_plot(DCN_PROJECT)  # same for HS
    data_sizes_plot()
