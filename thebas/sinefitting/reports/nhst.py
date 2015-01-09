# coding=utf-8
import numpy as np
from itertools import product
from scipy.stats.mstats_basic import mannwhitneyu
from sklearn.metrics import roc_auc_score
from thebas.sinefitting import HS_PROJECT, DCN_PROJECT
from thebas.sinefitting.data import perturbation_data_to_records


def mann(pbproject=HS_PROJECT, summ=np.mean):
    """Runs NHST on summaries of wba, per frequency and control/blocked pairs."""
    df = perturbation_data_to_records(pbproject=pbproject)
    # compare each genotype's (mean|median|std...) wba per frequency
    by_freq = df[['genotype', 'freq', 'flyid', 'wba', 'wba_t']].groupby(('freq',))
    for freq, freq_data in by_freq:
        genotypes = sorted(freq_data['genotype'].unique())
        blocked = [genotype for genotype in genotypes if 'tnte' in genotype.lower()]
        control = [genotype for genotype in genotypes if 'tnte' not in genotype.lower()]
        for b, c in product(blocked, control):
            bdat = freq_data[freq_data['genotype'] == b]['wba'].apply(summ).dropna()
            cdat = freq_data[freq_data['genotype'] == c]['wba'].apply(summ).dropna()
            print 'freq=%g; %s vs %s' % (freq, b, c)
            print '\tThere are %d control flies and %d blocked flies' % (len(cdat), len(bdat))
            auc = roc_auc_score(([0] * len(cdat)) + ([1] * len(bdat)), np.hstack((cdat, bdat)))
            U, p = mannwhitneyu(cdat, bdat)
            print '\tAUC=%.2f (MWU p=%.6f)' % (auc, p)

if __name__ == '__main__':

    # Use the mean to summarize
    print 'summ=mean'
    mann(pbproject=HS_PROJECT, summ=np.mean)
    mann(pbproject=DCN_PROJECT, summ=np.mean)
    print '-' * 80

    # Use the median to summarize
    print 'summ=median'
    mann(pbproject=HS_PROJECT, summ=np.median)
    mann(pbproject=DCN_PROJECT, summ=np.median)
    print '-' * 80

    # Center and then max (trying to account for weird DCs / biases on some genotypes, look at meanwba plots)
    print 'summ=center-max'

    def center_max(x):
        if 0 == len(x):
            return np.nan
        return (x - x.mean()).max()

    mann(pbproject=HS_PROJECT, summ=center_max)
    mann(pbproject=DCN_PROJECT, summ=center_max)
    print '-' * 80

    # Center, remove outliers, max
    print 'summ=center-mad-max'

    def center_mad_max(x, zmad_threshold=3.5):
        if zmad_threshold > 0:
            mu = np.mean(x)
            mad = np.median(np.abs(x - mu))
            z = (x - mu) / mad
            is_outlier = z > zmad_threshold
            x = x[~is_outlier]
        if 0 == len(x):
            return np.nan
        return (x - x.mean()).max()
    mann(pbproject=HS_PROJECT, summ=center_mad_max)
    mann(pbproject=DCN_PROJECT, summ=center_mad_max)

#
# RESULTS:
# (look too at meanwba and co. plots)
#
# summ=mean
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.34 (MWU p=0.202449)
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.54 (MWU p=0.719614)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.31 (MWU p=0.131977)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.59 (MWU p=0.441756)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.31 (MWU p=0.131977)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.55 (MWU p=0.681618)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.31 (MWU p=0.117750)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.53 (MWU p=0.837472)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.48 (MWU p=0.862015)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.50 (MWU p=1.000000)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 12 blocked flies
# 	AUC=0.41 (MWU p=0.479084)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 12 blocked flies
# 	AUC=0.44 (MWU p=0.665006)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.49 (MWU p=0.947645)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 11 blocked flies
# 	AUC=0.50 (MWU p=0.975451)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.42 (MWU p=0.554530)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 11 blocked flies
# 	AUC=0.59 (MWU p=0.451345)
# freq=0.5; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.49 (MWU p=0.938503)
# freq=1; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.60 (MWU p=0.425315)
# freq=2; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.57 (MWU p=0.589154)
# freq=4; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.60 (MWU p=0.396066)
# freq=8; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.58 (MWU p=0.487453)
# freq=16; VT37804_TNTE vs VT37804_TNTin
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.53 (MWU p=0.816735)
# freq=32; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 13 blocked flies
# 	AUC=0.46 (MWU p=0.764818)
# freq=40; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.47 (MWU p=0.816961)
# --------------------------------------------------------------------------------
# summ=median
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.29 (MWU p=0.082194)
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.52 (MWU p=0.877731)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.28 (MWU p=0.072489)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.62 (MWU p=0.305061)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.33 (MWU p=0.164384)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.41 (MWU p=0.472789)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.30 (MWU p=0.104756)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.45 (MWU p=0.681618)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.45 (MWU p=0.728126)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.49 (MWU p=0.918309)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 12 blocked flies
# 	AUC=0.45 (MWU p=0.689122)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 12 blocked flies
# 	AUC=0.47 (MWU p=0.839860)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.46 (MWU p=0.792813)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 11 blocked flies
# 	AUC=0.49 (MWU p=0.975451)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.46 (MWU p=0.792813)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 11 blocked flies
# 	AUC=0.57 (MWU p=0.562343)
# freq=0.5; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.48 (MWU p=0.857136)
# freq=1; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.58 (MWU p=0.487453)
# freq=2; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.58 (MWU p=0.487453)
# freq=4; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.64 (MWU p=0.226774)
# freq=8; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.60 (MWU p=0.396066)
# freq=16; VT37804_TNTE vs VT37804_TNTin
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.52 (MWU p=0.907753)
# freq=32; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 13 blocked flies
# 	AUC=0.45 (MWU p=0.683313)
# freq=40; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.45 (MWU p=0.661972)
# --------------------------------------------------------------------------------
# summ=center-max
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.44 (MWU p=0.643011)
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.28 (MWU p=0.064870)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.53 (MWU p=0.816735)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.43 (MWU p=0.538301)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.56 (MWU p=0.643011)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.19 (MWU p=0.007661)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.73 (MWU p=0.063744)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.40 (MWU p=0.411924)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.52 (MWU p=0.862015)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.41 (MWU p=0.441756)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 12 blocked flies
# 	AUC=0.58 (MWU p=0.558760)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 12 blocked flies
# 	AUC=0.36 (MWU p=0.260236)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.61 (MWU p=0.393302)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 11 blocked flies
# 	AUC=0.45 (MWU p=0.689122)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.54 (MWU p=0.792813)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 11 blocked flies
# 	AUC=0.38 (MWU p=0.353934)
# freq=0.5; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.60 (MWU p=0.396066)
# freq=1; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.83 (MWU p=0.004309)
# freq=2; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.71 (MWU p=0.075982)
# freq=4; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.58 (MWU p=0.520269)
# freq=8; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.63 (MWU p=0.268795)
# freq=16; VT37804_TNTE vs VT37804_TNTin
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.66 (MWU p=0.202449)
# freq=32; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 13 blocked flies
# 	AUC=0.70 (MWU p=0.097120)
# freq=40; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.68 (MWU p=0.129187)
# --------------------------------------------------------------------------------
# summ=center-mad-max
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.39 (MWU p=0.384821)
# freq=0.5; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.17 (MWU p=0.004795)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.54 (MWU p=0.772059)
# freq=1; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.35 (MWU p=0.199825)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.58 (MWU p=0.523928)
# freq=2; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.17 (MWU p=0.004081)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.71 (MWU p=0.092926)
# freq=4; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.34 (MWU p=0.182422)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.59 (MWU p=0.486906)
# freq=8; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 13 blocked flies
# 	AUC=0.40 (MWU p=0.383320)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 12 blocked flies
# 	AUC=0.56 (MWU p=0.644373)
# freq=16; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 12 blocked flies
# 	AUC=0.39 (MWU p=0.370844)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.61 (MWU p=0.393302)
# freq=32; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 12 control flies and 11 blocked flies
# 	AUC=0.45 (MWU p=0.689122)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_KIR
# 	There are 11 control flies and 11 blocked flies
# 	AUC=0.52 (MWU p=0.895514)
# freq=40; VT58487_tshirtgal80_TNTE vs VT58487_tshirtgal80_TNTin
# 	There are 13 control flies and 11 blocked flies
# 	AUC=0.36 (MWU p=0.246566)
# freq=0.5; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.72 (MWU p=0.060469)
# freq=1; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.75 (MWU p=0.032799)
# freq=2; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.67 (MWU p=0.157231)
# freq=4; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.57 (MWU p=0.589154)
# freq=8; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.61 (MWU p=0.368066)
# freq=16; VT37804_TNTE vs VT37804_TNTin
# 	There are 11 control flies and 13 blocked flies
# 	AUC=0.64 (MWU p=0.270986)
# freq=32; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 13 blocked flies
# 	AUC=0.70 (MWU p=0.097120)
# freq=40; VT37804_TNTE vs VT37804_TNTin
# 	There are 12 control flies and 14 blocked flies
# 	AUC=0.65 (MWU p=0.207617)
#
