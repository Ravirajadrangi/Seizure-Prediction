# --------------------------------------------------------
#                               PYTHON LIBRARIES
# --------------------------------------------------------
import scipy.io
from sklearn.linear_model import LogisticRegression, Ridge
import scipy.io
import scipy.io
from scipy.fftpack import rfft
from scipy.signal import correlate, resample, welch
import pysptk
import scipy.io.wavfile
import warnings
warnings.filterwarnings('ignore')

# numerical processing and scientific libraries
import numpy as np
import scipy

import os,sys,inspect
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search
import math
from  platform import system

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split

import pandas
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score, log_loss, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split
import pandas
from scipy.stats import norm, invgamma
from sklearn import linear_model
import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
from sklearn.preprocessing import normalize

import spectrum

import numpy as np
import scipy.ndimage as ndimage

from numpy import unwrap, angle
from scipy.signal import hilbert

# import scipy
from datetime import datetime

import scipy.stats as st

from IeegConsts import *

from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------
#                FUNCTION DECLARATION
# --------------------------------------------------------

n_16 = 16
n_fft = 256
n_welch = 9
n_psd = 176 +16
# n_psd = 64
n_wave = 64
n_mfcc = 12
n_pearson = 16

ar_elements = 16
n_AR = 16 * ar_elements

n_240 = 240000
n_interval = n_240 / 600
n_p_corr=n_240/600

n_corr_coeff=120
n_plv=120
n_eigen=16

class IeegFeatures(object):
    """
        Class to generate features for the patients
        @author Solomomk
    """
    def __init__(self, baseDir, isTrain=True):
        now = datetime.now()
        print 'Starting:' + 'ieegFeatures:' + str(now)
        self.isTrain=isTrain
        self.baseDir=baseDir
        # self.totalCols=len(self.ieegGenCols())

    def max_slope(self, t, x):
        """Compute the largest rate of change in the observed data."""
        slopes = np.diff(x) / np.diff(t)
        return np.max(np.abs(slopes))

    def PLV(self, x):
        data = x
        n_ch, time = data.shape
        n_pairs = n_ch * (n_ch - 1) / 2
        # initiate matrices
        phases = np.zeros((n_ch, time))
        delta_phase_pairwise = np.zeros((n_pairs, time))
        plv = np.zeros((n_pairs,))

        # extract phases for each channel
        for c in range(n_ch):
            phases[c, :] = unwrap(angle(hilbert(data[c, :])))

        # compute phase differences
        k = 0
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                delta_phase_pairwise[k, :] = phases[i, :] - phases[j, :]
                k += 1

        # compute PLV
        for k in range(n_pairs):
            plv[k] = np.abs(np.sum(np.exp(1j * delta_phase_pairwise[k, :])) / time)

        # features = a 1d ndarray
        return plv

    def featureFFT(self,y, fs=400):
        """ Get the FFT of a given signal and corresponding frequency bins.
        Parameters:
            y  - signal
            fs - sampling frequency
        Returns:
            (mag, freq) - tuple of spectrum magitude and corresponding frequencies
        """
        n  = len(y)      # Get the signal length
        dt = 1/float(fs) # Get time resolution

        fft_output = np.fft.rfft(y)     # Perform real fft
        rfreqs = np.fft.rfftfreq(n, dt) # Calculatel frequency bins
        fft_mag = np.abs(fft_output)    # Take only magnitude of spectrum

        # Normalize the amplitude by number of bins and multiply by 2
        # because we removed second half of spectrum above the Nyqist frequency
        # and energy must be preserved
        fft_mag = fft_mag * 2 / n

        return np.array(fft_mag), np.array(rfreqs)

    def phase_correlation(self, a, b):
        G_a = np.fft.fft2(a)
        G_b = np.fft.fft2(b)
        conj_b = np.ma.conjugate(G_b)
        R = G_a * conj_b
        R /= np.absolute(R)
        r = np.fft.ifft2(R).real
        return r

    def blur(self, x):
        # h = scipy.signal.gaussian(50, 10)  # gaussian filter to get rid of electrical artefacts
        # ieegData = scipy.signal.convolve(ieegData, h, mode='full')  # filter
        img = ndimage.gaussian_filter(x, sigma=(50, 50), order=0)
        return img

    def remove_dc(self, x):
        # print x.shape
        assert (type(x) == np.ndarray)
        """
        Remove mean of signal
        :return: x - mean(x)
        """
        x_dc = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_dc[i, :] = x[i, :] - np.mean(x[i, :])
        return x_dc


    def to_np_array(self,X):
        if isinstance(X[0], np.ndarray):
            # return np.vstack(X)
            out = np.empty([len(X)] + list(X[0].shape), dtype=X[0].dtype)
            for i, x in enumerate(X):
                out[i] = x
            return out
        return np.array(X)


    def upper_right_triangle(self, matrix):
        accum = []
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                accum.append(matrix[i, j])

        return self.to_np_array(accum)

    def hurst(self,x):
        x -= x.mean()
        z = np.cumsum(x)
        r = (np.maximum.accumulate(z) - np.minimum.accumulate(z))[1:]
        s = pandas.expanding_std(x)[1:]

        # prevent division by 0
        s[np.where(s == 0)] = 1e-12
        r += 1e-12

        y_axis = np.log(r / s)
        x_axis = np.log(np.arange(1, len(y_axis) + 1))
        x_axis = np.vstack([x_axis, np.ones(len(x_axis))]).T

        m, b = np.linalg.lstsq(x_axis, y_axis)[0]
        return m


    def pfd_for_ch(self, ch):
        diff = np.diff(ch, n=1, axis=1)

        asign = np.sign(diff)
        sign_changes = ((np.roll(asign, 1) - asign) != 0).astype(int)
        N_delta = np.count_nonzero(sign_changes)

        n = len(ch)
        log10n = np.log10(n)
        return log10n / (log10n + np.log10(n / (n + 0.4 * N_delta)))

    def varAR(self,x):
        params = VAR(x).fit(maxlags=2).params
        features = np.hstack(params.reshape((np.prod(params.shape), 1)))
        return features

        # --------------------------------------------------------
        #  # ieegData is a 16x240000 matrix => 16 channels reading data for 10 minutes at 400Hz
        # --------------------------------------------------------
    def getAllFeatures(self, ieegData):
        ieegData=self.remove_dc(ieegData)
        # ieegData=self.blur(ieegData)
        ieegT=ieegData.transpose()
        ieegRT = resample(ieegT, 600, axis=1, window=400)

        final_row = None
        x_mean = np.mean(ieegData, axis=0)
        x_median = np.median(ieegData, axis=0)
        x_std = np.std(ieegData, axis=0)
        x_skew = scipy.stats.skew(ieegData, axis=0)
        x_kurt = scipy.stats.kurtosis(ieegData, axis=0)
        x_var = np.var(ieegData, axis=0)

        x_m12=normalize((scipy.stats.moment(ieegData, moment=12, axis=0)),norm='l2')
        x_m12=x_m12.ravel()

        x_m4=normalize((scipy.stats.moment(ieegData, moment=4, axis=0)),norm='l2')
        x_m4 = x_m4.ravel()

        x_psd=self.featurePSDAdvanced(ieegData,400)
        x_AR = np.concatenate([self.fetureAR(ch) for ch in ieegRT], axis=0)

        #Time correlation matrix upper right triangle and sorted eigenvalues
        x_corr_coeef=np.corrcoef(ieegT)
        x_corr_coeef=self.upper_right_triangle(x_corr_coeef)

        x_hurst= np.apply_along_axis(self.hurst, -1, ieegRT)

        x_plv=(self.PLV(ieegT))

        x=ieegData
        correlations = None
        for j in range(n_interval):
            y1 = x[j * n_interval: (j + 1) * n_interval]
            y2 = x[(j + 1) * n_interval: (j + 2) * n_interval]
            p_c = self.phase_correlation(y1, y2)
            if correlations is None:
                correlations = np.max(p_c)
            else:
                correlations = np.vstack([correlations, np.max(p_c)])
        # x_var_ar=self.varAR(ieegT)
        correlations=correlations.ravel()
        final_row = np.concatenate([x_mean, x_median, x_std, x_skew, x_kurt, x_var,x_m12,x_m4, x_psd, x_AR, x_corr_coeef, x_hurst,x_plv, correlations])
        # print 'final_row:' + str(len(final_row))
        Inan = np.where(np.isnan(final_row))
        Iinf = np.where(np.isinf(final_row))
        final_row[Inan] = 0
        final_row[Iinf] = 0

        final_row = np.nan_to_num(final_row)
        # print str(final_row.shape)
        return final_row

    def getEigen(self, ieegData):
        w, v = np.linalg.eig(ieegData)
        w = np.absolute(w)
        w.sort()
        return w

    def ieegGenCols(self):
        cols = list()
        n=n_16
        cols.append('file')
        cols.append('id')
        cols.append('patient_id')
        if self.isTrain:
            cols.append('sequence_id')
            cols.append('file_size')

        # [x_mean, x_median, x_std, x_skew, x_kurt, x_var, x_m6, x_m4, x_psd, x_AR, x_corr_coeef, x_hurst, x_plv])

        # cols.append('sequence')
        for i in range(1, n + 1):
            cols.append('mean_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('median_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('std_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('skew_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('kurt_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('var_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('m6_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('m4_{}'.format(i))
        for i in range(1, n_psd + 1):
            cols.append('psd_{}'.format(i))
        for i in range(1, n_AR + 1):
            cols.append('AR_{}'.format(i))
        for i in range(1, n_corr_coeff + 1):
            cols.append('corcoef_{}'.format(i))
        for i in range(1, n + 1):
            cols.append('hurst_{}'.format(i))
            # -------Response allways last-----#
        for i in range(1,  n_plv+ 1):
            cols.append('plv_{}'.format(i))

        for i in range(1, n_p_corr + 1):
            cols.append('cpc_{}'.format(i))

        if self.isTrain:
            cols.append('segment')
            cols.append(singleResponseVariable)

        print "Cols:" + str(len(cols))
        return cols



    # --------------------------------------------------------
    #  # ieegData is a 16x240000 matrix => 16 channels reading data for 10 minutes at 400Hz
    # --------------------------------------------------------
    def getAllFeaturesOLDDDDDDDDDDD(self, ieegData):
        ieegDataResamTrans=ieegData

        # First transpose, then resample
        # Resamble each channel to get only a meassurement per second
        # 10 minutes of measurements, grouping data on each second
        # ieegData = resample(ieegData, 600, axis=1, window=400)
        # ieegDataTrans=ieegData.transpose()
        # ieegDataResamTrans = resample(ieegDataTrans, 600, axis=1, window=400)

        # np.fft.rfft(data, axis=axis)
        # x_fft = rfft(ieegData, n=n_16, axis=1)
        # x_fft = x_fft.ravel()  # 256 elements
        # temp = np.partition(-x_fft, n_fft)
        # x_fft = -temp[:n_fft]

        # img_freq = np.fft.fft2(ieegT)
        # x_amp = np.fft.fftshift(np.abs(img_freq))
        # x_mean_fft = np.mean(x_amp, axis=1)
        # x_mean_fft=x_mean_fft[0:8]
        # print 'fft mean:' + str(x_mean_fft.shape)
        # x = ieegData
        # correlations = None
        # for j in range(n_interval):
        #     y1 = x[j * n_interval: (j + 1) * n_interval]
        #     y2 = x[(j + 1) * n_interval: (j + 2) * n_interval]
        #     p_c = self.phase_correlation(y1, y2)
        #     if correlations is None:
        #         correlations = np.max(p_c)
        #     else:
        #         correlations = np.vstack([correlations, np.max(p_c)])
        #
        # correlations = [correlations]
        # final_row = np.concatenate([x_mean, x_median, x_std, x_skew, x_kurt, x_psd, x_AR, x_fft, x_corr_coeef, x_eigen_corr, correlations])

        x_mean = np.mean(ieegDataResamTrans, axis=0)
        # print 'x_mean:' + str(x_mean.shape)
        x_median = np.median(ieegDataResamTrans, axis=0)
        x_std = np.std(ieegDataResamTrans, axis=0)
        # hstack will collapse all entries into one big vector
        # print 'x_std16_cols:' + str(x_std.shape)

        # x_fft = rfft(ieegDataResamTrans, n=n_16, axis=1)[:1024]
        # x_fft = x_fft.ravel()  # 256 elements
        # temp = np.partition(-x_fft, 32)
        # x_fft = -temp[:32]
        # # print 'fft:' + str(x_fft.shape)
        #
        # img_freq = np.fft.fft2(ieegDataResamTrans)
        # x_amp = np.fft.fftshift(np.abs(img_freq))
        # x_mean_fft = np.mean(x_amp, axis=1)
        # # print 'fft mean:' + str(x_mean_fft.shape)
        #
        # x_mfcc=self.fetureMFCC(ieegDataResamTrans.ravel())
        # print 'mfcc:' + str(x_mfcc.shape)

        # x = ieegData

        # correlations = None
        # for j in range(n_interval):
        #     # print j
        #     y1 = x[j * n_interval: (j + 1) * n_interval]
        #     y2 = x[(j + 1) * n_interval: (j + 2) * n_interval]
        #     # print y1
        #     # print y2
        #     p_c = self.phase_correlation(y1, y2)
        #     if correlations is None:
        #         correlations = np.max(p_c)
        #     else:
        #         correlations = np.vstack([correlations, np.max(p_c)])

        # correlations = [correlations.ravel]
        # print str(type(correlations))
        # print str((correlations))

        # # correlations = None
        # correlations = None
        # for i in range(60):  # 10 seconds interval
        #     correlations_i = np.array([])
        #     for j in range(16):
        #         if i != j:
        #             corr_i = self.phase_correlation(ieegRT[i], ieegRT[j])
        #             # corr_i = correlate(ieegRT[i], ieegRT[j], mode='full')
        #             correlations_i = np.concatenate([correlations_i, corr_i])
        #             print str(len(correlations_i))
        #     if correlations is None:
        #         correlations = correlations_i
        #     else:
        #         correlations = np.vstack([correlations, correlations_i])
        #
        # print str(type(correlations))
        # print 'corr:' + (str(len(correlations.ravel())))
        # print 'corr 2:' + (str(len(correlations)))



        # x_hurst= np.apply_along_axis(self.fetureHURST, -1, ieegDataResamTrans)
        # print 'hurst:' + str(x_hurst.shape) # 24000
        # x_AR=np.concatenate([fetureAR(ch) for ch in ieegDataResamTrans], axis=0)
        # print 'AR:' + str(x_AR.shape)

        # final_row=np.concatenate([x_mean,x_median,x_std, x_fft, x_mean_fft, x_mfcc, x_hurst])
        final_row=np.concatenate([x_mean,x_median,x_std])
        return final_row


    def ieegProcessAllFilesAsDF(self, files):
        now = datetime.now()
        print 'Starting:' + 'ieegProcessAllFilesAsDF:' + str(now)
        X = None
        cols = list()
        cols = self.ieegGenCols()
        print 'cols for X:' + str(len(cols))
        print 'cols for X:' + str((cols))
        totalCols = len(cols)
        X = np.vstack([cols])

        total_files = len(files)
        print 'Files:' + str(total_files)

        for i, filename in enumerate(files):
            # if i % int(total_files / 10) == 0:
            #     print(u'%{}: Loading file {}'.format(int(i * 100 / total_files), filename))

            id_str = os.path.basename(filename)[:-4]
            arr = id_str.split("_")
            patient = int(arr[0])
            p_id = int(arr[1])
            new_id = patient * 100000 + p_id
            filename_arr = np.array([filename])

            file_size=int(os.path.getsize(''.join([self.baseDir, filename.decode('UTF-8')])))
            if system()=='Linux':
                mat_data = scipy.io.loadmat(''.join([self.baseDir, filename.decode('UTF-8')]), verify_compressed_data_integrity=False)
            else:
                mat_data = scipy.io.loadmat(''.join([self.baseDir, filename.decode('UTF-8')]))

            ieegData = mat_data['dataStruct'][0][0][0]
            seq_id= int ((scipy.math.floor((new_id - patient * 100000) / 6)) +1)
            segment=int((scipy.math.floor((new_id - patient * 100000))))
            # x_MOD['segment']=X_df_train.apply(lambda row: int ((scipy.math.floor((row['id'] - row['patient_id'] * 100000)))), axis=1)
            # x_MOD['sequence_id']=x_MOD.apply(lambda row: int ((scipy.math.floor((row['id'] - row['patient_id'] * 100000) / 6)) +1), axis=1)
            # x_MOD.head(5)
            # x_MOD.to_csv(F_NAME_TRAIN +'.csv' , sep=',')

            # print str(x_all.shape)

            if self.isTrain:
                print 'Train iteration:' ',' + str(i) + '... ' + filename
                if file_size > 55000:
                    x_all = self.getAllFeatures(ieegData)
                    response = np.array([self.ieegClassByFileName(filename)], float)
                    final_line = np.concatenate([filename_arr, [new_id], [patient], [seq_id], [file_size], x_all,[segment],response])
                    # print str(len(final_line))
                    if X is not None:
                        X = np.vstack([X, final_line])
                    else:
                        X = np.vstack([final_line])
                else:
                    print 'Skip bad train file:' + str(filename_arr)
                    # print str(file_size)
                    # continue
            else:
                print 'Test iteration:' ',' + str(i) + '... ' + filename
                if file_size > 55000:
                    x_all = self.getAllFeatures(ieegData)
                    # print str(len(x_all))
                    final_line = np.concatenate([filename_arr, [new_id], [patient], x_all])
                    # print str(len(final_line))
                else:
                    print 'Bad test file:' + str(filename_arr)
                    x_all =[0] * (totalCols -3)
                    print str(len(x_all))
                    final_line = np.concatenate([filename_arr, [new_id], [patient], x_all])

                if X is not None:
                    X = np.vstack([X, final_line])
                else:
                    X = np.vstack([final_line])

        return X


    # ---------------------------------------------------------------#

    # --------------------------------------------------------
    #       FEATURE COL NAMES
    # --------------------------------------------------------

    def fetureFFT(self,x):
        N = len(x)
        F = np.fft.fft(x, n=2 * N)  # 2*N because of zero-padding
        PSD = F * F.conjugate()
        res = np.fft.ifft(PSD)
        res= (res[:N]).real  # now we have the autocorrelation in convention B
        n = N * np.ones(N) - np.arange(0, N) #divide res(m) by (N-m)
        return res / n  # this is the autocorrelation in convention A
    # ---------------------------------------------------------------#

    # --------------------------------------------------------
    #
    # --------------------------------------------------------
    def fetureENTROPY(self,signal):
        signal=signal.ravel
        lensig = signal.size
        symset = list(set(signal))
        numsym = len(symset)
        propab = [np.size(signal[signal == i]) / (1.0 * lensig) for i in symset]
        ent = np.sum([p * np.log2(1.0 / p) for p in propab])
        return ent

    # --------------------------------------------------------
    #
    # --------------------------------------------------------
    def featurePSD(self,eegdata, Fs):
        # 1. Compute the PSD
        winSampleLength, nbCh = eegdata.shape

        # Apply Hamming window
        w = np.hamming(winSampleLength)
        dataWinCentered = eegdata - np.mean(eegdata, axis=0) # Remove offset
        dataWinCenteredHam = (dataWinCentered.T*w).T

        NFFT = self.nextpow2(winSampleLength)
        Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0)/winSampleLength
        PSD = 2*np.abs(Y[0:NFFT/2,:])
        f = Fs/2*np.linspace(0,1,NFFT/2)

        # SPECTRAL FEATURES
        # Average of band powers
        # Delta <4
        ind_delta, = np.where(f<4)
        meanDelta = np.mean(PSD[ind_delta,:],axis=0)
        # Theta 4-8
        ind_theta, = np.where((f>=4) & (f<=8))
        meanTheta = np.mean(PSD[ind_theta,:],axis=0)
        # Alpha 8-12
        ind_alpha, = np.where((f>=8) & (f<=12))
        meanAlpha = np.mean(PSD[ind_alpha,:],axis=0)
        # Beta 12-30
        ind_beta, = np.where((f>=12) & (f<30))
        meanBeta = np.mean(PSD[ind_beta,:],axis=0)
        feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha, meanBeta),axis=0)
        feature_vector = np.log10(feature_vector)
        return feature_vector
    # ---------------------------------------------------------------#

    def featurePSDAdvanced(self, eegdata, Fs):
        """Extract the features from the EEG
            Inputs:
              eegdata: array of dimension [number of samples, number of channels]
              Fs: sampling frequency of eegdata
            Outputs:
              feature_vector: [number of features points; number of different features
        """
        # Delete last column (Status)
        # eegdata = np.delete(eegdata, -1, 1)

        # 1. Compute the PSD
        winSampleLength, nbCh = eegdata.shape

        # Apply Hamming window
        w = np.hamming(winSampleLength)
        dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
        dataWinCenteredHam = (dataWinCentered.T * w).T

        NFFT = self.nextpow2(winSampleLength)
        Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
        PSD = 2 * np.abs(Y[0:NFFT / 2, :])
        f = Fs / 2 * np.linspace(0, 1, NFFT / 2)

        # SPECTRAL FEATURES
        # Average of band powers
        # Delta <4
        ind_delta, = np.where(f < 4)
        meanDelta = np.mean(PSD[ind_delta, :], axis=0)
        # Theta 4-8
        ind_theta, = np.where((f >= 4) & (f <= 8))
        meanTheta = np.mean(PSD[ind_theta, :], axis=0)
        # Low alpha 8-10
        ind_alpha, = np.where((f >= 8) & (f <= 10))
        meanLowAlpha = np.mean(PSD[ind_alpha, :], axis=0)
        # Medium alpha
        ind_alpha, = np.where((f >= 9) & (f <= 11))
        meanMedAlpha = np.mean(PSD[ind_alpha, :], axis=0)
        # High alpha 10-12
        ind_alpha, = np.where((f >= 10) & (f <= 12))
        meanHighAlpha = np.mean(PSD[ind_alpha, :], axis=0)
        # Low beta 12-21
        ind_beta, = np.where((f >= 12) & (f <= 21))
        meanLowBeta = np.mean(PSD[ind_beta, :], axis=0)
        # High beta 21-30
        ind_beta, = np.where((f >= 21) & (f <= 30))
        meanHighBeta = np.mean(PSD[ind_beta, :], axis=0)
        # Alpha 8 - 12
        ind_alpha, = np.where((f >= 8) & (f <= 12))
        meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
        # Beta 12-30
        ind_beta, = np.where((f >= 12) & (f <= 30))
        meanBeta = np.mean(PSD[ind_beta, :], axis=0)

        ind_r, = np.where((f >= 30) & (f <= 38))
        meanR = np.mean(PSD[ind_r, :], axis=0)


        feature_vector = np.concatenate((meanDelta, meanTheta, meanLowAlpha, meanMedAlpha, meanHighAlpha,
                                         meanLowBeta, meanHighBeta, meanR,
                                         meanDelta / meanBeta, meanTheta / meanBeta,
                                         meanAlpha / meanBeta, meanAlpha / meanTheta), axis=0)

        feature_vector = np.log10(feature_vector)
        # print str(len(feature_vector))
        return feature_vector

    # ---------------------------------------------------------------#
    import spectrum

    def fetureAR(self,ch):
        ar_coeffs, dnr, reflection_coeffs = spectrum.aryule(ch,ar_elements)
        return np.abs(ar_coeffs)
    #return np.concatenate([self.calc_for_ch(ch) for ch in X], axis=0)
    # ---------------------------------------------------------------#


    def nextpow2(self,i):
        n = 1
        while n < i:
            n *= 2
        return n


    # --------------------------------------------------------
    #         INTERICTAL = 0, PREICTAL = 1
    # --------------------------------------------------------
    def ieegClassByFileName(self,name):
        try:
            return float(name[-5])
        except:
            return 0.0


    # --------------------------------------------------------
    #       POPULATE A LIST WITH ALL THE NAMES OF FILES
    # --------------------------------------------------------
    def ieegAllFilesList(self):
        # ignored_files = ['.DS_Store']
        # ignored_files = BAD_FILES_TRAIN_LIST + BAD_FILES_TEST_LIST

        return np.array(
            [
                (file)
                # for file in os.listdir(self.baseDir) if file not in ignored_files
                for file in os.listdir(self.baseDir)
                ],dtype=[('file', '|S16')]
        )


    # --------------------------------------------------------
    #       MATLAB FILE TO ND ARRAY
    # --------------------------------------------------------
    def ieegMatToArray(self, path):
        mat = scipy.io.loadmat(path)
        names = mat['dataStruct'].dtype.names
        ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
        return ndata['data']


    # --------------------------------------------------------
    #       PATIENT ID
    # --------------------------------------------------------
    def getIdFromFileName(self,id_str):
        arr = id_str.split("_")
        #     print arr
        patient = int(arr[0])
        #     print patient
        p_id_str = str(arr[1])
        #     print p_id_str
        p_id = int((p_id_str)[:-4])
        #     print p_id
        new_id = [patient * 100000 + p_id]
        return new_id


    def kFoldCV(self,trainX, trainY, cv=10, test_size=.20):
        print 'Running k-Fold CV=' + str(cv)
        clf = LogisticRegression()

        # encoder = OneHotEncoder(handle_unknown='ignore').fit(trainX)
        # trainXEncoded = encoder.transform(trainX)  # Returns a sparse matrix (see numpy.sparse)

        scores = cross_val_score(clf, trainX, trainY, scoring='accuracy', cv=cv)
        print scores
        print scores.mean()

        SEED = 42

        mean_auc = 0.0
        n = 10  # repeat the CV procedure 10 times to get more precise results
        for i in range(n):
            # for each iteration, randomly hold out 20% of the data as CV set
            X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                trainX, trainY, test_size=test_size, random_state=i * SEED)

            # train model and make predictions
            clf.fit(X_train, y_train)
            preds = clf.predict_proba(X_cv)[:, 1]

            # compute AUC metric for this CV fold
            fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
            roc_auc = metrics.auc(fpr, tpr)
            print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
            mean_auc += roc_auc
        print "Mean AUC: %f" % (mean_auc / n)





    # #------------------------------------------------------------------------------#
    def singleAlgoAUC(self, X_df_train, algo, t_size=0.11, r_state=14, last_cols=None,glm_factor=None):
        # print 'Running:' + str(algo) + 'shape:' + str(X_df_train.shape)
        X_df_train_SINGLE = X_df_train.copy(deep=True)
        answers_1_SINGLE = list(X_df_train_SINGLE[singleResponseVariable].values)
        X_df_train_SINGLE = X_df_train_SINGLE.drop(singleResponseVariable, axis=1)

        if last_cols is not None:
            print 'Limit on last_cols'
            X_df_train_SINGLE = X_df_train_SINGLE[last_cols]

        if glm_factor is not None:
            print 'Limit on glm_factor'
            X_df_train_SINGLE = X_df_train_SINGLE[glm_factor]

        X_df_train_SINGLE = X_df_train_SINGLE.apply(lambda x: pandas.to_numeric(x, errors='ignore'))

        trainX, testX, trainY, testY = train_test_split(X_df_train_SINGLE, answers_1_SINGLE, test_size=t_size, random_state=r_state)  # CV
        clf_train = algo
        model_train = clf_train.fit(trainX, trainY)
        print model_train
        predictions = clf_train.predict_proba(testX)[:, 1]

        print 'ROC AUC:' + str(roc_auc_score(testY, predictions))
        print 'LOG LOSS:' + str(log_loss(testY, predictions))
        false_positive_rate, true_positive_rate, thresholds = roc_curve(testY, predictions)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        return trainX, testX, trainY, testY, clf_train, roc_auc
        # ------------------------------------------------------------------------------#

    # def singlePatientLRModel(self, X_df_train, glm_factor=None, withGrid=True):
    #     X_df_train_SINGLE = X_df_train.copy(deep=True)
    #     answers_1_SINGLE = list(X_df_train_SINGLE[TARGET_VAR].values)
    #     X_df_train_SINGLE = X_df_train_SINGLE.drop(TARGET_VAR, axis=1)
    #
    #     if glm_factor is not None:
    #         print 'Limit on the number of features'
    #         X_df_train_SINGLE = X_df_train_SINGLE[glm_factor]
    #     else:
    #         print 'No limit on the number of features'
    #     if withGrid is True:
    #         print 'Using grid'
    #         # Select only best features from previous section
    #         if glm_factor is not None:
    #             print 'Limit on the number of features'
    #             lr_best_params=self.gridLR(X_df_train, glm_factor)
    #         else:
    #             print 'No limit on the number of features'
    #             lr_best_params=self.gridLR(X_df_train)
    #
    #         algo_1= LogisticRegression(**lr_best_params)
    #     else:
    #         print 'Not using grid'
    #         algo_1 = LogisticRegression()
    #
    #     X_df_train_SINGLE = X_df_train_SINGLE.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
    #     print 'Running:' + str(algo_1) + 'shape:' + str(X_df_train_SINGLE.shape)
    #     trainX, testX, trainY, testY = train_test_split(X_df_train_SINGLE, answers_1_SINGLE, test_size=.22)  # CV
    #     model_train = algo_1.fit(trainX, trainY)
    #     print model_train
    #     predictions = algo_1.predict_proba(testX)[:, 1]
    #     print 'ROC AUC:' + str(roc_auc_score(testY, predictions))
    #     print 'LOG LOSS:' + str(log_loss(testY, predictions))
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(testY, predictions)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     plt.title('LOG_LOSS=' + str(log_loss(testY, predictions)))
    #     plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)
    #     plt.legend(loc='lower right')
    #     plt.plot([0, 1], [0, 1], 'r--')
    #     plt.xlim([-0.1, 1.2])
    #     plt.ylim([-0.1, 1.2])
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate')
    #     plt.show()
    #     return trainX, testX, trainY, testY, algo_1, roc_auc, lr_best_params


    # def gridLR(self, X_df_train, glm_factor=None):
    #     # lg = LogisticRegression(class_weight='balanced')
    #     lg = LogisticRegression()
    #     para_grid = {'penalty': ['l2'], 'C': [1, 5, 10, 50, 300, 500],
    #                  'solver': ['newton-cg'],
    #                  # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
    #                  'fit_intercept': [False, True]}
    #
    #     print para_grid
    #     para_search = grid_search.GridSearchCV(lg, para_grid, cv=10, verbose=False, scoring='roc_auc')
    #
    #     X_df_train_SINGLE = X_df_train.copy(deep=True)
    #     answers_1_SINGLE = list(X_df_train_SINGLE[TARGET_VAR].values)
    #     X_df_train_SINGLE = X_df_train_SINGLE.drop(TARGET_VAR, axis=1)
    #
    #     if glm_factor is not None:
    #         X_df_train_SINGLE=X_df_train_SINGLE[glm_factor]
    #     else:
    #         print 'No limit on the number of features'
    #
    #
    #     trainX, testX, trainY, testY = train_test_split(X_df_train_SINGLE, answers_1_SINGLE, test_size=.22)  # CV
    #     print 'Running:' + str(para_search) + ',shape:' + str(X_df_train_SINGLE.shape)
    #     model_train = para_search.fit(trainX, trainY)
    #     print para_search.best_params_
    #     print para_search.best_score_
    #     return para_search.best_params_

    # def bestFeaturesXGB(self, X_df_train_1, howManyFeatures=300, mx_depth=100):
    #     algo_xgbm1 = xgb.XGBClassifier(base_score=0.5, colsample_bytree=0.5,
    #                                    gamma=0.017, learning_rate=0.15, max_delta_step=0,
    #                                    max_depth=mx_depth, min_child_weight=3, n_estimators=2000,
    #                                    nthread=-1, objective='binary:logistic', seed=0,
    #                                    silent=1, subsample=0.9)
    #
    #     X_df_train_SINGLE = X_df_train_1.copy(deep=True)
    #     answers_1_SINGLE = list(X_df_train_SINGLE[TARGET_VAR].values)
    #     X_df_train_SINGLE = X_df_train_SINGLE.drop(TARGET_VAR, axis=1)
    #
    #     print 'Running:' + str(algo_xgbm1) + 'shape:' + str(X_df_train_SINGLE.shape)
    #     X_df_train_SINGLE = X_df_train_SINGLE.apply(lambda x: pandas.to_numeric(x, errors='ignore'))
    #
    #     trainX, testX, trainY, testY = train_test_split(X_df_train_SINGLE, answers_1_SINGLE, test_size=.33)  # CV
    #     model_train = algo_xgbm1.fit(trainX, trainY, early_stopping_rounds=200, eval_metric="auc",
    #                                  eval_set=[(testX, testY)], verbose=False)
    #     # print model_train
    #     predictions = algo_xgbm1.predict_proba(testX)[:, 1]
    #
    #     print 'ROC AUC:' + str(roc_auc_score(testY, predictions))
    #     print 'LOG LOSS:' + str(log_loss(testY, predictions))
    #     false_positive_rate, true_positive_rate, thresholds = roc_curve(testY, predictions)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     plt.title('LOG_LOSS=' + str(log_loss(testY, predictions)))
    #     plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)
    #     plt.legend(loc='lower right')
    #     plt.plot([0, 1], [0, 1], 'r--')
    #     plt.xlim([-0.1, 1.2])
    #     plt.ylim([-0.1, 1.2])
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate')
    #     plt.show()
    #
    #     feat_imp1 = pd.Series(algo_xgbm1.booster().get_fscore()).sort_values(ascending=False)
    #     feat_imp1 = feat_imp1.head(60)  # display only 60
    #     feat_imp1.plot(kind='bar', title='Feature Importances')
    #     plt.ylabel('Feature Importance Score')
    #
    #     # Build pasty expression -- feed best features automatically
    #     glm_factor1 = pd.Series(algo_xgbm1.booster().get_fscore()).sort_values(ascending=False)
    #     print 'Limiting to features:' + str(howManyFeatures)
    #     glm_factor1 = glm_factor1.head(howManyFeatures)
    #     glm_factor1 = list(glm_factor1.index)
    #
    #     return glm_factor1