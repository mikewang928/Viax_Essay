import numpy as np
import scipy
from scipy import signal
class NeuralFeature():
    def all_features(self, data):  # data = (windows,channel,sample)
        lln = self.lln(data)
        var = self.var(data)
        pow = self.pow(data)
        bp = self.bandpower(data, fmin=[1,4,8,13,30,50,80,150], fmax=[4,8,13,30,50,80,150,250])
        return np.concatenate((lln, var, pow, bp), axis=1)

    def lln(self, data):
        temp = np.concatenate((data[:,:,0][:,:,None],data[:,:,:-1]),axis=2)
        lln = np.sum(np.abs(data - temp),axis=-1)
        return lln

    def var(self,data):
        temp = np.mean(data, axis=-1)
        var = data - temp[:, :, None]
        var = np.mean(var**2, axis=-1)
        return var

    def pow(self,data):
        pow = np.mean(data**2, axis=-1)
        return pow

    def bandpower(self, x, fmin, fmax):
        n, c, fs = x.shape
        bp = np.zeros((n, c*len(fmin)))
        for i in range(n):
            for ii in range(c):
                f, Pxx = signal.periodogram(x[i, ii, :], fs=fs)
                for iii in range(len(fmin)):
                    ind_min = scipy.argmax(f > fmin[iii]) - 1
                    ind_max = scipy.argmax(f > fmax[iii]) - 1
                    bp[i, ii*len(fmin)+iii] = scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
        return bp