#! /usr/bin/python

import matplotlib.pylab as plb
import PulsePortrait as pp
import numpy as np
import pywt
import sys

# generate a fake profile.
nbin  = 1024*1.
phi   = np.arange(np.int(nbin))/(nbin-1.)
prof  = np.exp(-((phi-0.5)/(0.05))**2)+0.5*np.exp(-((phi-0.3)/(0.01))**2)

# add some noise.
noise_mean, noise_sig = 0., 0.01
noise = np.random.normal(noise_mean,noise_sig,len(prof))
prof = prof + noise

# denoise.
data = pp.wavelet1D(prof,ncycle=np.int(nbin))
diff = prof-data

# plot it up.
plb.subplot(3,1,1)
plb.plot(prof,'r-')
plb.ylabel('Original')
plb.subplot(3,1,2)
plb.plot(prof,'r-')
plb.plot(data,'b-')
plb.ylabel('Denoised')
plb.ylim(np.min(prof),np.max(prof)/5.)
plb.subplot(3,1,3)
plb.plot(diff,'b-')
plb.ylabel('Residual')
plb.show()

print 'Original noise mean, standard deviation: {0:.8f}, {1:.8f}'.format(noise_mean,noise_sig)
print 'Residual noise mean, standard deviation: {0:.8f}, {1:.8f}'.format(np.mean(diff),np.std(diff))
