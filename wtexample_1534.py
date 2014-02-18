#! /usr/bin/python

import matplotlib.pylab as plb
import PulsePortrait as pp
import numpy as np
import aspProfile
import pywt
import sys

inf = './1534prof.std'
indata = aspProfile.Profile(inf)
prof = indata.power
prof = np.roll(prof,512)

data = pp.wavelet1D(prof)
diff = prof-data


plb.subplot(3,1,1)
plb.plot(prof,'r-')
plb.subplot(3,1,2)
plb.plot(prof,'r-')
plb.plot(data,'b-')
plb.ylim(np.min(prof),np.max(prof)/50.)
plb.subplot(3,1,3)
plb.plot(diff,'b-')
plb.show()

print 'Residual mean, standard deviation: {0:.8f}, {1:.8f}'.format(np.mean(diff),np.std(diff))

