#! /usr/bin/python

import numpy as np
import PulsePortrait as pp
import matplotlib.pylab as plb
import pywt
import sys


# set fake-data params.
freq  = 1400.
nchan = 16
nbins = 1024*1
bw    = 64.    # MHz

# get grid settings.
dx, dy = 1./np.float(nbins), np.float(bw/nchan)
x, y   = np.mgrid[slice(0.,1.,dx),slice(freq-bw/2.,freq+bw/2.+dy,dy)] 

# get function sorted with grid settings.
mean, sig = 0.5, 0.5
z = np.exp(-((x-mean)/(sig*y/freq/4.))**2)+np.exp(-((x-0.25)/(sig*y/freq/10.))**2)

# add some noise, and remove last 'spare' channel.
noise_mean, noise_sig = 0., 0.04
z = z + np.random.normal(noise_mean,noise_sig,size=(nbins,nchan+1))
z = z[:,:-1]

# "scintillate."
for i in range(nchan):
  z[:,i] = z[:,i]*np.random.uniform(0.8,1.0)

# plot stuff in color!
plb.pcolormesh(x,y,z,vmin=z.min(),vmax=z.max())
plb.axis([x.min(),x.max(),y.min(),y.max()])
plb.colorbar()
plb.show()

# denoise.
z_wt = pp.wavelet2D(z,ncycle=nbins,threshtype='soft')

# plot wavelet-denoised data in color!
diff = z-z_wt

plb.pcolormesh(x,y,z_wt,vmin=z_wt.min(),vmax=z_wt.max())
plb.axis([x.min(),x.max(),y.min(),y.max()])
plb.colorbar()
plb.show()

plb.pcolormesh(x,y,diff,vmin=diff.min(),vmax=diff.max())
plb.axis([x.min(),x.max(),y.min(),y.max()])
plb.colorbar()
plb.show()

# print numbers for comparison.
print "Original noise mean, standard deviation: {0:.4f}, {1:.4f}".format(noise_mean,noise_sig)
print "Residual noise mean, standard deviation: {0:.4f}, {1:.4f}".format(np.mean(diff),np.std(diff))
