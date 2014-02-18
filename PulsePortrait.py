#! /usr/bin/python

import numpy as np
import pywt

def wavelet1D(indat,wave='sym8',nlevel=5,ncycle=10,threshtype='hard'):
  """
  Compute the wavelet-denoised version of an input profile/waveform. 

  Required argument: 
    - 'indat' = input data in 1-dimensional NumPy array. 

  Default arguments: 
    - 'wave' = mother wavelet (default 'sym8') 
    - 'nlevel' = number of decomposition levels (default '5') 
    - 'ncycle' = number of circulant averages to compute (default '10') 
    - 'threshtype' = type of wavelet thresholding (default 'hard')
  """

  nbins = len(indat)
  data  = np.zeros(nbins)

  # average shifted/denoised/re-shifted data 'ncycle' times.
  for j in range(ncycle):
    m = j - ncycle/2 - 1
    coeffs = pywt.wavedec(np.roll(indat,m),wave,level=nlevel)
    # get coefficient threshold value.
    lopt = np.median(np.fabs(coeffs[1]))/0.6745*np.sqrt(2.*np.log(nbins))
    # now do wavelet thresholding.
    for k in range(1,nlevel+1):
      # hard threshold.
      if (threshtype == 'hard'):
        (coeffs[k])[np.where(np.fabs(coeffs[k]) < lopt)] = 0.
      # soft threshold. 
      else:
        (coeffs[k])[np.where(np.fabs(coeffs[k]) < lopt)] = 0.
        (coeffs[k])[np.where(coeffs[k] > lopt)] = (coeffs[k])[np.where(coeffs[k] > lopt)]+lopt
        (coeffs[k])[np.where(coeffs[k] < -lopt)] = (coeffs[k])[np.where(coeffs[k] < -lopt)]-lopt
    # reconstruct data.
    data = data + np.roll(pywt.waverec(coeffs,wave),-m)   
  
  # return smoothed profile.
  return data/np.float(ncycle)




def wavelet2D(indat,wave='sym8',nlevel=5,ncycle=10,threshtype='hard'):
  """
  Compute the wavelet-denoised version of a set of profiles/waveforms.
  
  Required argument:
    - 'indat' = (nbins x nchans) NumPy data array.
  
  Default arguments: 
    - 'wave' = mother wavelet (default 'sym8') 
    - 'nlevel' = number of decomposition levels (default '5') 
    - 'ncycle' = number of circulant averages to compute (default '10') 
    - 'threshtype' = type of wavelet thresholding (default 'hard')
  """

  nbins = len(indat[:,0])
  nchan = len(indat[0,:])

  outdat = np.zeros((nbins,nchan))
  
  # smooth each channel.
  for i in range(nchan):
    prof = indat[:,i]
    data = 0.
    # average shifted/denoised/re-shifted data 'ncycle' times.
    for j in range(ncycle):
      m = j - ncycle/2 - 1
      coeffs = pywt.wavedec(np.roll(prof,m),wave,level=nlevel)
      # get threshold value.
      lopt = np.median(np.fabs(coeffs[1]))/0.6745*np.sqrt(2.*np.log(nbins))
      # now do wavelet thresholding.
      for k in range(1,nlevel+1):
        # hard threshold.
        if (threshtype == 'hard'):
          (coeffs[k])[np.where(np.fabs(coeffs[k]) < lopt)] = 0.
        # or soft threshold.
        else:
          (coeffs[k])[np.where(np.fabs(coeffs[k]) < lopt)] = 0.
          (coeffs[k])[np.where(coeffs[k] > lopt)] = (coeffs[k])[np.where(coeffs[k] > lopt)]+lopt
          (coeffs[k])[np.where(coeffs[k] < -lopt)] = (coeffs[k])[np.where(coeffs[k] < -lopt)]-lopt
      # reconstruct data.
      data = data + np.roll(pywt.waverec(coeffs,wave),-m)
    # save averaged profile.
    outdat[:,i] = data/np.float(ncycle)

  # return smoothed portrait.
  return outdat
