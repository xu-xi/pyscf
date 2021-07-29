#!usr/bin/env python3
'''
calculate power spectrum and IR spectrum based on MD trajectory
'''
import math
import scipy
import numpy
from ase.io.trajectory import Trajectory

def step_average(data):
    'get cumulative averaged data'
    return numpy.cumsum(data)/numpy.arange(1, len(data) + 1)

def calc_ACF(traj):
    'calculate auto-correlation functions'
    autocorr = scipy.signal.fftconvolve(traj, traj[::-1], mode='full')[len(traj)-1:] #/ ynorm
    #traj_fft = numpy.fft.fft(traj)
    #autocorr = numpy.real(scipy.fft.ifft(traj_fft * numpy.conj(traj_fft))) /traj_fft.size
    return autocorr

def hann2(length):
    'Hann window function'
    n = numpy.arange(length)
    return numpy.power(numpy.cos(math.pi*n/(2*(length-1))), 2)

def vacf(datafile, start=1, end=-1, step=1):
    'calculate verlocity auto-correlation function (VACF) from trajectory of MD simulations'
    traj = Trajectory(datafile)

    v = []
    for i in range(len(traj)):
        v.append(traj[i].get_velocities())
    v = numpy.array(v)

    t, n, x = v.shape

    mass = traj[-1].get_masses()
    acf = 0

    for i in range(n):
        for j in range(x):
            acf += calc_ACF(v[start:end:step,i,j]) * mass[i]

    return acf

def dacf(datafile, timestep, start=1, end=-1, step=1):
    'calculate dipole auto-correlation function (DACF) from trajectory of MD simulations'
    traj = Trajectory(datafile)

    dipole = []
    for i in range(len(traj)):
        dipole.append(traj[i].get_dipole_moment())
    dipole = numpy.array(dipole)

    t, x = dipole.shape

    acf = 0

    for j in range(x):
        de = numpy.gradient(dipole[start:end:step, j], timestep)
        acf += calc_ACF(de)

    return acf

def calc_FFT(sig):
    'get Fourier transform of ACF'

    # zero padding
    N = int(2 ** math.ceil(math.log(len(sig), 2)))

    # data mirroring
    #sig = numpy.concatenate((sig, sig[1::-1]), axis=0)

    yfft = numpy.fft.fft(sig, N, axis=0) / len(sig)
    # return numpy.square(numpy.absolute(yfft))
    return yfft

def spectrum(acf, corr_depth=4096):
    'get wavenumber and intensity of spectrum'
    c = 2.9979245899e10 # speed of light in vacuum in [cm/s], from Wikipedia.

    acf = acf[:corr_depth]
    acf *= hann2(len(acf))
    #acf = acf*scipy.signal.windows.hann(len(acf), sym=False)
    yfft = calc_FFT(acf)

    wavenumber = numpy.fft.fftfreq(len(yfft), 0.5*1e-15*c)[0:int(len(yfft)/2)]
    intensity = yfft[0:int(len(yfft)/2)]
    temperature = scipy.integrate.cumtrapz(intensity, wavenumber) # TODO: change units
    return wavenumber, intensity 

