#!usr/bin/env python3
'''
calculate power spectrum and IR spectrum based on MD trajectory
'''
import math
import scipy
import numpy
from scipy import signal
from ase.io.trajectory import Trajectory
from ase import units

def step_average(data):
    'get cumulative averaged data'
    return numpy.cumsum(data)/numpy.arange(1, len(data) + 1)

def calc_ACF(traj):
    'calculate auto-correlation functions'
    autocorr = signal.fftconvolve(traj, traj[::-1], mode='full')[len(traj)-1:] / len(traj)
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

def dacf(datafile, time_step=0.5, start=1, end=-1, step=1):
    'calculate dipole auto-correlation function (DACF) from trajectory of MD simulations'
    traj = Trajectory(datafile)

    dipole = []
    for i in range(len(traj)):
        dipole.append(traj[i].get_dipole_moment())
    dipole = numpy.array(dipole)

    t, x = dipole.shape

    acf = 0

    for j in range(x):
        de = numpy.gradient(dipole[start:end:step, j], time_step)
        acf += calc_ACF(de)

    return acf

def calc_FFT(acf):
    'get Fourier transform of ACF'

    # zero padding
    N = int(2 ** math.ceil(math.log(len(acf), 2)))

    # data mirroring
    #acf = numpy.concatenate((acf, acf[1::-1]), axis=0)

    yfft = numpy.fft.fft(acf, N, axis=0)
    # return numpy.square(numpy.absolute(yfft))
    return yfft

def spectrum(acf, time_step=0.5, corr_depth=4096):
    'get wavenumber and intensity of spectrum'

    acf = acf[:corr_depth]
    acf *= hann2(len(acf))
    yfft = calc_FFT(acf)

    fs2cm = 1e-15 * units._c * 100

    wavenumber = numpy.fft.fftfreq(len(yfft), time_step * fs2cm)[0:int(len(yfft)/2)]
    intensity = yfft[0:int(len(yfft)/2)]
    factor = 11604.52500617 * units.fs * fs2cm # eV2K * au2cm -> K*cm
    intensity *= factor
    temperature = scipy.integrate.cumtrapz(intensity, wavenumber)
    return wavenumber, intensity, temperature 

