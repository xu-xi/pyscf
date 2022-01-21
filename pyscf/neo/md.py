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
    autocorr = signal.correlate(traj, traj, mode='full')[len(traj)-1:]
    autocorr = numpy.multiply(autocorr, 1 / numpy.arange(len(traj), 0, -1))
    return autocorr

def hann(length):
    'Hann window function'
    n = numpy.arange(length)
    return numpy.power(numpy.cos(math.pi*n/(2*(length-1))), 2)

def vacf(datafile, start=0, end=-1, step=1):
    'calculate verlocity auto-correlation function (VACF) from trajectory of MD simulations'
    traj = Trajectory(datafile)

    v = []

    if end == -1:
        end = len(traj)

    for i in range(start, end, step):
        v.append(traj[i].get_velocities())
    v = numpy.array(v)

    global n # the number of atoms
    t, n, x = v.shape

    mass = traj[-1].get_masses()
    acf = 0

    for i in range(n): # i-th atom
        for j in range(x): # j-th component of verlocity
            acf += calc_ACF(v[:,i,j]) * mass[i]

    return acf

def dacf(datafile, time_step=0.5, start=0, end=-1, step=1):
    'calculate dipole auto-correlation function (DACF) from trajectory of MD simulations'
    traj = Trajectory(datafile)
    global n # the number of atoms
    n = len(traj[-1].get_positions())

    if end == -1:
        end = len(traj)

    dipole = []
    for i in range(start, end, step):
        dipole.append(traj[i].get_dipole_moment())
    dipole = numpy.array(dipole)

    t, x = dipole.shape

    acf = 0

    for j in range(x):
        de = numpy.gradient(dipole[:, j], time_step)
        acf += calc_ACF(de)

    return acf

def calc_FFT(acf):
    'get Fourier transform of ACF'

    # window function
    acf *= hann(len(acf))

    # zero padding
    N = 2 ** (math.ceil(math.log(len(acf), 2)) + 2)
    acf = numpy.append(acf, numpy.zeros(N - len(acf)))

    # data mirroring
    acf = numpy.concatenate((acf, acf[::-1][:-1]), axis=0)

    yfft = numpy.fft.fft(acf, axis=0)

    #numpy.set_printoptions(threshold=numpy.inf)
    #print(yfft)

    return yfft

def spectrum(acf, time_step=0.5, corr_depth=4096):
    'get wavenumber and intensity of spectrum'

    acf = acf[:corr_depth]

    yfft = calc_FFT(acf)

    fs2cm = 1e-15 * units._c * 100

    wavenumber = numpy.fft.fftfreq(len(yfft), time_step * fs2cm)[0:int(len(yfft)/2)]
    intensity = numpy.real(yfft[0:int(len(yfft)/2)])
    factor = 11604.52500617 * fs2cm / (3 * n) # eV2K * fs2cm -> K*cm
    intensity *= factor
    temperature = scipy.integrate.cumtrapz(intensity, wavenumber)
    print('Integrated temperature (K)', temperature[-1])
    return wavenumber, intensity, temperature
