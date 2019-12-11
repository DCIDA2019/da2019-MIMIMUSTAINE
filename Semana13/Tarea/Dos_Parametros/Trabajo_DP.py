#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import pymultinest
import os
import matplotlib.pyplot as plt
import copy
if not os.path.exists("chains"): os.mkdir("chains")

from scipy.integrate import trapz,simps
from colossus.cosmology import cosmology
#%matplotlib inline
plt.style.use('seaborn-whitegrid')

#try: os.mkdir('chains')
#except OSError: pass

#Datos
cosmo = cosmology.setCosmology('planck15')
cosmo.Om0, cosmo.Omde, cosmo.ns, cosmo.H0, cosmo.relspecies = 0.29, 0.71, 0.97, 70, False
cosmo.OmbO = 0.02247
cosmo.checkForChangedCosmology()
pk_cmasdr12 = numpy.loadtxt("GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt").T
print(pk_cmasdr12)
#Principal function
cosmo = cosmology.setCosmology('planck15',)

def Pk_Om(k, Om_, b, z = 0.57):
    cosmo.Om0 = Om_
    return b ** 2 * cosmo.matterPowerSpectrum(k, z)

# probability function, taken from the eggbox problem.

#Two parameters:
def myprior(cube):
	cube[0] = 0.5 * cube[0] + 0.2
	cube[1] = 3 * cube[1]
	return cube

#def myloglike(cube):
#	chi = (cos(cube / 2.)).prod()
#	return (2. + chi)**5

def myloglike(cube):
    x, y, yerr = pk_cmasdr12[0], pk_cmasdr12[1], pk_cmasdr12[2]
    model = Pk_Om(x, cube[0], cube[1])
    chisq = (y - model) ** 2 / yerr ** 2
    return - 0.5 * chisq.sum()

# number of dimensions our problem has
parameters = ["omega", "b"]
n_params = len(parameters)
# name of the output files
prefix = "chains/3-"

# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, 
	n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
#print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
	print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=prefix)
s = a.get_stats()
print(s)

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)

print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))
