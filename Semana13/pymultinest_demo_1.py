#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import pymultinest
import os
if not os.path.exists("chains") : os.mkdir("chains")
#try: os.mkdir('chains')
#except OSError: pass

# probability function, taken from the eggbox problem.

def myprior(cube):
	return cube * 10 * pi

def myloglike(cube):
	chi = (cos(cube / 2.)).prod()
	return (2. + chi)**5

# number of dimensions our problem has
parameters = ["x", "y"]
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

#lets analyse the results
a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=prefix)
s = a.get_stats()
print(s)

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
	json.dump(parameters, f, indent=2)

with open('%sstats.json' %  a.outputfiles_basename, mode='w') as f:
	json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))
