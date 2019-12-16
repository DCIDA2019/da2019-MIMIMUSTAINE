from __future__ import print_function
import matplotlib.pyplot as plt
import copy
import numpy as np
from scipy.integrate import trapz,simps
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-whitegrid')

np.random.seed(40)

from colossus.cosmology import cosmology

cosmo = cosmology.setCosmology('planck15')

cosmo.Om0, cosmo.Omde, cosmo.ns, cosmo.H0, cosmo.relspecies = 0.29, 0.71, 0.97, 70, False
#cosmo.OmbO = 0.02247
cosmo.checkForChangedCosmology()

cosmo = cosmology.setCosmology('planck15',)

pk_cmasdr12 = np.loadtxt("GilMarin_2016_CMASSDR12_measurement_monopole_post_recon.txt").T

def Pk_Om(k, Om_, b, z = 0.57):
    cosmo.Om0 = Om_
    return b ** 2 * cosmo.matterPowerSpectrum(k, z)

def Pk_Om(k, Om_, b, beta, z = 0.57):
    cosmo.Om0 = Om_
    return b ** 2 * (1 + beta) * cosmo.matterPowerSpectrum(k, z)

def likelihood(parametro, v):
    #Valores medidos:
    x, y, yerr = v[0], v[1], v[2]
    model = Pk_Om(x, parametro[0], parametro[1], parametro[2])
    chisq = (y - model) ** 2 / yerr ** 2
    return - 0.5 * chisq.sum()

def prior(parametro):
    if 0.001 < parametro[0] < 1.0 and 0.001 < parametro[1] < 5.0 and 0.001 < parametro[2] < 3.0:
        return 0.0
    return - np.inf

def ejecucion(parametro, v):
    prior_ = prior(parametro)
    if not np.isfinite(prior_):
        return - np.inf
    return likelihood(parametro, [pk_cmasdr12[0], pk_cmasdr12[1], pk_cmasdr12[2]]) + prior_

N = 50000
caminos = 20
sigma = 0.05
#Arreglos
om_inicial = []
b_inicial = []
beta_inicial = []
lh_inicial = []

#Valor inicial (a, b, lh):
for i in range(caminos):
    om_inicial.append(np.random.normal(0.34, sigma))
    b_inicial.append(np.random.normal(2.0, sigma))
    beta_inicial.append(np.random.normal(1.5, sigma))
    #Valor inicial del likelihood:
    lh_inicial.append(likelihood([om_inicial[i], b_inicial[i], beta_inicial[i]], [pk_cmasdr12[0], pk_cmasdr12[1], pk_cmasdr12[2]]))

#Arreglos con los valores de a, b, likelihood (resultado):
om = []
b = []
beta = []
lh = []

#Se guardan los primeros valores propuestos:
for i in range(caminos):
    om.append([om_inicial[i]])
    b.append([b_inicial[i]])
    beta.append([beta_inicial[i]])
    lh.append([lh_inicial[i]])

#Contador para guardar en lugar indicado resutado para arreglos a, b, lh:
k = 0

#El algoritmo se repite para cada camino:
for j in range(caminos):
    for i in range(N - 1):
        om_aux = np.random.normal(om[j][k], sigma)
        b_aux = np.random.normal(b[j][k], sigma)
        beta_aux = np.random.normal(beta[j][k], sigma)
        lh_aux = ejecucion([om_aux, b_aux, beta_aux], [pk_cmasdr12[0], pk_cmasdr12[1], pk_cmasdr12[2]])
        if lh_aux > lh_inicial[j]:
            om[j].append(om_aux)
            b[j].append(b_aux)
            beta[j].append(beta_aux)
            lh[j].append(lh_aux)
            lh_inicial[j] = lh_aux
            k += 1
        else:
            comparador = - np.log(np.random.uniform(0,1))
            if (lh_aux - lh_inicial[j]) > comparador:
                om[j].append(om_aux)
                b[j].append(b_aux)
                beta[j].append(beta_aux)
                lh[j].append(lh_aux)
                lh_inicial[j] = lh_aux
                k += 1
            else:
                om[j].append(om[j][i - 1])
                b[j].append(b[j][i - 1])
                beta[j].append(beta_aux)
                lh[j].append(lh_inicial[j])
                k += 1
    k = 0

#fig = plt.figure()
#colors = ('r', 'g', 'b')
#ax = fig.add_subplot(111, projection='3d')
#plt.title('Convergencia de om, b')
#for i in range(caminos):
#    ax.scatter(om[i], b[i], beta[i], c = colors[i], alpha = 0.6, label = 'Ruta {}'.format(i))
#    ax.set_ylabel('b')
#    ax.set_xlabel('om')
#    ax.set_zlabel('$\\beta$')
#    ax.legend(frameon = True)
#plt.savefig('Convergencia.png')

#for i in range(caminos):
#	plt.scatter(om[i], b[i], alpha = 0.6, label = 'Ruta = {}'.format(i))
#plt.xlabel('$\\Omega$')
#plt.ylabel('b')
#plt.title('Convergencia')
#plt.savefig('Convergencia1.png')

#for i in range(caminos):
#        plt.scatter(om[i], beta[i], alpha = 0.6, label = 'Ruta = {}'.format(i))
#plt.xlabel('$\\Omega$')
#plt.ylabel('$\\beta$')
#plt.title('Convergencia')
#plt.savefig('Convergencia2.png')

#for i in range(caminos):
#        plt.scatter(beta[i], b[i], alpha = 0.6, label = 'Ruta = {}'.format(i))
#plt.xlabel('$\\beta$')
#plt.ylabel('b')
#plt.title('Convergencia')
#plt.savefig('Convergencia3.png')


om_total = []

for i in range(caminos):
    for j in range(np.int(len(om[i]) / 2), len(om[i])):
        om_total.append(om[i][j])
plt.hist(om_total, 10, color = 'black', alpha = 0.3)
#plt.axvline(a_0, color = 'red', label = 'Real = {}'.format(a_0))
plt.axvline(np.mean(om_total), color = 'yellow', label = 'Media = {}'.format(np.round(np.mean(om_total), 4)))
plt.axvline(np.median(om_total), color = 'purple', linestyle = ':', label = 'Mediana = {}'.format(np.round(np.median(om_total), 4)))
plt.xlabel('om')
plt.ylabel('Frecuencia')
plt.legend(frameon = True)
plt.savefig('Omega.png')

b_total = []

for i in range(caminos):
    for j in range(np.int(len(b[i]) / 2), len(b[i])):
        b_total.append(b[i][j])
plt.hist(b_total, 20, color = 'black', alpha = 0.3)
#plt.axvline(b_0, color = 'red', label = 'Real = {}'.format(b_0))
plt.axvline(np.mean(b_total), color = 'yellow', label = 'Media = {}'.format(np.round(np.mean(b_total), 4)))
plt.axvline(np.median(b_total), color = 'purple', linestyle = ':', label = 'Mediana = {}'.format(np.round(np.median(b_total), 4)))
plt.xlabel('b')
plt.ylabel('Frecuencia')
plt.legend(frameon = True)
plt.savefig('b.png')

beta_total = []

for i in range(caminos):
    for j in range(np.int(len(beta[i]) / 2), len(beta[i])):
        beta_total.append(beta[i][j])
plt.hist(beta_total, 20, color = 'black', alpha = 0.3)
#plt.axvline(b_0, color = 'red', label = 'Real = {}'.format(b_0))
plt.axvline(np.mean(beta_total), color = 'yellow', label = 'Media = {}'.format(np.round(np.mean(beta_total), 4)))
plt.axvline(np.median(beta_total), color = 'purple', linestyle = ':', label = 'Mediana = {}'.format(np.round(np.median(beta_total), 4)))
plt.xlabel('$\\beta$')
plt.ylabel('Frecuencia')
plt.legend(frameon = True)
plt.savefig('beta.png')

k = 10 ** np.linspace(-6, 5, 100000)
plt.figure()
plt.loglog()
plt.errorbar(pk_cmasdr12[0], pk_cmasdr12[1], yerr = pk_cmasdr12[2], fmt='.', color = 'black')
plt.plot(k, Pk_Om(k, np.mean(om_total), np.mean(b_total), np.mean(beta)), color = 'red', label = 'a = {}, b = {}, $\\beta$ = {}'.format(np.round(np.mean(om_total), 4), np.round(np.mean(b_total), 4), np.round(np.mean(beta_total), 4)))
plt.legend(frameon = True)
plt.xlim(1e-3, 1)
plt.ylim(100, 3e5)
plt.savefig('Resultado.png')
