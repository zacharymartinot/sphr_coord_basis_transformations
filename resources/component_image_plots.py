import numpy as np, healpy as hp, matplotlib.pyplot as plt
from spherical_coordinates_basis_transformation import *

nside = 256
npix = hp.nside2npix(nside)
hpxidx = np.arange(npix)
theta, phi = hp.pix2ang(nside, hpxidx)
phi = 2. * np.pi - phi

RotAngle = np.radians(30.)
RotAxis = np.array([0,-1,0])

R = rotation_matrix(RotAxis, RotAngle)

cosX, sinX = spherical_basis_transformation_components(theta, phi, R)

fig = plt.figure(figsize=(16,8))
hp.orthview(cosX,rot=[-45,45], cmap='RdBu_r', sub=(1,2,1), half_sky=True, min=-1,max=1)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_title(r'$\cos \chi$',fontsize=32)

hp.orthview(sinX,rot=[-45,45], cmap='RdBu_r',sub=(1,2,2),half_sky=True, min=-1,max=1)
ax = plt.gca()
ax.set_aspect('equal')
ax.set_title(r'$\sin \chi$',fontsize=32)

# plt.savefig('plots/component_images.png')
plt.show()
