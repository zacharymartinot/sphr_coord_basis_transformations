import numpy as np, healpy as hp
from mayavi import mlab
from spherical_coordinates_basis_transformation import *

nside = 8
npix = hp.nside2npix(nside)
hpxidx = np.arange(npix)
theta, phi = hp.pix2ang(nside, hpxidx)
phi = 2. * np.pi - phi

RotAngle = np.radians(30.)
RotAxis = np.array([0,-1,0])

R = rotation_matrix(RotAxis, RotAngle)

beta, alpha = spherical_coordinates_map(R, theta, phi)

q1,q2,q3 = r_hat(theta, phi)

# p1,p2,p3 = r_hat(beta, alpha) # is equivalent
p1,p2,p3 = np.einsum('ab...,b...->a...', R, r_hat(theta, phi))

th1,th2,th3 = theta_hat(theta, phi)
ph1,ph2,ph3 = phi_hat(theta, phi)

bh1,bh2,bh3 = np.einsum('ab...,b...->a...', R.T, theta_hat(beta, alpha))
ah1,ah2,ah3 = np.einsum('ab...,b...->a...', R.T, phi_hat(beta, alpha))

tg = np.linspace(0., np.pi, 100)
pg = np.linspace(0., 2. * np.pi, 200)

xg = np.outer(np.cos(pg), np.sin(tg))*0.99
yg = np.outer(np.sin(pg), np.sin(tg))*0.99
zg = np.outer(np.ones_like(pg), np.cos(tg))*0.99

# mayavi's line plotter (mlab.plot3d) is broken, gotta make my own lines. lame.
Z_tp = 2. * np.array([0,0,1])
Z_ba = 2. * np.dot(R.T, np.array([0,0,1.])) # in q coordinates, so the rotation is R.T
x = np.linspace(0,1,1000)

Z_ba_line = -(1. - x) * Z_ba[:,None] + x * Z_ba[:,None]
Z_tp_line = -(1. - x) * Z_tp[:,None] + x * Z_tp[:,None]

######
mlab.figure(1, bgcolor=(1,1,1), fgcolor=(0,0,0), size=(600,600))
mlab.clf()

mlab.mesh(xg,yg,zg, color=(0.8,0.8,0.8))
mlab.points3d(q1,q2,q3, scale_factor=0.01)

mlab.quiver3d(q1,q2,q3,th1,th2,th3, scale_factor=0.06, color=(0,0,0.8))
mlab.quiver3d(q1,q2,q3,ph1,ph2,ph3, scale_factor=0.06, color=(0,0,0.8))

mlab.points3d(Z_tp_line[0],Z_tp_line[1],Z_tp_line[2],color=(0,0,0.8), scale_factor=0.01)

#######
mlab.figure(2, bgcolor=(1,1,1), fgcolor=(0,0,0), size=(600,600))
mlab.clf()


mlab.mesh(xg,yg,zg, color=(0.8,0.8,0.8))
mlab.points3d(q1,q2,q3, scale_factor=0.01)

mlab.quiver3d(q1,q2,q3,ah1,ah2,ah3, scale_factor=0.06, color=(0.8,0,0))
mlab.quiver3d(q1,q2,q3,bh1,bh2,bh3, scale_factor=0.06, color=(0.8,0,0))

mlab.points3d(Z_ba_line[0],Z_ba_line[1],Z_ba_line[2],color=(0.8,0,0), scale_factor=0.01)

########
mlab.figure(3, bgcolor=(1,1,1), fgcolor=(0,0,0), size=(600,600))
mlab.clf()

mlab.mesh(xg,yg,zg, color=(0.8,0.8,0.8))
mlab.points3d(q1,q2,q3, scale_factor=0.01)

mlab.quiver3d(q1,q2,q3,th1,th2,th3, scale_factor=0.06, color=(0,0,0.8))
mlab.quiver3d(q1,q2,q3,ph1,ph2,ph3, scale_factor=0.06, color=(0,0,0.8))

mlab.quiver3d(q1,q2,q3,ah1,ah2,ah3, scale_factor=0.06, color=(0.8,0,0))
mlab.quiver3d(q1,q2,q3,bh1,bh2,bh3, scale_factor=0.06, color=(0.8,0,0))

mlab.points3d(Z_tp_line[0],Z_tp_line[1],Z_tp_line[2],color=(0,0,0.8), scale_factor=0.01)
mlab.points3d(Z_ba_line[0],Z_ba_line[1],Z_ba_line[2],color=(0.8,0,0), scale_factor=0.01)

mlab.show()
