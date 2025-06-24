### [2025.4.13] Simulation of speckle interferometry

##This code simulates the signal with different Kg visibilities, receiver aperture siezes, and mismatched speckles.


##process:
# 1. A rough taget at distance $z$ is illuminated with $M = N\times N$ beams

# 2. Assume the wavefront of all $M = N\times N$ tilted beams to be flat (plane wave)

# 3. Tilted beams have Gaussain beam profile with the tilted angle $\phi^{x}_{m}$ in $\hat{x}$ and $\phi^{y}_{n}$ in $\hat{y}$

# 4. Amplitude reflectivity $\tilde{R}(x,y)$ and randomized phasor $\Gamma$ to simulate $1^{st}$-order speckle patterns

# 5. Fourier transform the backscattered echos to the receiver pupil plane for $M = N\times N$ beams

# 6. Calculate the amplitude and phase from the spatial interferometry of $M(M-1)/2$ beams for a given fronzen time

# 7. Run time evolution

# Note:
# 1. ``k_mag'' is the magnitude of the Kg (object). The magnitude of 0.602 set in the simulation is to match the k-vector of the non-redundant beam array. 
# If Kg is parallel to x-direction, the beam pair that resonates with Kg will produce a good mesurment for the speckle interferometry of that RF frequency tone.

from matplotlib import pyplot as plt
import matplotlib as mpl
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import math
import sys
import time
import os
import h5py
import sympy
import pandas as pd
from datetime import date
from tqdm import tqdm

## the defalt path is ".../code". Need to go to upper directory for parent_dir
os.path.abspath('')
code_dir = "%s/"%os.getcwd()
parent_dir = code_dir.replace('code/','')
function_dir = parent_dir + 'functions/'
data_dir = parent_dir + 'data/'
fig_dir = parent_dir + 'figures/'
sys.path.insert(0,parent_dir)

## import coustom functions
import functions.f_complex_phasor_hsl_v1 as chsl
import functions.fbasis_functions_all_v4 as funs_v4
import functions.fbasis_functions_simulation_only_v1 as funs_sim

## functions
def imag_linear2log_v1(c1,c2,input):
    Ny,Nx = np.shape(input)
    input_mag = np.abs(input)
    in_max = np.max(input_mag)
    out_log = c1*np.log(1+c2*input)/np.log(1+c2*in_max)
    out_log_imag = np.zeros([Ny,Nx])
    out_log_imag[:,:] =out_log
    return out_log_imag 

def showfig_recon_image_spkl_v1(shouldsave,mgx,mgy,D_cir,dimension,irun,Ica,Ics,Icsa):

    fig,ax = plt.subplots(1,3,figsize=(12,4))
    fig.suptitle(f"Receiver aperture Size = {np.round(D_cir/mgx).astype('int')}x{np.round(D_cir/mgy).astype('int')} specklons")
    ax[0].imshow(Ica,cmap='gray')
    ax[0].set_title("Specular Surface")
    ax[0].axis('off')
    ax[1].imshow(Ics,cmap='gray')
    ax[1].set_title("Rough Surface")
    ax[1].axis('off')
    ax[2].imshow(Icsa,cmap='bwr')
    ax[2].axis('off')
    ax[2].set_title("Difference")

    if shouldsave:
        save_dir = parent_dir + f"figures/Recon_image/{dimension}x{dimension}/D_cir{D_cir}pixels/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + f"Recon_image_{dimension}x{dimension}_D_cir{D_cir}pixels_run{irun}.pdf", dpi=1000, bbox_inches='tight',transparent=True)


def showfig_recon_image_spkl_v2(shouldsave,mgx,mgy,D_cir,dimension,irun,Ica,Ics,I_complex_ag_hsl,I_complex_hsl):


    fig,ax = plt.subplots(1,4,figsize=(12,3))
    fig.suptitle(f"Receiver aperture Size = {np.round(D_cir/mgx).astype('int')}x{np.round(D_cir/mgy).astype('int')} specklons")
    ax[0].imshow(Ica,cmap='gray')
    ax[0].set_title("Object (Specular)")
    ax[0].axis('off')
    ax[1].imshow(I_complex_ag_hsl)
    ax[1].set_title("uv-Plane (Specular)")
    ax[1].axis('off')
    ax[2].imshow(Ics,cmap='gray')
    ax[2].set_title("Object (Rough)")
    ax[2].axis('off')
    ax[3].imshow(I_complex_hsl)
    ax[3].set_title("uv-Plane (Specular)")
    ax[3].axis('off')

    if shouldsave:
        save_dir = parent_dir + f"figures/Imag_uv/{dimension}x{dimension}/D_cir{D_cir}pixels/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + f"Imag_uv_{dimension}x{dimension}_D_cir{D_cir}pixels_run{irun}.pdf", dpi=1000, bbox_inches='tight',transparent=True)
        plt.close()


def showfig_recon_image_spkl_v3(shouldsave,mgx,mgy,D_cir,dimension,irun,Ica,Ics,I_complex_ag_hsl,I_complex_hsl):


    fig,ax = plt.subplots(1,4,figsize=(12,3))
    fig.suptitle(f"Receiver aperture Size = {np.round(D_cir/mgx).astype('int')}x{np.round(D_cir/mgy).astype('int')} specklons")
    ax[0].imshow(Ica,cmap='binary',origin='lower')
    ax[0].set_title("Object (Specular)")
    ax[0].axis('off')
    ax[1].imshow(I_complex_ag_hsl,origin='lower')
    ax[1].set_title("uv-Plane (Specular)")
    ax[1].axis('off')
    ax[2].imshow(Ics,cmap='binary',origin='lower')
    ax[2].set_title("Object (Rough)")
    ax[2].axis('off')
    ax[3].imshow(I_complex_hsl,origin='lower')
    ax[3].set_title("uv-Plane (Specular)")
    ax[3].axis('off')

    if shouldsave:
        save_dir = parent_dir + f"figures/grat_imag_uv_v3/{dimension}x{dimension}/D_cir{D_cir}pixels/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + f"Grat_imag_uv_{dimension}x{dimension}_D_cir{D_cir}pixels_run{irun}.pdf", dpi=1000, bbox_inches='tight',transparent=True)
        plt.close()
    else:
        plt.show()

## Load objects
saveobj = True
Nx = 512
Ny = 512
Nrx = int(2**10)
Nry = int(2**10)
Npadx = int((Nrx-Nx)/2)
Npady = int((Nry-Ny)/2)

# m_k = 0.1     ## scaled factor of the angle/k-vector
m_k = 0.1     ## scaled factor of the angle/k-vector

## object with three Kg-vector and visibility
Nobj = 1
phi = np.linspace(0,np.pi,Nobj,endpoint=True)
# k_mag = 0.689
k_mag = 0.689*m_k
kx_au = k_mag*np.cos(phi)
ky_au = k_mag*np.sin(phi)
mv = np.ones(Nobj)  ## mv<=1

# kx_au = (0.6,0.6,0,0.6)
# ky_au = (0,0.6,0.6,-0.6)
# mv = (1,1,1,1)  ## mv<1

## coordinate
x_au = np.arange(-Nx/2,Nx/2)
y_au = np.arange(-Ny/2,Ny/2)
X_au,Y_au = np.meshgrid(x_au,y_au)
x_au_pad = np.arange(-Nrx/2,Nrx/2)
y_au_pad = np.arange(-Nry/2,Nry/2)
X_au_pad,Y_au_pad = np.meshgrid(x_au_pad,y_au_pad)

## Gaussian tapering function
mgx = 1
mgy = 1
wx = mgx*Nx
wy = mgy*Ny
gaus = np.exp(-X_au**2/wx**2 - Y_au**2/wy**2)

## object reflectivity
Robj1 = np.zeros((Ny,Nx))
for isf in range(len(kx_au)):
# for isf in range(1):
    # Robj1_temp = 1 + mv[isf]*np.cos(*np.pi*(kx_au[isf]*X_au+ky_au[isf]*Y_au)+2*np.pi*isf/len(kx_au))
    Robj1_temp = 1 + mv[isf]*np.cos(2*np.pi*(kx_au[isf]*X_au+ky_au[isf]*Y_au)+np.pi/2)*gaus
    Robj1 = Robj1 + Robj1_temp
Robj1 = Robj1/np.mean(Robj1)
Robj1_pad = np.pad(Robj1,(Npadx,Npady))
FT_Robj1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Robj1_pad)))