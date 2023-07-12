import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
from sample_sqw4 import sample_sqw4

#Here we define our sqw4 function 
def gauss(E,E0,damping,minI=0):
    vals = (1.0/np.sqrt(2.0*np.pi*damping**2))*np.exp(-(E-E0)**2 / (2.0*damping**2))
    return vals
def get_color(val,vmin,vmax,cmap):
    cmap = matplotlib.colormaps[cmap]
    norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
    rgba = cmap(norm(val))
    return rgba

def calc_sqw4_example(Qx,Qy,Qz,E,m1x,m1y,m1z,m2x,m2y,m2z,gap1,gap2):
    dispE1 = gap1 + m1x*np.abs(np.sin(2.0*np.pi*Qx/2)) + m1y*np.abs(np.sin(2.0*np.pi*Qy/2))+\
                    m1z*np.abs(np.sin(2.0*np.pi*Qz/2))
    dispE2 = gap2 + m2x*np.abs(np.sin(2.0*np.pi*Qx/2)) + m2y*np.abs(np.sin(2.0*np.pi*Qy/2))+\
                    m2z*np.abs(np.sin(2.0*np.pi*Qz/2))
    sqw4 = 1.0*gauss(E,dispE1,0.05,minI=0)/E+gauss(E,dispE2,0.05)/E # Factor of 1/w for spinwaves
    return sqw4

def calc_sqw4_delta(Qx,Qy,Qz,E):
    sqw4 = np.ones(np.shape(Qx))#*1e3 # Simulates simple incoherent or delta function scattering.
    return sqw4


plt.close('all')
m1x,m1y,m1z = 2.0,2.0,0.0 
m2x,m2y,m2z = 0.5,0.5,0.0
gap1,gap2 = 0.5,4.0

a,b,c = 2.0*np.pi,2.0*np.pi,2.0*np.pi
alpha,beta,gamma = 90.0,90.0,90.0

params = [m1x,m1y,m1z,m2x,m2y,m2z,gap1,gap2]


#Delta function example.
delta = 0.01
qx = np.linspace(1-delta,1+delta,51)
qy = np.linspace(0.5-delta,0.5+delta,51)
qz = np.linspace(0.0-delta,0.0+delta,21)
omega = np.linspace(1.0-delta,1.0+delta,50)
nH,nK,nL,nE = 10,10,10,10
Qy, Qx, Qz, E = np.meshgrid(qy,qx,qz,omega)
M_delta = sample_sqw4(calc_sqw4_delta,[],qx,qy,qz,omega,nH,nK,nL,nE,    
    lat_a=a,lat_b=b,lat_c=c,
    alpha=alpha,beta=beta,gamma=gamma,
    xnoise=True,ynoise=True,znoise=True,omeganoise=True,
    fname = 'sqw_delta_result.sqw4')

#Square in 1st quadrant. 
delta = 0.5
qx = np.linspace(1-delta,1+delta,51)
qy = np.linspace(1-delta/2,1+delta/2,51)
qz = np.linspace(0.0-delta/5,0.0+delta/5,21)
omega = np.linspace(1.0-delta,1.0+delta,50)
nH,nK,nL,nE = 20,20,10,10
Qy, Qx, Qz, E = np.meshgrid(qy,qx,qz,omega)
M_square = sample_sqw4(calc_sqw4_delta,[],qx,qy,qz,omega,nH,nK,nL,nE,    
    lat_a=a,lat_b=b,lat_c=c,
    alpha=alpha,beta=beta,gamma=gamma,
    xnoise=True,ynoise=True,znoise=True,omeganoise=True,
    fname = 'sqw_box_result.sqw4')

#Spin wave example.
qx = np.linspace(-1,1,51)
qy = np.linspace(-1,1,51)
qz = np.linspace(-0.05,0.05,21)
omega = np.linspace(0.2,6,50)
nH,nK,nL,nE = 50,50,5,40
Qy, Qx, Qz, E = np.meshgrid(qy,qx,qz,omega)
M = sample_sqw4(calc_sqw4_example,params,qx,qy,qz,omega,nH,nK,nL,nE,
    lat_a=a,lat_b=b,lat_c=c,
    alpha=alpha,beta=beta,gamma=gamma,
    xnoise=True,ynoise=True,znoise=True,omeganoise=True,
    fname = 'sqw_spinwave_result.sqw4')

print("Sampling is done. ")
#Let's make a 3D scatterplot to show how well the routine works. 
Imax = np.log(np.max(M[:,4]))
Imin = np.log(np.min(M[:,4]))
colors = get_color(np.log(M[:,4]),vmin=Imin,vmax=Imax,cmap='hot')
if len(M)<1e4:
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':'3d'})
    ax.scatter(M[:,0],M[:,1],M[:,3],marker='s',c=colors,alpha=0.3,linewidths=0,s=1)
    fig.show()
# Show some 2D meshes to demonstrate sampling. 
plots=True
if plots==True:
    #First plot spin waves. 
    fig,ax = plt.subplots(1,2,figsize=(8,3))
    #First a S(Qx,Qy,Qz=0.5,E=2) slice. 
    SQxQy_i = np.logical_and([np.abs(M[:,2]-0.0)<0.2],[np.abs(M[:,3]-2.1)<0.1]).flatten()
    colors = get_color(M[:,4],vmin=0,vmax=np.mean(M[:,4])*5,cmap='rainbow')
    ax[0].scatter(M[SQxQy_i][:,0],M[SQxQy_i][:,1],c=colors[SQxQy_i],marker='s',linewidths=0,s=3,rasterized=True)
    #Now a S(Qx,Qy=0,Qz=0.5,E) slice
    SQxE_i = np.logical_and([np.abs(M[:,1]-0.0)<0.05],[np.abs(M[:,2]-0.0)<0.2]).flatten()

    ax[1].scatter(M[SQxE_i][:,0],M[SQxE_i][:,3],c=colors[SQxE_i],marker='s',linewidths=0,s=3,rasterized=True,alpha=0.5)
    ax[0].set_xlabel('[H00] (r.l.u.)')
    ax[0].set_ylabel('[0K0] (r.l.u.)')
    ax[1].set_xlabel('[H00] (r.l.u.)')
    ax[0].set_title(r'Spin-waves, $\hbar\omega$=[2.0,2.2] (meV), $l$=[-0.2,0.2]')
    ax[0].set_title(r"$\hbar\omega$=[2.0,2.2] meV")
    ax[1].set_title(r"$k$=[-0.1,0.1], $l$=[-0.2,0.2]")
    ax[1].set_ylabel(r'$\hbar\omega$ (meV)')

    fig.savefig('sqw4_sampling_fig.pdf',bbox_inches='tight',dpi=300)
    fig.show()

    #Plot the delta function
    fig,ax = plt.subplots(1,2,figsize=(8,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.4)
    #First a S(Qx,Qy,Qz=0.5,E=1) slice. 
    SQxQy_i = np.logical_and([np.abs(M_delta[:,2]-0.0)<0.2],[np.abs(M_delta[:,3]-1.0)<0.1]).flatten()
    colors = get_color(M_delta[:,4],vmin=0,vmax=np.mean(M_delta[:,4])*5,cmap='rainbow')
    ax[0].scatter(M_delta[SQxQy_i][:,0],M_delta[SQxQy_i][:,1],c=colors[SQxQy_i],marker='s',linewidths=0,s=3,rasterized=True)
    #Now a S(Qx,Qy=0,Qz=0.5,E) slice
    SQxE_i = np.logical_and([np.abs(M_delta[:,1]-0.5)<0.1],[np.abs(M_delta[:,2]-0.0)<0.2]).flatten()

    ax[1].scatter(M_delta[SQxE_i][:,0],M_delta[SQxE_i][:,3],c=colors[SQxE_i],marker='s',linewidths=0,s=3,rasterized=True,alpha=0.5)
    ax[0].set_xlabel('[H00] (r.l.u.)')
    ax[0].set_ylabel('[0K0] (r.l.u.)')
    ax[1].set_xlabel('[H00] (r.l.u.)')
    ax[1].set_ylabel(r'$\hbar\omega$ (meV)')
    ax[1].set_title(r'[1,1/2,0] 1 meV Delta func, $\hbar\omega$ (meV), $l$=[-0.2,0.2]')
    ax[0].set_title(r"$\hbar\omega$=[0.9,1.1] meV")
    ax[0].set_xlim(0.9,1.1)
    ax[0].set_ylim(0.4,0.6)
    ax[1].set_xlim(0.9,1.1)
    ax[1].set_ylim(0.5,1.5)
    fig.savefig('sqw4_delta_sampling_fig.pdf',bbox_inches='tight',dpi=300)
    fig.show()

    #Finally the box centered around (1+-0.5,1+-0.25,0) at 1 meV
    fig,ax = plt.subplots(1,2,figsize=(8,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.4)
    #First a S(Qx,Qy,Qz=0.5,E=1) slice. 
    SQxQy_i = np.logical_and([np.abs(M_square[:,2]-0.0)<0.2],[np.abs(M_square[:,3]-1.0)<0.1]).flatten()
    colors = get_color(M_square[:,4],vmin=0,vmax=np.mean(M_square[:,4])*5,cmap='rainbow')
    ax[0].scatter(M_square[SQxQy_i][:,0],M_square[SQxQy_i][:,1],c=colors[SQxQy_i],marker='s',linewidths=0,s=3,rasterized=True)
    #Now a S(Qx,Qy=0,Qz=0.5,E) slice
    SQxE_i = np.logical_and([np.abs(M_square[:,1]-0.5)<0.1],[np.abs(M_square[:,2]-0.0)<0.2]).flatten()

    ax[1].scatter(M_square[SQxE_i][:,0],M_square[SQxE_i][:,3],c=colors[SQxE_i],marker='s',linewidths=0,s=3,rasterized=True,alpha=0.5)
    ax[0].set_xlabel('[H00] (r.l.u.)')
    ax[0].set_ylabel('[0K0] (r.l.u.)')
    ax[1].set_xlabel('[H00] (r.l.u.)')
    ax[1].set_ylabel(r'$\hbar\omega$ (meV)')
    ax[1].set_title(r'[1,1/2,0] 1 meV Box func, $\hbar\omega$ (meV), $l$=[-0.2,0.2]')
    ax[0].set_title(r"$\hbar\omega$=[0.9,1.1] meV")
    ax[0].set_xlim(-2,2)
    ax[0].set_ylim(-2,2)
    ax[1].set_xlim(-2,2)
    ax[1].set_ylim(0.0,2.5)
    fig.savefig('sqw4_box_sampling_fig.pdf',bbox_inches='tight',dpi=300)
    fig.show()



