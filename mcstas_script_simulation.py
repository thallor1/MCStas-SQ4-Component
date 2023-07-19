import mcstasscript as ms
import numpy as np
my_configurator = ms.Configurator()
my_configurator.set_mcrun_path("/usr/bin/")
my_configurator.set_mcstas_path("/usr/share/mcstas/3.3/")

#my_configurator.set_mxrun_path("/usr/bin/")
#my_configurator.set_mcxtrace_path("/usr/share/mcxtrace/1.5/")
#Remove old scans. 
import os
import copy
import scipy
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import binned_statistic_2d

from joblib import Parallel,delayed

for root, dirs, files in os.walk('scandir/'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

#Output .sqw4 file:
sqw_file='sqw_calc_result.sqw4'
#plt.close('all')
#Check if this file exists, if not generate it. 
if os.path.exists(sqw_file):
    pass
else:
    print("Warning: Need to generate sqw4 file first.")



inel_test_inst = ms.McStas_instr("test_instrument")
A3 = inel_test_inst.add_parameter("A3", value=0.0,
                                      comment="Sample A3 Rotation angle.")
Ei = inel_test_inst.add_parameter("Ei", value=10.0,
                                      comment="Incident neutron energy mean.")

vix = inel_test_inst.add_user_var("double", "vix", comment="Initial neutron x velocity")
viy = inel_test_inst.add_user_var("double", "viy", comment="Initial neutron y velocity")
viz = inel_test_inst.add_user_var("double", "viz", comment="Initial neutron z velocity")


progress = inel_test_inst.add_component("progress","Progress_bar")
progress.set_AT([0,0,0])
#Simple neutron source 
source = inel_test_inst.add_component("source", "Source_gen")
source.radius = 0.0775
source.focus_xw = 0.01 
source.focus_yh = 0.01
source.E0 = Ei 
source.dE = 0.1
source.I1 = 1e10
source.verbose=1
source.set_AT([0,0,0])
source.set_ROTATED([0,0,0])
#source.append_EXTEND("vix=vx;viy=vy;viz=vz;")

crystal_assembly = inel_test_inst.add_component("crystal_assembly","Arm")
crystal_assembly.set_AT([0,0,2],RELATIVE=source)
crystal_assembly.set_ROTATED([0,A3,0],RELATIVE=source)
#Union processes
init = inel_test_inst.add_component("init","Union_init")

#Collimator 
'''
col1 = inel_test_inst.add_component("col1","Slit")
col1.radius=(0.01)
col1.set_AT([0,0,-0.3],RELATIVE="crystal_assembly")
col1.set_ROTATED([0,0,0])
'''
#Sample sqw4 processs
sample_sqw4 = inel_test_inst.add_component("sample_sqw4", "Sqw4_process")
'''
recip_cell=0, barns=1,
ax = 6.283, ay = 0, az = 0,
bx = 0, by = 0, bz = 6.283,
cx = 0, cy = 6.283, cz = 0,
aa=90, bb=90, cc=90,
interact_fraction=-1, packing_factor=1, init="init"
'''
sample_sqw4.sqw = '"sqw_spinwave_result.sqw4"'
sample_sqw4.ax=6.283
sample_sqw4.ay=0
sample_sqw4.az=0
sample_sqw4.bx=0
sample_sqw4.by=0
sample_sqw4.bz=6.283
sample_sqw4.cx=0
sample_sqw4.cy=6.283
sample_sqw4.cz=0
sample_sqw4.aa=90
sample_sqw4.bb=90
sample_sqw4.cc=90
sample_sqw4.barns=1
sample_sqw4.max_stored_ki=1e5
sample_sqw4.max_bad=1e5
sample_sqw4.stored_dTheta = 1
sample_sqw4.stored_dkmag = 1e-3
sample_sqw4.recip_cell=0
sample_sqw4.interact_fraction=-1
#sample_sqw4.init="'init'"
#sample_sqw4.append_EXTEND("// Remove direct beam\nif(!SCATTERED) ABSORB;")
sample_sqw4.set_AT([0,0,0])
sample_sqw4.set_ROTATED([0,0,0])

sample_material = inel_test_inst.add_component("sample_material","Union_make_material")
sample_material.my_absorption=0.0
sample_material.process_string='"sample_sqw4"'

#Actual box representing the crystal. 
sample_box = inel_test_inst.add_component("sample_box", "Union_box", AT=[0.0,0,0], RELATIVE=crystal_assembly)
sample_box.xwidth = 0.01
sample_box.yheight = 0.01
sample_box.zdepth = 0.01
sample_box.material_string = '"sample_material"'
sample_box.priority = 100
sample_box.set_AT([0,0,0],RELATIVE="crystal_assembly")
sample_box.set_ROTATED([0,0,0],RELATIVE="crystal_assembly")


#Union master.
master = inel_test_inst.add_component("master", "Union_master")
master.append_EXTEND("// Remove direct beam\nif(!SCATTERED) ABSORB;")

#Union stop necessary. 
stop = inel_test_inst.add_component("stop", "Union_stop") # This component has to be called stop

#Beamstop
'''
beamstop = inel_test_inst.add_component("beamstop", "Beamstop")
beamstop.xmin = -0.05
beamstop.xmax = 0.05
beamstop.ymin = -0.5
beamstop.ymax = 0.5 
beamstop.set_AT([0,0,1.25],RELATIVE=source)
beamstop.set_ROTATED([0,0,0])
'''

#MonitorND version below:
monitornd =inel_test_inst.add_component("monitornd","Monitor_nD")
monitornd.radius=0.3
monitornd.yheight=0.4
numthetabins = 200
numomegabins = 60
optionsstr = "banana, theta limits=[-90,90], bins="+f"{numthetabins}"+ f", energy limits=[4.0,9.7], bins={numomegabins}"
monitornd.options='\"'+optionsstr+'\"'
monitornd.filename='"banana_det_theta_E.dat"'
monitornd.set_AT([0,0,0.00],RELATIVE="crystal_assembly")
monitornd.set_ROTATED([0,0,0])


numA3=300
A3_list = np.linspace(-90,90,numA3)
#A3_list=np.array([ ])
#Preallocate output array, cols are H,K,E,I,Err,N
npts = numA3*numthetabins*numomegabins
Outmat = np.zeros((npts,6))

matind = 0 
n_threads = 8
Ei_test=10.0
#Start by compiling the instrument, then vary the params. 
inel_test_inst.settings(ncount=100,force_compile=True,output_path=f"scandir/mcscript_test_monitorND_A3_0",
    checks=False)
inel_test_inst.set_parameters(A3=0,Ei=10.0)

dat_a3 = inel_test_inst.backengine()
#Initialize the output matrix. 
output_I_shape = np.shape(dat_a3[0].Intensity) # Will be of dimension energy x theta bins
#W will only keep one energy, so output matrix will be of numA3xthetabins
#Matrix columns will be Qx, Qy, I, Err
outmat = np.zeros((numA3*numthetabins,4))

#Second plot of (h00) vs E
h_pts=[]
omega_pts = []
i_pts = []
#Clear out the directory. 
for root, dirs, files in os.walk('scandir/'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

#Define a function to parallelize. 
def run_scan(instrument, A3):
    print(f"Running A3={A3:.3f}")
    inst_copy = instrument# copy.deepcopy(instrument)
    inst_copy.settings(ncount=1e5,force_compile=False,output_path=f"scandir/mcscript_test_monitorND_A3_{A3:.3f}deg".replace('.','p').replace('-','m'),
        checks=False)
    inst_copy.set_parameters(A3=A3,Ei=10.0)
    dat = inst_copy.backengine()
    return dat

datlist = Parallel(n_jobs = n_threads, backend="threading")(delayed(run_scan)(inel_test_inst,A3) for A3 in A3_list)

# also need a way to load files if we don't want to redo all of this. 


h_pts=[]
omega_pts = []
i_pts = []
for ii,dat in enumerate(datlist):
    if ii==0:
        matind=0
    print(f"ii={ii}, A3={A3_list[ii]:.3f} deg")
    A3_test = A3_list[ii]
    dat_a3 = datlist[ii]
    if len(dat_a3)==0:
        #Some error occurred during the scan. 
        print("Error.")
        continue 
    DEG2RAD = np.pi/180.0
    #Project crystal axes to Q. 
    Rmat = np.array([[np.cos(A3_test*DEG2RAD),-np.sin(A3_test*DEG2RAD)],[np.sin(A3_test*DEG2RAD),np.cos(A3_test*DEG2RAD)]])
    qu = np.array([0,1])
    qv = np.array([1,0])
    qu_rot = np.matmul(Rmat,qu)
    qv_rot = np.matmul(Rmat,qv)
    metadata = dat_a3[0].metadata 
    Intensities = dat_a3[0].Intensity
    dims = metadata.dimension # Length of longitude, energy respectively
    lims = metadata.limits
    banana_a4 = np.linspace(lims[0],lims[1],dims[0])
    #The convention for this component is backwards...
    banana_a4*=-1.0
    energies = np.linspace(lims[2],lims[3],dims[1])
    A4,E = np.meshgrid(banana_a4,energies) #Mat of bin centers

    #Get the total Q for each scattering event based on twotheta and energy
    ki_mat = np.ones(np.shape(A4))*np.sqrt(Ei_test/2.072)
    kf_mat = np.sqrt(E/2.072)
    #Get directions. 
    ki_dir = np.array([0,1]) #Defining beam direction as y, 90 deg rotation as x 
    ki_dir_mat = ki_dir[:,np.newaxis,np.newaxis]*np.ones(np.shape(E))
    kf_dir_mat = np.array([-np.sin(A4*DEG2RAD),np.cos(A4*DEG2RAD)])

    #Apply magnitudes. 
    ki_vec_mat = ki_mat*ki_dir_mat
    kf_vec_mat = kf_mat*kf_dir_mat
    Qlab = ki_vec_mat-kf_vec_mat
    #Rotate these directions into the sample frame
    ki_mat_smpl = np.copy(ki_vec_mat)
    kf_mat_smpl = np.copy(kf_vec_mat)
    Q_smpl = ki_mat_smpl-kf_mat_smpl

    for i in range(np.shape(ki_mat_smpl)[1]):
        for j in range(np.shape(ki_mat_smpl)[2]):

            kilab = ki_vec_mat[:,i,j]
            kflab = kf_vec_mat[:,i,j]
            ki_smpl = np.matmul(Rmat,kilab)
            kf_smpl = np.matmul(Rmat,kflab)
            ki_mat_smpl[:,i,j]=ki_smpl
            kf_mat_smpl[:,i,j]=kf_smpl
            Qs = ki_smpl-kf_smpl
            Qlab = kilab-kflab
            Q_smpl[:,i,j]=Qs
    #Filter out some energies. 
    Omega = Ei_test-E
    #Check energies around 1 meV. 
    Omega_i = np.argmin(np.abs(Ei_test-energies-2.0))
    Qu=Q_smpl[0,:,:]
    Qv=Q_smpl[1,:,:]
    Qu_slice=Qu[Omega_i,:]
    Qv_slice=Qv[Omega_i,:]
    I_slice = Intensities[Omega_i,:]

    #Fill output matrix
    N_A3 = dat_a3[0].Ncount[Omega_i,:]
    Err_A3 = dat_a3[0].Error[Omega_i,:]
    outmat[matind:matind+len(Qu_slice.flatten()),:]=np.array([Qu_slice.flatten(),Qv_slice.flatten(),I_slice,Err_A3]).T

    #For second slice, append all values to the output list where |Qv|<0.1
    qv_i =np.where(np.abs(Qv.flatten()-0.0)<0.04)[0]
    Qu_list = Qu.flatten()[qv_i].tolist()
    E_list = Omega.flatten()[qv_i].tolist()
    I_list = Intensities.flatten()[qv_i].tolist()
    if(len(I_list)>0):
        h_pts=h_pts+Qu_list
        omega_pts=omega_pts+E_list
        i_pts=i_pts+I_list

    matind+=len(Qu_slice.flatten())
print("Success.")
#Bin

#Bin the space in 2D.
fig,ax = plt.subplots(1,2,figsize=(8,3))
qx_binedges = np.linspace(-2,2,131)
qy_binedges = np.linspace(-2,2,131)
Qx,Qy = np.meshgrid(qx_binedges,qy_binedges)
I_slice = np.zeros(np.shape(Qx))

binned_statistic = scipy.stats.binned_statistic_2d(outmat[:,0], outmat[:,1], outmat[:,2], statistic='mean', bins=(qx_binedges,qy_binedges), 
    range=None, expand_binnumbers=False)
Z = binned_statistic.statistic
#ax[0].pcolormesh(Qx,Qy,Z.T,norm=colors.LogNorm(vmin=np.nanmean(Z)/10,vmax=np.nanmax(Z)),cmap='rainbow',rasterized=True)
ax[0].pcolormesh(Qx,Qy,Z.T,vmin=0,vmax=np.nanmax(Z)/1.5,cmap='rainbow',rasterized=True)

#Now the (h00) vs E slice
h_arr = np.array(h_pts).flatten()
omega_arr= np.array(omega_pts).flatten()
I_arr = np.array(i_pts).flatten()
en_binedges = np.linspace(0.2,6,60)
h_binedges = np.linspace(-2,2,131)
Qh, Ebins = np.meshgrid(h_binedges,en_binedges)
binned_h00_statistic = scipy.stats.binned_statistic_2d(h_arr,omega_arr,I_arr,statistic='mean',bins=(h_binedges,en_binedges))
Z_h00 = binned_h00_statistic.statistic
#ax[1].pcolormesh(Qh,Ebins,Z_h00.T,norm=colors.LogNorm(vmin=np.nanmean(Z_h00)/10,vmax=np.nanmax(Z_h00)),cmap='rainbow',rasterized=True)
ax[1].pcolormesh(Qh,Ebins,Z_h00.T,vmin=0,vmax=np.nanmax(Z_h00)/1.5,cmap='rainbow',rasterized=True)
ax[0].set_xlabel('[H00] (r.l.u.)')
ax[0].set_ylabel('[0K0] (r.l.u.)')
ax[1].set_xlabel('[H00] (r.l.u.)')
ax[0].set_title(r'Spin-waves, $\hbar\omega$=[2.0,2.2] (meV), $l$=[-0.2,0.2]')
ax[0].set_title(r"$\hbar\omega$=[2.0,2.2] meV")
ax[1].set_title(r"$k$=[-0.1,0.1], $l$=[-0.2,0.2]")
ax[1].set_ylabel(r'$\hbar\omega$ (meV)')
ax[0].set_xlim(-2,2)
ax[0].set_ylim(-2,2)
ax[1].set_xlim(-2,2)
ax[1].set_ylim(0,6)
fig.savefig('test_inel_spinwave_slices.pdf',bbox_inches='tight',dpi=300)
fig.show()


#ms.make_sub_plot(dat_a3)



'''
#
a3_list = np.linspace(0,180,15)
dat_a3_dict = {}
I_tot = 0 
Err_tot = 0 
N_tot = 0 
for i,A3 in enumerate(a3_list):
    if i==0:
        force_comp = True
    else:
        force_comp = False
    print(f"On A3={A3:.2f} deg")
    inel_test_inst.settings(ncount=1E6,force_compile=force_comp,output_path=f"scandir/mcscript_test_A3_{A3:.2f}",checks=False)
    inel_test_inst.set_parameters(A3=A3)
    dat_a3 = inel_test_inst.backengine()
    for j,dat in enumerate(dat_a3): #Iterates through the energy bins
        #Skip the first entry, it's the sum of all energy bins. 
        if j==0:
            continue
        if i==0 and j==1:
            dat_final = dat_a3.copy()
        dat_a3_I = dat_a3[j].Intensity 
        dat_a3_Err = dat_a3[j].Error
        dat_a3_N = dat_a3[j].Ncount
        Inorm = dat_a3_I#/dat_a3_N
        Errnorm = dat_a3_I#/dat_a3_N
        N_new = dat_a3_N
        #N_new = np.ones(np.shape(dat_a3_N))
        #Fix nan's 
        Inorm[np.isnan(Inorm)]=0
        Errnorm[np.isnan(Errnorm)]=0
        dat_a3[j].Intensity=Inorm
        dat_a3[j].Error = Errnorm 
        dat_a3[j].Ncount = N_new
        if i==0 and j==1:
            I_tot = Inorm
            Err_tot=Errnorm
            N_tot = N_new 
        else:
            I_tot+=Inorm 
            Err_tot+=Errnorm 
            N_tot+=N_new
        dat_final[j].Intensity=I_tot
        dat_final[j].Error=Err_tot
        dat_final[j].N=N_tot 
    dat_a3_dict[f"A3_{A3:.2f}_deg"]=dat_a3 
    print(f"Done with A3={A3:.2f} deg")


#Edit the data in here to be scaled over N 
for i,dat in enumerate(dat_final): #Iterates through energies. 
    I_slice = dat.Intensity
    Err_slice = dat.Error 
    N_slice = dat.Ncount
    I_norm_slice = I_slice#/N_slice 
    Err_norm_slice = Err_slice#/ N_slice
    #N_slice/=N_slice 
    dat_final[i].Intensity=I_norm_slice
    dat_final[i].Error=Err_norm_slice
    dat_final[i].Ncount=N_slice #Should be 1 

ms.make_sub_plot(dat_final)
'''