import numpy as np 
import matplotlib.pyplot as plt 
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import binned_statistic_2d
import scipy
plt.close('all')
'''
Simple script to plot CrI3 data, more robust implementation of a UB matrix
should be done at some point. 
'''


def import_sqqw(fname):
	'''
	Provided a mcstas sqq_w_monitor output file, returns energy, A3, I, Err, Qx, Qy
	'''
	f = open(fname)
	lines = f.readlines()
	f.close()
	for line in lines:
		if "Param: A3" in line:
			A3 = float(line.split(' ')[2].split('=')[1].replace('\n',''))
		elif "param: Ei" in line:
			Ei = float(line.split(' ')[2].split('=')[1].replace('\n',''))
		elif "xylimits:" in line:
			qx_min = float(line.split(' ')[2])
			qx_max = float(line.split(' ')[3])
			qz_min = float(line.split(' ')[4])
			qz_max = float(line.split(' ')[5])
		elif "title: qa vs qb monitor" in line:
			omega = float(line.split('~')[-1].split(' ')[1])
		elif "type:" in line:
			n_qx = int(line.split('(')[-1].split(',')[0])
			n_qz = int(line.split('(')[-1].split(',')[1].replace(' ','').replace(')',''))
	dat = np.genfromtxt(fname)
	qz_list = np.linspace(qz_min,qz_max,n_qz)
	qx_list = np.linspace(qx_min,qx_max,n_qx)
	I, Err, N = dat[0:n_qz], dat[n_qz:2*n_qz],dat[2*n_qz:]
	return [A3,omega,qx_list,qz_list,I,Err,N]


uvec = np.array([1,0.0]) # We are ignoring the L-axis, defining beam direction as 2nd index
vvec = np.array([0,1]) # These are in rlu, u is perpendicular to beam v is along it. 

#The astar, bstar, cstar are the cartsian rlu for A3 = 0 
#In general this can be found in SpinW using CrI3.rl
astar = np.array([1.0563,0])
bstar=np.array([0.528265,0.914982])
scandir = 'CrI3_scans/'
dirs = glob.glob(scandir+"*")
Ei = 30.0

qx_samplebins = np.linspace(-2.5,2.5,300)
qz_samplebins = np.linspace(-2,2,100)
omegabins = np.linspace(0,25,50)

for i,d in enumerate(dirs):
	print(f"On folder {i:.0f} / {len(dirs)}")
	fnames = glob.glob(d+'/qa_vs_qb_*.dat')
	files_d = [f for f in fnames if "Sum" not in f]
	#These need to be sorted. 
	fnums = [int(files_d[s].split('.')[0].split('_')[-1]) for s in range(len(files_d))]
	maxf = np.max(fnums)
	files = [d+f"/qa_vs_qb_{ind}.dat" for ind in range(maxf+1)]
	for jj, f in enumerate(files):
		if i==0 and jj==0:
			#Need to initialize the output matrices. 
			out_init = import_sqqw(f)
			#We will keep track of 3D Q-E space through a matrix of dimension nQx * nQz * nE
			numE = len(files)
			#Assume that the energies are sorted.
			e_list = [out_init[1]]
			#Output matrix has dimensions of Omega, Qu, Qv
			outmat = np.zeros((numE,len(qx_samplebins)-1,len(qz_samplebins)-1))
			outnum = np.zeros(np.shape(outmat)) #Keep track of number of events for averaging
			#Now project on to the sample.
			A3 = out_init[0]*np.pi/180.0
			R = np.array([[np.cos(A3),-np.sin(A3)],[np.sin(A3),np.cos(A3)]])
			Qx_lab, Qz_lab = np.meshgrid(out_init[2],out_init[3])
			Qlab = np.array([Qx_lab,Qz_lab])
			Qsamp = np.zeros(np.shape(Qlab))
			for i2 in range(np.shape(Qx_lab)[0]):
				for j2 in range(np.shape(Qx_lab)[1]):
					Qxpt_lab = Qlab[0,i2,j2]
					Qypt_lab = Qlab[1,i2,j2]
					Qsamp[:,i2,j2]=np.matmul(R,np.array([Qxpt_lab,Qypt_lab]))
			#Populate the output. 
			Qx_f = Qsamp[0].flatten()
			Qz_f = Qsamp[1].flatten()
			I_f = out_init[4].flatten()
			N_f = out_init[5].flatten()
			binned_sqqw = scipy.stats.binned_statistic_2d(Qx_f,Qz_f,I_f,
				statistic='sum',bins=(qx_samplebins,qz_samplebins))
			binned_sqqw_num = scipy.stats.binned_statistic_2d(Qx_f,Qz_f,N_f,
				statistic='sum',bins=(qx_samplebins,qz_samplebins))
			outmat[0,:,:] = binned_sqqw.statistic
			outnum[0,:,:] = binned_sqqw_num.statistic
		else:
			out = import_sqqw(f)
			omega = out[1]
			if i==0: #Only need to do this once
				e_list.append(omega)
			A3 = out[0]*np.pi/180.0
			R = np.array([[np.cos(A3),-np.sin(A3)],[np.sin(A3),np.cos(A3)]])
			Qx_lab, Qz_lab = np.meshgrid(out[2],out[3])
			Qx_lab, Qz_lab = Qx_lab.flatten(), Qz_lab.flatten()
			Qlab = np.array([Qx_lab,Qz_lab])
			Qsamp = np.dot(R,Qlab)
			#Populate the output. 
			Qx_f = Qsamp[0]
			Qz_f = Qsamp[1]
			I_f = out[4].flatten()
			N_f = out[5].flatten()
			binned_sqqw = scipy.stats.binned_statistic_2d(Qx_f,Qz_f,I_f,
				statistic='sum',bins=(qx_samplebins,qz_samplebins))
			binned_sqqw_num = scipy.stats.binned_statistic_2d(Qx_f,Qz_f,N_f,
				statistic='sum',bins=(qx_samplebins,qz_samplebins))
			outmat[jj,:,:] += binned_sqqw.statistic
			outnum[jj,:,:] += binned_sqqw_num.statistic		
#Normalize each to number of counts. 
#outmat[outnum==0]=np.nan
outmat/=outnum 
outmat[outnum==0]=0
e_list = np.array(e_list)

qx_bincen = qx_samplebins[1:]-np.abs(qx_samplebins[1]-qx_samplebins[0])/2.0
qz_bincen = qz_samplebins[1:]-np.abs(qz_samplebins[1]-qz_samplebins[0])/2.0


Qx, Qz, E = np.meshgrid(qx_bincen,qz_bincen, e_list)
#Pick a const E slice to check. 
qx_eslicebins = np.linspace(-2.5,2.5,71)
qz_eslicebins = np.linspace(-2.5,2.5,71)

eslice_bin = scipy.stats.binned_statistic_dd(sample=[Qz.flatten(),Qx.flatten(),E.flatten()],values=outmat.T.flatten(),statistic=np.nanmean,bins=[qz_eslicebins,qx_eslicebins,[5.5,6.5]])
Qx_Eslice, Qz_Eslice = np.meshgrid(qx_eslicebins,qz_eslicebins)
E_i = np.argmin(np.abs(e_list-E_test))	

qxbins = np.linspace(-2.5,2.5,100)
Eh00bins = np.linspace(0,25,40)

#Slice along H00
h00bin = scipy.stats.binned_statistic_dd(sample=[Qz.flatten(),Qx.flatten(),E.flatten()],values=outmat.T.flatten(),statistic=np.nanmean,bins=[[-0.2,0.2],qxbins,Eh00bins])
Qx_h00, E_h00 = np.meshgrid(qxbins,Eh00bins)
fig,ax = plt.subplots(1,2,figsize=(9,4))
vmax=5
ax[0].pcolormesh(Qx_Eslice,Qz_Eslice,eslice_bin.statistic[:,:,0],cmap='viridis',vmin=0,vmax=vmax)
mesh = ax[1].pcolormesh(Qx_h00,E_h00,h00bin.statistic[0,:,:].T,cmap='viridis',vmin=0,vmax=vmax)
ax[0].set_xlabel(r'$Q_x$ ($\AA^{-1}$)')
ax[0].set_ylabel(r'$Q_z$ ($\AA^{-1}$)')
ax[0].set_title(r"$\hbar\omega$=6.0(5) meV slice")
ax[1].set_ylabel(r"$\hbar\omega$ (meV)")
ax[1].set_xlabel(r"$(h00)$ (r.l.u.)")
ax[1].set_title("CrI$_3$ SPINW Simulation, $k$=0")
fig.show()




