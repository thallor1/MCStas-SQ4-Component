import numpy as np 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm 
from scipy.interpolate import interp1d

def sample_sqw4(sqw_f,p,qx,qy,qz,omega,nQx,nQy,nQz,nE,lat_a=None,lat_b=None,lat_c=None,
	alpha=None,beta=None,gamma=None,
	fname=None,
	xnoise=True,ynoise=False,znoise=False,omeganoise=False,
	domega_estimate=None,force_omegas=False,force_qx=False,force_qy=False,force_qz=False):
	'''
	Function to efficiently sample a 4D SQW function for input 
	into McStas simulations. The approach is based upon which 
	parts of the SQW function carry the most spectral weight. 

	Inputs:
	sqw_f - function of the form sqw(qx,qy,qz,*p) that returns SQW
	p - input arguments for sqw function, positional
	qx,qy,qz,omega - np arrays dictating points to calculate inital coarse grid of sqw
	nE,nQx,nQy,nQz - number of points in each dimension for final sampling
	xnoise,ynoise,znoise - Boolean that will add gaussian around each point in these axes. 
		Expece significant slowdown. 
	domega_estimate - estimate from user of the useful energy resolution to calculate. 
	force_qx/y/z/omega - manually specify particular axes for final output, skips sampling process.
	fname - filename for optional output.sqw4 file
	lat_a, lab_b, lat_c, alpha, beta, gamma : Lattice parameters, in Ang and deg respectively
	'''
	# Intitialize sample parameters, generate recip lattice. 
	alpha_r = alpha*np.pi/180.0
	beta_r = beta*np.pi/180.0
	gamma_r = gamma*np.pi/180.0
	avec = np.array([lat_a,0.0,0.0])
	bvec = np.array([lat_b*np.cos(gamma_r),lat_b*np.sin(gamma_r),0])
	cvec = np.array([lat_c*np.cos(beta_r),lat_c*(np.cos(alpha_r)-np.cos(beta_r)*np.cos(gamma_r))/(np.sin(gamma_r)),\
		lat_c*np.sqrt(1.0-np.cos(beta_r)**2-((np.cos(alpha_r)-np.cos(beta_r)*np.cos(gamma_r))/np.sin(gamma_r))**2)])
	V_recip=np.dot(avec,np.cross(bvec,cvec))
	astar_vec = 2.0*np.pi*np.cross(bvec,cvec)/V_recip
	bstar_vec = 2.0*np.pi*np.cross(cvec,avec)/V_recip
	cstar_vec = 2.0*np.pi*np.cross(avec,bvec)/V_recip

	debug=False
	#Begin by generating coarse grid of SQW 
	Qy,Qx,Qz,E = np.meshgrid(qy,qx,qz,omega)
	SQW= sqw_f(Qx,Qy,Qz,E,*p)
	# Get a spectral weight function G(omega)
	Gw_xy = np.trapz(SQW,x=qz,axis=2)
	Gw_x = np.trapz(Gw_xy,x=qy,axis=1)
	Gw = np.trapz(Gw_x,x=qx,axis=0)

	#Now we want the integral of Gw. 
	gw_int = []
	for i in range(len(omega)):
		gw_int.append(np.trapz(Gw[0:i],x=omega[0:i]))
	#Normalize Gw and gw_int to one. 
	norm = np.trapz(Gw,x=omega)
	Gw =Gw/norm
	gw_int = np.array(gw_int)/norm
	#Create a finer sampled linear interpolation:
	test_omegas = np.linspace(np.min(omega),np.max(omega),int(1e4))
	gw_int_interp = np.interp(test_omegas,omega,gw_int)
	gw_interp = interp1d(x=omega,y=Gw,fill_value=0,bounds_error=False)
	#Every time the integral goes up by desired amount, mark as a test omega. 
	dInt = 1.0/nE
	cross_points = np.linspace(0.0,1,nE)
	dist_vec = np.abs(gw_int_interp - cross_points[:,np.newaxis])
	cross_i = [np.argmin(dist_vec[ii]) for ii in range(len(dist_vec))]
	if len(cross_i) != nE:
		#This can happen in cases with no dispersion. Use user input. 
		cross_omegas = np.linspace(np.min(omegas),np.max(omegas),nE)
	else:
		cross_omegas = test_omegas[cross_i]
	if force_omega==True:
		final_omegas=omega
	else:
		final_omegas = cross_omegas # Just an easier name to remember. 
	# Optional plot of extracted DOS. 
	if debug==True:
		fig,ax = plt.subplots(1,1)
		ax_gw = ax.twinx()
		ax.plot(omega,gw_int,'k-',label=r'$\int d\omega G(\omega)$')
		ax.plot(cross_omegas,cross_points,'bo',label='Sample points')
		ax.plot(omega,Gw,'r-',label=r'G($\omega$)')
		ax.legend()
		ax.set_xlabel('E (meV)')
		ax.set_ylabel(r'G($\omega$)')
		ax_gw.legend()
		fig.savefig('gw_fig.pdf',bbox_inches='tight')
		fig.show()
	#At each energy transfer, generate 3D S(Q,w=w0)
	nPts_per_w = nQx*nQy*nQz
	totalRows = nQx*nQy*nQz*len(cross_omegas)
	#Preallocate our output array:
	outmat = np.empty((totalRows,5))#H,K,L,W,SQW
	matindex = 0
	i=0
	for i in tqdm(range(len(cross_omegas)), desc=f"Energies: ",leave=True):
		w_pt = cross_omegas[i]
		Qy_pt, Qx_pt, Qz_pt = np.meshgrid(qy,qx,qz)
		SQxyz = sqw_f(Qx_pt,Qy_pt,Qz_pt,w_pt,*p)
		#Determine the correct density of points along the L-axis using the same spectral weight method.
		SQyz = np.trapz(SQxyz,x=qx,axis=0)
		SQz = np.trapz(SQyz,x=qy,axis=0)
		norm = np.trapz(SQz,x=qz)
		SQz/=norm
		sqz_int = np.array(SQz)/norm
		#Create a finer sampled linear interpolation:
		test_qz = np.linspace(np.min(qz),np.max(qz),int(1e4))
		sqz_int_interp = np.interp(test_qz,qz,sqz_int)
		sqz_interp = interp1d(x=qz,y=SQz)
		#Every time the integral goes up by desired amount, mark as a test omega. 
		dInt = 1.0/nQz
		cross_qzpoints = np.linspace(0.0,1+dInt/2.0,nQz)
		dist_vec = np.abs(sqz_int_interp - cross_qzpoints[:,np.newaxis])
		cross_i = [np.argmin(dist_vec[ii]) for ii in range(len(dist_vec))]
		if len(np.unique(test_qz[cross_i])) != nQz:
			#This can happen in cases with no dispersion. 
			cross_qz = np.linspace(np.min(qz),np.max(qz),nQz) 
		else:
			cross_qz = test_qz[cross_i]
		final_qz = cross_qz # Just an easier name to remember. 	
		# At this particular energy transfer, the list of Qz / L points has been chosen.
		# Now, generate a much finer grid in the Qx/Qy plane now that memory will allow. 
		for j in tqdm(range(len(final_qz)),desc='Qz',leave=False):
			qz_pt = final_qz[j]
			SQz_pt = SQz[j]
			qx_fine, qy_fine = np.linspace(np.min(qx),np.max(qx),round(nQx*5)),np.linspace(np.min(qy),np.max(qy),round(nQy*5))
			Qy_z, Qx_z = np.meshgrid(qy_fine,qx_fine)
			SQxQy = sqw_f(Qx_z,Qy_z,qz_pt,w_pt,*p)/norm # The overall spectral weight of the Q3D slice is conserved. 
			if debug==True and i==int(round(len(cross_omegas)/3)) and j==int(round(len(final_qz)/2)):
				fig3,ax3 = plt.subplots(1,1,figsize=(4,3))
				ax3.pcolormesh(Qx_z,Qy_z,SQxQy,cmap='Spectral_r',vmin=0,vmax=np.max(SQxQy)/1.5)
				ax3.set_xlabel('Qx')
				ax3.set_ylabel('Qy')
				fig3.show()
			SQy = np.trapz(SQxQy,x=qx_fine,axis=0)
			dqy = np.abs(qy_fine[1]-qy_fine[0])
			SQy_int = np.cumsum(SQy*dqy/np.trapz(SQy,x=qy_fine))

			SQy_interp = interp1d(x=qy_fine,y=SQy)
			dI_y = 1.0/nQy
			cross_qy_points = np.linspace(0.0,1,nQy)
			dS = np.abs(SQy_int - cross_qy_points[:,np.newaxis])
			cross_i = [np.argmin(dS[ii]) for ii in range(len(dS))]
			if len(np.unique(qy_fine[cross_i]))!=nQy:
				cross_qy = np.linspace(np.min(qy),np.max(qy),nQy)
			else:
				cross_qy = qy_fine[cross_i]
			#Now, we know which Qy values we will evaluate. Finally, iterate through these and get our final points. 
			for k in range(len(cross_qy)):
				qy_pt = cross_qy[k]
				SQx = SQxQy[:,cross_i[k]]
				SQx[np.isnan(SQx)]=0
				qx_pts = np.random.choice(qx_fine,size=(nQx),replace=False,p=SQx/np.nansum(SQx))
				#Get the SQW values at this H,K,L,E
				pts_i = np.argmin(np.abs(qx_fine-qx_pts[:,np.newaxis]),axis=1)
				SQx_picked = SQx[pts_i]#*norm undo probability distribution nature of it. 
				outmat[matindex:matindex+len(SQx_picked),0]=qx_pts 
				# If desired, can add additional noise to each axis here.
				if ynoise is False:
					outmat[matindex:matindex+len(SQx_picked),1]=np.ones(len(qx_pts))*qy_pt
				else:
					#Pick a random point between the lower and upper neighbor of this Qy. 
					if k==0:
						#No lower neighbor
						lowQy = cross_qy[k]
						highQy = cross_qy[k+1]
					elif k==range(len(cross_qy))[-1]:
						#No upper neighbor
						lowQy = cross_qy[k-1]
						highQy = cross_qy[k]
					else: 
						lowQy = cross_qy[k-1]
						highQy = cross_qy[k+1]
					#Return a random array of Qy points in this range, centered around original val.
					origQy = qy_pt 
					qy_range = np.abs(highQy-lowQy)
					stddev = qy_range/5
					newQypts = np.random.normal(qy_pt,stddev,len(qx_pts))
					#ensure that no points exceed the boundaries. 
					newQypts[newQypts>highQy]=highQy 
					newQypts[newQypts<lowQy]=lowQy 
					outmat[matindex:matindex+len(SQx_picked),1]=newQypts
				if znoise is False:
					outmat[matindex:matindex+len(SQx_picked),2]=np.ones(len(qx_pts))*qz_pt 
				else:
					#Repeat same as Qy noise procedure. 
					if j==0:
						lowQz = final_qz[j]- np.abs(final_qz[j+1] -final_qz[j])
						highQz = final_qz[j+1]
					elif j==range(len(final_qz))[-1]:
						lowQz = final_qz[j-1]
						highQz = final_qz[j]+np.abs(final_qz[j]-final_qz[j-1])
					else:
						lowQz = final_qz[j-1]
						highQz = final_qz[j+1]
					orig_Qz = qz_pt
					qz_range = np.abs(highQz-lowQz)
					stddev = qz_range/5.0
					newQzpts = np.random.normal(qz_pt,stddev,len(qx_pts))
					newQzpts[newQzpts>highQz]=highQz 
					newQzpts[newQzpts<lowQz]=lowQz
					outmat[matindex:matindex+len(SQx_picked),2] = newQzpts
				if omeganoise is False:
					outmat[matindex:matindex+len(SQx_picked),3]=np.ones(len(qx_pts))*w_pt
				else:
					if i==0:
						lowE = cross_omegas[i]
						highE = cross_omegas[i+1]
					elif i==range(len(cross_omegas))[-1]:
						lowE = cross_omegas[i-1]
						highE = cross_omegas[i]
					else:
						lowE = cross_omegas[i-1]
						highE = cross_omegas[i+1]
					orig_w = w_pt
					w_range = np.abs(highE-lowE)
					stddev = w_range/5.0
					newEpts = np.random.normal(w_pt,stddev,len(qx_pts))
					newEpts[newEpts>highE]=highE 
					newEpts[newEpts<lowE]=lowE
					outmat[matindex:matindex+len(SQx_picked),3] = newEpts
				if ynoise is False and znoise is False and omeganoise is False:
					outmat[matindex:matindex+len(SQx_picked),4]=SQx_picked
				else:
					#Need to recalculate SQW for new set of points. 
					qx_new_arr = outmat[matindex:matindex+len(SQx_picked),0]
					qy_new_arr = outmat[matindex:matindex+len(SQx_picked),1]
					qz_new_arr = outmat[matindex:matindex+len(SQx_picked),2]
					w_new_arr = outmat[matindex:matindex+len(SQx_picked),3]
					#The actual values of SQW have been accounted for in spectral weight. 
					sqw_new = sqw_f(qx_new_arr,qy_new_arr,qz_new_arr,w_new_arr,*p)
					#Get a normalized probability of this point:

					outmat[matindex:matindex+len(SQx_picked),4]=sqw_new
				matindex+=len(qx_pts)
	#List of S(q,w) points should be ordered by |Q|+(2m/hbar^2)*E/|Q|,
	#and then by |Q| where these are degenerate
	print("All values calculated. ")
	Qvecs_cart = np.around(outmat[:,0],5)*astar_vec[:,np.newaxis] + \
				np.around(outmat[:,1],5)*bstar_vec[:,np.newaxis]+ \
				np.around(outmat[:,2],5)*cstar_vec[:,np.newaxis]
	Qvecs_cart=Qvecs_cart.T
	Qmod = np.linalg.norm(Qvecs_cart,axis=1)
	const = 0.4825966246
	omegas = np.around(outmat[:,3],5)
	chi_vec = (Qmod + (const*omegas)/Qmod)/2.0
	print(f"Sorting outmat of size {np.shape(outmat)}")
	chi_sorti = np.argsort(chi_vec)
	outmat = outmat[chi_sorti]
	chi_vec = chi_vec[chi_sorti]
	Qmod = Qmod[chi_sorti]
	#Now order by |Q| where degenerate, but the overlap should be small between points due to noise. 
	#outmat[:,5]=chi_vec
	#outmat[:,6]=Qmod
	unique_chi = np.unique(chi_vec)
	if (len(unique_chi)<len(outmat) and 1==0):
		#Only need to do this if there are degenerate points. 
		#This is nearly always not the case due to noise.
		for chi in unique_chi:
			like_i = np.where(chi_vec==chi)[0]
			like_chis = chi_vec[like_i]
			like_Qmod = Qmod[like_i]
			#Sort by this. 
			degen_i_sorted = like_i[np.argsort(like_Qmod)]
			#Replace the outmat rows witht the correct order.
			outmat[like_i]=outmat[degen_i_sorted]
	else:
		sorti = np.argsort(chi_vec)
		outmat=outmat[sorti]
	print("Sorted.")
	#Finally, save the file. 
	#Need a header for sample information 
	header = f"	lattice_a {lat_a:.3f}\n\
	lattice_b {lat_b:.3f}\n\
	lattice_c {lat_c:.3f}\n\
	lattice_aa {alpha:.3f}\n\
	lattice_bb {beta:.3f}\n\
	lattice_cc {gamma:.3f}\n\
	temperature 2\n\
	field||a 0\n\
	field||b 0\n\
	field||c 0\n\
	column_h 1\n\
	column_k 2\n\
	column_l 3\n\
	column_E 4\n\
	column_S 5\n\
	h k l En S(q,w)\n"
	print("Saving.")
	np.savetxt(fname,outmat,fmt='%.5f %.5f %.5f %.5f %.5e',header=header)
	print(f"File {fname} saved successfully.")
	return outmat


