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

for root, dirs, files in os.walk('CrI3_scans/'):
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


CrI3_inst = ms.McStas_instr("CrI3_inst")
A3 = CrI3_inst.add_parameter("A3", value=0.0,
                                      comment="Sample A3 Rotation angle.")
Ei = CrI3_inst.add_parameter("Ei", value=30.0,
                                      comment="Incident neutron energy mean.")

vix = CrI3_inst.add_user_var("double", "vix", comment="Initial neutron x velocity")
viy = CrI3_inst.add_user_var("double", "viy", comment="Initial neutron y velocity")
viz = CrI3_inst.add_user_var("double", "viz", comment="Initial neutron z velocity")

progress = CrI3_inst.add_component("progress","Progress_bar")
progress.set_AT([0,0,0])
#Simple neutron source 
source = CrI3_inst.add_component("source", "Source_gen")
source.radius = 0.0775
source.focus_xw = 0.01 
source.focus_yh = 0.01
source.E0 = Ei 
source.dE = 0.2
source.I1 = 1e10
source.verbose=1
source.set_AT([0,0,0])
source.set_ROTATED([0,0,0])
source.append_EXTEND("vix=vx;viy=vy;viz=vz;")

crystal_assembly = CrI3_inst.add_component("crystal_assembly","Arm")
crystal_assembly.set_AT([0,0,2],RELATIVE=source)
crystal_assembly.set_ROTATED([0,A3,0],RELATIVE=source)
#Union processes
init = CrI3_inst.add_component("init","Union_init")

#Sample sqw4 processs
sample_sqw4 = CrI3_inst.add_component("sample_sqw4", "Sqw4_process")
sample_sqw4.sqw = '"spinw_CrI3.sqw4"'
#We are choosing to have the (100) vector along the x-axis
sample_sqw4.ax=6.867*np.sqrt(3)/2.0
sample_sqw4.ay=0
sample_sqw4.az=-6.867*0.5
sample_sqw4.bx=0
sample_sqw4.by=0
sample_sqw4.bz=6.867
sample_sqw4.cx=0
sample_sqw4.cy=19.807
sample_sqw4.cz=0
sample_sqw4.aa=90
sample_sqw4.bb=90
sample_sqw4.cc=120
sample_sqw4.barns=1
sample_sqw4.max_stored_ki=1e5
sample_sqw4.max_bad=1e5
sample_sqw4.stored_dTheta = 0.1
sample_sqw4.stored_dkmag = 1e-4
sample_sqw4.recip_cell=0
sample_sqw4.interact_fraction=-1
#sample_sqw4.init="'init'"
#sample_sqw4.append_EXTEND("// Remove direct beam\nif(!SCATTERED) ABSORB;")
sample_sqw4.set_AT([0,0,0])
sample_sqw4.set_ROTATED([0,0,0])

sample_material = CrI3_inst.add_component("sample_material","Union_make_material")
sample_material.my_absorption=0.0
sample_material.process_string='"sample_sqw4"'

#Actual box representing the crystal. 
sample_box = CrI3_inst.add_component("sample_box", "Union_box", AT=[0.0,0,0], RELATIVE=crystal_assembly)
sample_box.xwidth = 0.01
sample_box.yheight = 0.01
sample_box.zdepth = 0.01
sample_box.material_string = '"sample_material"'
sample_box.priority = 100
sample_box.set_AT([0,0,0],RELATIVE="crystal_assembly")
sample_box.set_ROTATED([0,0,0],RELATIVE="crystal_assembly")


#Union master.
master = CrI3_inst.add_component("master", "Union_master")
master.append_EXTEND("// Remove direct beam\nif(!SCATTERED) ABSORB;")

#Union stop necessary. 
stop = CrI3_inst.add_component("stop", "Union_stop") # This component has to be called stop

#monitor_N version below:
numthetabins = 360
numomegabins = 120
monitor_N =CrI3_inst.add_component("monitor_N","Monitor_nD")
monitor_N.radius=0.3
monitor_N.yheight=0.4
monitor_N.restore_neutron=1
optionsstr = "banana, theta limits=[-90,90], bins="+f"{numthetabins}"+ f", energy limits=[3.0,29.7], bins={numomegabins}"
monitor_N.options='\"'+optionsstr+'\"'
monitor_N.filename='"banana_det_theta_E.dat"'
monitor_N.set_AT([0,0,0.00],RELATIVE="crystal_assembly")
monitor_N.set_ROTATED([0,0,0])

#Sqq_w monitor
sqqw_monitor = CrI3_inst.add_component("sqqw_monitor","Sqq_w_monitor")
sqqw_monitor.radius = 0.25
sqqw_monitor.yheight=0.3
sqqw_monitor.qax = 1
sqqw_monitor.qaz = 0 
sqqw_monitor.qbx = 0
sqqw_monitor.qbz = 1
sqqw_monitor.qamax=3
sqqw_monitor.qbmax=3
sqqw_monitor.qamin=-3
sqqw_monitor.qbmin=-3
sqqw_monitor.Emin=0
sqqw_monitor.Emax=20
sqqw_monitor.nqa = 101
sqqw_monitor.nqb = 101
sqqw_monitor.nE = 60
sqqw_monitor.filename = '"qa_vs_qb"'
sqqw_monitor.vix = '"vix"'
sqqw_monitor.viy = '"viy"'
sqqw_monitor.viz = '"viz"'
sqqw_monitor.set_AT([0,0,0], RELATIVE="crystal_assembly")
sqqw_monitor.set_ROTATED([0,0,0])

numA3=300
A3_list = np.linspace(-120,120,numA3)
#A3_list=np.array([0])
#Preallocate output array, cols are H,K,E,I,Err,N
npts = numA3*numthetabins*numomegabins
Outmat = np.zeros((npts,6))

matind = 0 
n_threads = 8

#We won't project / plot in this script, just run the scans. 
#Start by compiling the instrument, then vary the params. 
CrI3_inst.settings(ncount=100,force_compile=True,output_path=f"CrI3_scans/mcscript_test_monitorND_A3_0",
    checks=False)
CrI3_inst.set_parameters(A3=0,Ei=30.0)
dat = CrI3_inst.backengine()
#Clear out the directory. 
for root, dirs, files in os.walk('CrI3_scans/'):
    for f in files:
        os.unlink(os.path.join(root, f))
    for d in dirs:
        shutil.rmtree(os.path.join(root, d))

#Define a function to parallelize. 
def run_scan(instrument, A3):
    print(f"Running A3={A3:.3f}")
    inst_copy = instrument# copy.deepcopy(instrument)
    inst_copy.settings(ncount=1e5,force_compile=False,output_path=f"CrI3_scans/mcscript_test_monitorND_A3_{A3:.3f}deg".replace('.','p').replace('-','m'),
        checks=False)
    inst_copy.set_parameters(A3=A3,Ei=30.0)
    dat = inst_copy.backengine()
    return dat

datlist = Parallel(n_jobs = n_threads, backend="threading")(delayed(run_scan)(CrI3_inst,A3) for A3 in A3_list)
#dat = run_scan(CrI3_inst,0.0)