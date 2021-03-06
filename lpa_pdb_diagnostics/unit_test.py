import numpy as np
import pylab as plt
from lpa_pdb_diagnostics import *
from scipy.constants import e, c, m_e, epsilon_0

## Getting the path
dir_path = "//Volumes/Orsay/circ_a01.1_foc1.9mm_400um_dens7.8_nzplambda50/data/"

# Some definitions of the laser plasma parameters
lambda0 = 0.8e-6
w0 = 2*np.pi*c/lambda0
laser_waist = 17e-6
plasma_dens = 7.8e24
plasma_length = 3e-3
wp = np.sqrt(plasma_dens * e**2/ (epsilon_0 * m_e))
lambda_plasma = 2*np.pi*c/wp

#Numerical Parameters
zmin = -8*lambda_plasma
zmax = 2*lambda0
nzplambda = 20
w_size = zmax - zmin
Nz = int(w_size*nzplambda/lambda0)
dz = w_size/Nz

# Generating values for file reading
inf = 0
sup = 0
period_int = 1000
period_ext = 1000

val = values( inf, sup, period_int, period_ext, plasma_length/dz )

field = np.empty(len(val)).astype(str)
N5 = np.empty(len(val)).astype(str)
N6 = np.empty(len(val)).astype(str)
N7 = np.empty(len(val)).astype(str)

# Initialize file names
for i, v in enumerate(val):
    field[ i ] = dir+"fields%06d.pdb" %v
    N5[ i ] = dir+"N5%06d.pdb" %v
    N6[ i ] = dir+"N6%06d.pdb" %v
    N7[ i ] = dir+"N7%06d.pdb" %v
    H[ i ] = dir+"H%06d.pdb" %v

longitudinal_position = val*dz

#

file_data = dir_path + "fields111000.pdb"
file_N6 = dir_path + "N6111000.pdb"
field = FieldInstant(file_data, np.pi/2, quantities= ["E", "zfield", "densH", "densN5", "densN6", "densN6"])
value = field.laser_ctau()
buckets = field.wakefield_zero_crossing()
#N6 = ParticleInstant(file_N6)
#chosen_particles = N6.select(gamma_threshold = [20,200], ROI = buckets[0])
#beam_spectrum(chosen_particles[N6.qdict["gamma"]],chosen_particles[N6.qdict["w"]], lwrite=True)
field.laser_envelop( lwrite = True )
#energy, dQdE  = beam_spectrum( chosen_particles[N6.qdict["gamma"]],  chosen_particles[N6.qdict["w"]] )
#peakInd, energyp, dQdEp = beam_peak( energy, dQdE, peak_width = 40 )
#deltaE, deltaEE = beam_energy_spread( energy, dQdE, peak = (peakInd[1], energyp[1], dQdEp[1] ))
#wy = beam_emittance(chosen_particles[N6.qdict["y"]], chosen_particles[N6.qdict["uy"]], chosen_particles[N6.qdict["w"]])
#plt.scatter(chosen_particles[N6.qdict["z"]][::100], chosen_particles[N6.qdict["uz"]][::100])
#plt.plot(field.zfield,field.ez[int(field.extent[0]/2),:])
#plt.scatter(buckets[0],np.zeros(len(np.array(buckets[0]).flatten())), color="red")
# dplt.scatter(np.zeros(len(buckets)), buckets)
#plt.plot(field.zfield, field.laser_field[int(field.extent[0]/2),:])
#plt.show()
#chosen_particles = N6.select(gamma_threshold = [20,200])
#x = np.arange(100)
#file_array = [ dir_path+"N6108000.pdb", dir_path+"N6109000.pdb", dir_path+"N6110000.pdb", file_data ]
#PT = ParticleTracking(file_array, chosen_particles, NUM_TRACKED_PARTICLES= 5)
#PT.run()

#PT.particle_buffer
#gname = ["Species #" + str(i) for i in xrange(len(PT.particle_buffer[:][0][0])) ]
#qname = ["PID", "Weight", "x", "y", "z", "ux", "uy", "uz", "gamma", "ex", "ey", "ez", "bx", "by", "bz"]
#FH = FileWriting(qname, "Test"  , PT.particle_buffer , groups = gname)
#f = FH.read()
