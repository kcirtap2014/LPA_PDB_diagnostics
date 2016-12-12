import sys
import numpy as np
from generics import randintGenerator
from file_handling import FileWriting
from particles import ParticleInstant

try:
    import pandas as pd
except ImportError:
    print "If you wish to use pandas to manipulate your data, please install \
    the module follwing the instruction on this website \
    http://pandas.pydata.org/pandas-docs/stable/install.html"
    pass

class ParticleTracking():
    """
    Class that traces the evolution of the quantities of the particles. One
    species at a time.
    """

    def __init__(self, file_array, chosen_particles,
                quantities = ["PID", "Weight", "Position", "Momentum", "E", "B"],
                NUM_TRACKED_PARTICLES = None, write_period = 20):

        """
        Initialize the ParticleTracking object.

        Parameters:
        ----------
        Particles: particles object extended from Particles class

        NUM_TRACKED_PARTICLES: int value
            number of particles to be traced.

        file_array: a 1D array of string
            for iteration purpose

        chosen_particles: a 2D numpy array
            contains all information on the particles

        quantities: a 1D array of string
            contain name of quantities to be returned

        write_period: int
            define the dumping period

        """
        self.quantities = quantities
        self.file_array = file_array
        m, n = np.shape(chosen_particles)
        self.num_quantities = m
        self.num_chosen_particles = n
        self.write_period = write_period

        # Prepare the particle quantities for time step N
        if NUM_TRACKED_PARTICLES is not None:
            self.NUM_TRACKED_PARTICLES = NUM_TRACKED_PARTICLES
            randIntArray = randintGenerator(self.NUM_TRACKED_PARTICLES,
                        self.num_chosen_particles)
            transposed_chosen_particles = np.transpose(chosen_particles)[randIntArray]
            self.chosen_particles = sortArray(transposed_chosen_particles)
        else:
            self.NUM_TRACKED_PARTICLES = self.num_chosen_particles
            transposed_chosen_particles = np.transpose(chosen_particles)
            self.chosen_particles = sortArray(transposed_chosen_particles)

        # Initialize FileWriting object
        qname = []

        for quantity in self.quantities:
            if quantity == "PID":
                qname.append("PID")

            if quantity == "Weight":
                qname.append("Weight")

            if quantity == "Position":
                qname.extend(["x", "y", "z"])

            if quantity == "Momentum":
                qname.extend(["ux", "uy", "uz"])

            if quantity == "E":
                qname.extend(["ex", "ey", "ez"])

            if quantity == "B":
                qname.extend(["bx", "by", "bz"])

        gname = np.arange(self.NUM_TRACKED_PARTICLES).astype('str')

        self.FW = FileWriting(qname, "TrParticles", groups = gname)

    def run(self):
        """
        Runs the particle tracing algorithm and save it in a hdf5 file.

        """

        previous_chosen_particles = self.chosen_particles.copy()
        previous_ssn = previous_chosen_particles[:, 0]
        ssn_dict = {} # for indexing purpose
        self.particle_buffer = [[] for i in xrange(self.NUM_TRACKED_PARTICLES)]

        # Populate the dictionary
        for index, ssn in enumerate (previous_ssn):
            ssn_dict[ssn] = index

        # Populate the particle_buffer with particle quantities
        # of the last iteration
        for index in xrange( len(ssn_dict) ):
            # Reshape is done here so that we have information written in this
            # formality:
            # ssn_number | {ssn_number}| {ssn_number} | {ssn_number} | ...
            # x          | {x^{tn}}    | {x^{t(n-1)}} | {x^{t(n-2)}} | ...
            # y          | {y^{tn}}    | {y^{t(n-1)}} | {y^{t(n-2)}} | ...
            # z          | {z^{tn}}    | {z^{t(n-1)}} | {z^{t(n-2)}} | ...
            # .                  .              .              .
            # .                  .              .              .
            # .                  .              .              .

            reshaped_previous_chosen_particles = np.reshape((
                    previous_chosen_particles[index]),
                    (self.num_quantities, 1))
            self.particle_buffer[index] = reshaped_previous_chosen_particles

        # Iteration through all files and all ssnum of the traced particles,
        # look for the index of particles,
        # retrieve the information of that particle,
        # reshape the array to obtain the same dimension as the particle_buffer
        # before appending it to the particle_buffer.
        # Notice that len(previous_ssn) will decrease as we go back in time,
        # this reduces the number of iteration.

        # some indices for the population of dset
        dset_index_start  = 0
        dset_index_stop = 2

        # We start the iteration at before last timestep, N
        print "Running particle tracking algorithm. This may take a while..."

        for ifile, file in enumerate(self.file_array[-2::-1]):
            Ins = Instant( file, self.quantities )
            # Here, we don't filter particles
            chosen_particles = Ins.select()
            # Binary search only works with sorted array
            transposed_chosen_particles = np.transpose(chosen_particles)
            current_chosen_particles = sortArray( transposed_chosen_particles )
            # for bookkeeping purpose
            ssn_list = []

            # Iterate each ssnum
            for i, ssn in enumerate( previous_ssn ) :
                index = binarySearch( ssn, current_chosen_particles[:, 0],
                        0, len(current_chosen_particles[:, 0])  )

                if index != -1:
                    ssn_list.append(ssn)
                    reshaped_current_chosen_particles = np.reshape((
                            current_chosen_particles[index]),
                            (self.num_quantities, 1))

                    if len(self.particle_buffer[ssn_dict[ssn]]) == 0:
                        self.particle_buffer[ssn_dict[ssn]] = reshaped_current_chosen_particles
                    else:
                        self.particle_buffer[ssn_dict[ssn]] = np.hstack((self.particle_buffer[ssn_dict[ssn]],
                    reshaped_current_chosen_particles))

            # Dump the file at a regular interval to avoid having a large
            # particle_buffer
            if (ifile%self.write_period==0 or ifile==len(self.file_array)-2):

                # Call the file writing object to dump data in the hdf5 file
                self.FW.write(self.particle_buffer, np.shape(self.file_array),
                dset_index_start = dset_index_start,
                dset_index_stop = dset_index_stop)

                # Re-indexing of dset_index_start and dset_index_stop
                temp = dset_index_stop

                if ifile == len(self.file_array)-2:
                    dset_index_stop = len(self.file_array)
                else:
                    dset_index_stop += self.write_period

                dset_index_start = temp

                #Empty the buffer once data are dumped
                self.particle_buffer = [[] for i in xrange(
                                                self.NUM_TRACKED_PARTICLES)]

            previous_ssn = ssn_list

def sortArray(unsorted_chosen_particles):
    """
    returns the sorted dictionary according to the ssnum

    Returns:
    --------
    sortedDict : a 2D array
        contains all information on the traced particles sorted according
        to ssnum.
    """

    #We sort the array according to order 0 for ssnum
    sorted_chosen_particles = np.sort(unsorted_chosen_particles, axis=0)

    return sorted_chosen_particles

def binarySearch( key, ssnum, left, right):

    """
    Recursive binary search algorithm

    Parameters:
    -----------
    key : int
        the value that we are looking for in the array

    ssnum : int
        the array where we are looking at

    Returns :
    ---------
    mid_index : int
        index of the key in the ssnum array
    """
    mid_index = int((right + left)/2)

    if left!=right:
        if key == ssnum[mid_index]:
            res = mid_index
        elif key > ssnum[mid_index]:
            res = binarySearch(key, ssnum, mid_index+1, right )
        elif key < ssnum[mid_index]:
            res = binarySearch(key, ssnum, left, mid_index)
    else:
        res = -1

    return res
