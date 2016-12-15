import cPickle as pickle
import numpy as np
import math
import sys
from scipy.constants import e
from scipy.signal import find_peaks_cwt
from generics import gamma2Energy, leftRightFWHM, \
                     bilinearInterpolation, w2charge
import config
from pylab import plt
from file_handling import FileWriting
import matplotlib
import cubehelix

try:
    import pandas as pd
except ImportError:
    print "If you wish to use pandas to manipulate your data,\
    please install the module follwing the instruction on this \
    website http://pandas.pydata.org/pandas-docs/stable/install.html"
    pass

class ParticleInstant():

    def __init__(self, filename,
            quantities = ["PID", "Weight", "Position", "Momentum", "E", "B"]):
        """
        Initialize an instant of particles.

        Parameters:
        -----------

        file: a string
            Name of the file including the path
        quantities: a 1D array
            Specify the particle quantities to be initialized

        """

        print "** Processing ** Particles: Initialisation of "\
                +str(filename)+" **"

        with open( filename ) as pickle_file:
            tmp = pickle.load( pickle_file )

        self.quantities = quantities
        self.num_quantities = 0
        self.qdict = {}

        self.pandas = False
        if "pandas" in sys.modules.keys():
            self.pandas = True
            frame = []

        for quantity in self.quantities:
            if quantity == "PID":
                self.ssn = np.array(tmp["ssnum"]).astype(int)
                self.qdict["PID"] = self.num_quantities
                self.num_quantities += 1
                if self.pandas:
                    PID = pd.DataFrame({"PID": self.ssn})
                    frame.append(PID)

            if quantity == "Weight":
                self.w = np.array(tmp["w"])#*tmp(sw)
                self.qdict["w"] = self.num_quantities
                self.num_quantities += 1
                if self.pandas:
                    w = pd.DataFrame({"Weight": self.w})
                    frame.append(w)

            if quantity == "Position":
                self.x = np.array(tmp["x"])
                self.y = np.array(tmp["y"])
                self.z = np.array(tmp["z"])

                self.qdict["x"] = self.num_quantities
                self.qdict["y"] = self.num_quantities + 1
                self.qdict["z"] = self.num_quantities + 2
                self.num_quantities += 3

                if self.pandas:
                    pos = pd.DataFrame({"x": self.x,
                                        "y": self.y,
                                        "z": self.z})
                    frame.append(pos)

            if quantity == "Momentum":
                self.ux = np.array(tmp["ux"])
                self.uy = np.array(tmp["uy"])
                self.uz = np.array(tmp["uz"])
                self.gamma = np.sqrt(1. + self.ux**2 + self.uy**2 + self.uz**2)

                self.qdict["ux"] = self.num_quantities
                self.qdict["uy"] = self.num_quantities + 1
                self.qdict["uz"] = self.num_quantities + 2
                self.qdict["gamma"] = self.num_quantities + 3
                self.num_quantities += 4

                if self.pandas:
                    momentum = pd.DataFrame({"ux": self.ux,
                                        "uy": self.uy,
                                        "uz": self.uz,
                                        "gamma": self.gamma})
                    frame.append(momentum)

            if quantity == "E":
                self.ex = np.array(tmp["ex"])
                self.ey = np.array(tmp["ey"])
                self.ez = np.array(tmp["ez"])

                self.qdict["ex"] = self.num_quantities
                self.qdict["ey"] = self.num_quantities + 1
                self.qdict["ez"] = self.num_quantities + 2
                self.num_quantities += 3

                if self.pandas:
                    Efield = pd.DataFrame({"ex": self.ex,
                                        "ey": self.ey,
                                        "ez": self.ez})
                    frame.append(Efield)

            if quantity == "B":
                self.bx = np.array(tmp["bx"])
                self.by = np.array(tmp["by"])
                self.bz = np.array(tmp["bz"])

                self.qdict["bx"] = self.num_quantities
                self.qdict["by"] = self.num_quantities + 1
                self.qdict["bz"] = self.num_quantities + 2
                self.num_quantities += 3

                if self.pandas:
                    Bfield = pd.DataFrame({"bx": self.bx,
                                        "by": self.by,
                                        "bz": self.bz})
                    frame.append(Bfield)

        if self.pandas:
            self.df = pd.concat(frame, axis = 1 )

    def get_qdict( self ):
        """
        returns qdict for reference purpose
        """

        return self.qdict

    def select( self, gamma_threshold = None, ROI = None ):
        """
        This method selects the particles according to the energy and
        region of interest
        Example of usage:
            Instant.filter(gamma = [50, 100], ROI = [2000e-6, 3000e-6])
            Particles that meet these cited conditions will be chosen.

        Parameters:
        -----------
        gamma: a 1D array of shape 2
            consists of the lower bound and the upper bound in gamma
        ROI : a 1D array of shape 2
            consists of the lower bound and the upper bound in terms of
            position

        Returns:
        --------
        selected: a dictionary of quantities
            Quantities of the chosen particles
        """
        #Check if there's any particle
        try:
            indexListGamma = set()
            indexListROI = set()
            n_array = np.arange(len(self.w))

            if gamma_threshold:
                #Test the gamma_threshold structure
                countValidAgg = 0

                if len(gamma_threshold)!=2:
                    raise "gamma_threshold should be a 1D array of size 2."

                for agg in gamma_threshold:
                    if agg is not None:
                        countValidAgg+=1

                if countValidAgg == 1:
                    indexListGamma = set( np.compress(
                                    self.gamma > gamma_threshold[0], n_array) )

                else:
                    indexListGamma = set(np.compress(
                    np.logical_and(self.gamma>gamma_threshold[0],
                    self.gamma<gamma_threshold[1]), n_array))

            if ROI:
                #Test the ROI structure
                countValidAgg = 0

                if len(ROI)!=2:
                    raise "ROI should be a 1D array of size at least 1."

                for agg in ROI:
                    if agg is not None:
                        countValidAgg+=1

                if countValidAgg == 1:
                    indexListROI = set(np.compress(self.z>ROI[0],
                    n_array))

                else:
                    indexListROI = set(np.compress(
                    np.logical_and(self.z>ROI[0],
                    self.z<ROI[1]), n_array))

            if gamma_threshold and ROI:
                indexList = list(indexListGamma.intersection(indexListROI))
            elif gamma_threshold:
                indexList = list(indexListGamma)
            elif ROI:
                indexList = list(indexListROI)
            else:
                indexList = list(n_array)

            return self.filterwithIndexList( indexList )

        except ValueError:
            print "No particles are detected"
            return

    def filterwithIndexList(self, indexList):
        """
        Return a dictionary of selected particles.

        Parameters:
        -----------
        indexList: a 1D array
            contains the index of chosen particles

        Returns:
        --------
        chosenParticles: a 2D Numpy Array
            contains the quantity of chosen particles

        """
        chosenParticles = [[] for i in xrange(self.num_quantities)]
        index = 0

        for quantity in self.quantities:
            if quantity == "PID":
                chosenParticles[index] = self.ssn[indexList].astype(int)
                index += 1

            if quantity == "Weight":
                chosenParticles[index] = self.w[indexList]
                index += 1

            if quantity == "Position":
                chosenParticles[index ] = self.x[indexList]
                chosenParticles[index + 1] = self.y[indexList]
                chosenParticles[index + 2] = self.z[indexList]
                index += 3

            if quantity == "Momentum":
                chosenParticles[index ] = self.ux[indexList]
                chosenParticles[index + 1] = self.uy[indexList]
                chosenParticles[index + 2] = self.uz[indexList]
                chosenParticles[index + 3] = self.gamma[indexList]
                index += 4

            if quantity == "E":
                chosenParticles[index ] = self.ex[indexList]
                chosenParticles[index + 1] = self.ey[indexList]
                chosenParticles[index + 2] = self.ez[indexList]
                index += 3

            if quantity == "B":
                chosenParticles[index ] = self.bx[indexList]
                chosenParticles[index + 1] = self.by[indexList]
                chosenParticles[index + 2] = self.bz[indexList]
                index += 3

        return chosenParticles

def beam_charge( w ):
    """
    Returns the beam charge of the chosen particles.

    Parameters:
    -----------
    w : 1D array
        Weight of particles

    Returns:
    --------
    Charge: float value (in Coulomb)
    """
    try:
        s_charge = w2charge(w)
        charge = np.sum(s_charge)
    except ValueError:
        charge = 0.0

    return charge

def beam_spectrum( frame_num, gamma, w, lwrite = False,
                    bin_size = 0.5, density = False, lsavefig = True,
                    leg = None):
    """
    Returns the beam spectrum of the chosen particles.

    Parameters:
    -----------
    frame_num: int
        frame_num for writing purpose

    gamma: 2D numpy array
        gamma of particles for different species

    w: 2D numpy array
        Weight of particles for different species

    bin_size: int
        in MeV, default value 0.5 MeV

    density: boolean
        whether to normalize the spectrum, default value False

    lsavefig: boolean
        flag to save figure or not

    legend: 1D string list
        legend for the figure

    Returns:
    --------
    energy: float value
        binned energy in MeV

    dQdE: float value
        charge in Coulomb/MeV
    """
    try:
        num_species = len(gamma)
        energy = []
        dQdE = []

        for index in xrange(num_species):
            en = gamma2Energy(gamma[index])
            bins = int((np.max(en) - np.min(en))/bin_size)
            temp_dQdE, temp_energy = np.histogram( en, bins = bins,
                        weights = w[index] , density = density)
            temp_dQdE *= e
            temp_energy = np.delete( temp_energy, 0 ) #removing the first element
            dQdE.append(temp_dQdE)
            energy.append(temp_energy)

        index_largest_dynamics = np.argmax( map(lambda x: np.max(x) - np.min(x),
                                 energy))
        ref_energy = energy[index_largest_dynamics]

        dQdE_interp = []

        for index in xrange(num_species):
            dQdE_interp.append(np.interp(ref_energy, energy[index], dQdE[index]))

        #this is to store the contribution of all species to the spectrum
        dQdE.append(np.sum( dQdE_interp[0:num_species], axis = 0 ))
        energy.append(ref_energy)

        if 'inline' in matplotlib.get_backend():
            fig, ax = plt.subplots(dpi=150)
        else:
            fig, ax = plt.subplots( dpi = 500 )

        fig.patch.set_facecolor('white')
        c = [ "blue", "red", "black", "green", "magenta" ]

        for i in xrange(num_species + 1):
            if leg is not None:
                ax.plot( energy[i], dQdE[i], color = c[i%(num_species + 1)],
                label = leg[i], linewidth = 2)
            else:
                ax.plot( energy[i], dQdE[i], color = c[i%(num_species + 1)],
                linewidth = 2 )

        ax.set_xlabel(r"$\mathrm{Energy\,(MeV)}$")
        ax.set_ylabel(r"$\mathrm{dQdE\,(C/MeV)}$")
        ax.xaxis.set_tick_params(width=2, length = 8)
        ax.yaxis.set_tick_params(width=2, length = 8)
        font = {'family':'sans-serif'}
        plt.rc('font', **font)

        if leg is not None:
            # Now add the legend with some customizations.
            legend = plt.legend(loc='best', shadow=True)

            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width

        if lwrite:
            gname = np.arange(num_species + 1).astype('str')
            qname = ["energy", "dQdE"]
            f = FileWriting( qname , "beam_spectrum_%d" %frame_num ,
                            groups = gname)
            stacked_data = np.stack( (energy, dQdE), axis = 1 )
            f.write( stacked_data, np.shape(stacked_data),
                    attrs = [ "MeV", "C" ])

        if lsavefig:
            fig.savefig( config.result_path + "beam_spectrum_%d.png" %frame_num)

    except ValueError:
        print "Check if the particle arrays are empty."
        energy = []
        dQdE = []

    return energy, dQdE

def beam_peak( energy, dQdE, peak_width = 50.0 ):
    """
    returns the index of a peak from a beam spectrum

    Parameters:
    -----------
    energy: a 1D numpy array
        binned energy

    dQdE: a 1D numpy array
        charge density from the beam spectrum

    peak_width: int
        estimated width of the peak, default value 20 MeV

    Returns:
    --------
    peakInd: 1D array
        Indices of the located peak(s)

    energy: 1D array
        Energy of the located peak(s)

    dQdE: 1D array
        Charge density of the located peak(s)

    """
    try:
        bin_size = energy[1] - energy[0]
        num_bin_peak_width = int(peak_width/bin_size)
        peakInd = find_peaks_cwt(dQdE, np.arange(1,num_bin_peak_width))

        return peakInd, energy[peakInd], dQdE[peakInd]

    except ValueError:
        print "No peak is found"

def beam_energy_spread( energy, dQdE, lfwhm = True, peak = None ):
    """
    Calculate the energy spread of the beam.

    Parameters:
    -----------
    energy: 1D array
        binned energy from the spectrum

    dQdE: 1D array
        binned energy from the spectrum

    lfwhm: boolean
        calculate either using FWHM or rsm definition. Default value True

    peak: tuple
        contains peakInd, energy[peakInd], dQdE[peakInd], corresponding to the
        output of beam_peak(...). Default peak None

    Returns:
    --------
    deltaE: float
        Energy spread in MeV
    deltaEE: float
        Energy spread in decimal value

    """
    try:
        if peak is None:
            average_energy = np.average( energy, weights = dQdE )
            variance = np.average( (energy - average_energy)**2, weights = dQdE )
            deltaE = np.sqrt( variance )
            deltaEE = deltaE/average_energy

        else:
            peakInd = peak[0]
            Epeak = peak[1]
            dQdEpeak = peak[2]
            y = np.std( dQdEpeak ) #y in RMS

            if lfwhm:
                y = 0.5*dQdEpeak #y in FWHM

            yleft = dQdE[ 0:peakInd ]
            xleft = energy[ 0:peakInd ]
            yright = dQdE[ peakInd+1:: ]
            xright = energy[ peakInd+1:: ]
            yleftleft, yleftright, yrightleft, yrightright,\
            xleftleft, xleftright, xrightleft, xrightright = \
            leftRightFWHM( yleft, yright, y, xleft, xright )

            #Bilinear interpolation for the left area
            x_left_exact = bilinearInterpolation( yleftleft, yleftright,
            xleftleft, xleftright, y)

            #Bilinear interpolation for the right area
            x_right_exact = bilinearInterpolation( yrightleft, yrightright,
            xrightleft, xrightright, y)

            #Evaluate energy spread, delta E (in MeV)
            deltaE = np.absolute( x_left_exact - x_right_exact )

            #Evaluate energy spread, delta E/Epeak (in decimal)
            deltaEE = deltaE/Epeak

    except ZeroDivisionError:
        print "Error in energy spread calculation"
        deltaE = None
        deltaEE = None

    return deltaE, deltaEE

def sorted_by_gamma_beam_emittance ( frame_num, chosen_particles, qdict,
                                    direction, num_bins = None, lwrite = True,
                                    lplot = False, lsavefig = False ):
    """
    runs an analysis on beam emittance with respect to gamma

    Paramaters:
    -----------
    num_frame: int
        frame number, for writing purpose

    chosen_particles: ndarray
        consists of quantities of selected particles

    qdict: dict
        dictionary that contains the correspondance to the array
        "chosen particles" indices

    direction: string
        transverse directions. Can be either "x" or "y"

    num_bins: int
        This is an argument related to sort_by_gamma option, it indicates the
        number of bins that the analysis should used. If none provided, num_bins
        will be calculated by taking into account the dynamic of gamma.
        Default: None

    Returns:
    --------
    mid_bin: 1D numpy array
        an array representing the binned gamma

    emit: 1D numpy array
        an array representing the emittance for each binned gamma

    lplot: boolean
        Plot figure if True. Default: False

    lsavefig: boolean
        Save figure of the emittance distribution if True. Default: False

    lwrite: boolean
        Save data of the emittance distribution if True. Default: False
    """

    try:

        gamma = chosen_particles[qdict["gamma"]]

        # By default, gamma = 2 in a bin
        num_bins = int( (np.max(gamma) - np.min(gamma))/2 )

        #Reconstruct the step
        steps = np.linspace(np.min(gamma), np.max(gamma), num = num_bins)
        mid_bin = (steps[0:-1] + steps[1:])/2
        bin_shape = np.shape(mid_bin)[0]

        #Initialize an empty array of emittance
        emit = np.empty( bin_shape )

        ##Binning the gamma and analyze the emittance for each bin
        for b in xrange( bin_shape ):
            index = np.compress( np.logical_and(gamma >= steps[b],
                gamma < steps[b+1]), np.arange(len(gamma)))
            bin_chosen_particles = np.take(chosen_particles, index, axis=1)
            emit[b] = beam_emittance( frame_num, bin_chosen_particles,
                                      qdict, direction )

        if lwrite:
            qname = ["gamma", "emittance"]
            f = FileWriting( qname , "sorted_beam_emittance_%s_%d" \
                            %(direction, frame_num ))
            stacked_data = np.stack( (mid_bin, emit), axis = 0 )
            f.write( stacked_data, np.shape(stacked_data) ,
                    attrs = [ "arb. units", "m.rad" ])

        if lplot:
            if 'inline' in matplotlib.get_backend():
                fig, ax = plt.subplots(dpi=150)
            else:
                fig, ax = plt.subplots( dpi = 500 )

            fig.patch.set_facecolor('white')

            ax.plot( mid_bin, emit*1e6, linewidth = 2 )

            ax.set_xlabel(r"$\mathrm{\gamma\,(arb.\, unit)}$")
            ax.set_ylabel(r"$\mathrm{\epsilon_{norm.}\,(mm.\,mrad)}$")
            ax.xaxis.set_tick_params(width=2, length = 8)
            ax.yaxis.set_tick_params(width=2, length = 8)
            ax.set_xlim(0.9*np.min(mid_bin), 1.1*np.max(mid_bin))
            font = {'family':'sans-serif'}
            plt.rc('font', **font)

            if lsavefig:
                fig.savefig( config.result_path + \
                            "sorted_beam_emittance_%s_%d.png" \
                            %(direction, frame_num ))

        if not lplot and lsavefig:
            print "Sorry, no plot, no save."

    except ValueError:
        print "Sorted by gamma beam emittance: "+ \
               "Analysis cannot be done because particles are not detected. "
        mid_bin = None
        emit = None

    return mid_bin, emit

def emittance_1D ( x, ux, w ):
    """
    returns the 1D histogam of both axes of the phase space plot, position and
    momentum, the values n_x and n_ux are normalized.

    Parameters:
    -----------
    x: 1D numpy array
        position of the beam

    ux: 1D numpy array
        momentum of the beam

    w: 1D numpy array
        weight of the beam

    Returns:
    --------
    n_x: 1D numpy_array
        the y - coordinates of the beam position

    bin_x: 1D numpy array
        the x - coordinates of the beam position

    n_ux: 1D numpy array
        the y - coordinates of the beam momentum

    bin_ux: 1D numpy array
        the y - coordinates of the beam momentum
    """

    num_bin = 1000
    n_x, bin_x = np.histogram( x, bins = num_bin,
                                weights = w, density = True)
    n_ux, bin_ux = np.histogram( ux, bins = num_bin,
                                weights = w, density = True )
    bin_x = np.delete( bin_x, -1 )
    bin_ux = np.delete( bin_ux, -1 )

    #Normalization of 1D emittances
    n_x *= -np.min(bin_ux)/np.max(n_x)
    n_ux *= np.min(bin_x)/np.max(n_ux)
    n_x += np.min(ux)
    n_ux += np.max(x)

    return n_x, bin_x, n_ux, bin_ux

def beam_emittance( frame_num, chosen_particles, qdict, direction,
                    histogram = False, num_bins = None, lplot = False,
                    lsavefig = False, lwrite = False ):
    """
    Calculation on emittance based on statistical approach in J. Buon (LAL)
    Beam phase space and Emittance. We first calculate the covariance, and
    the emittance is epsilon=sqrt(det(covariance matrix)).
    Covariance is calculated with the weighted variance based on
    http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

    Parameters:
    -----------
    chosen_particles: ndarray
        consists of quantities of selected particles

    qdict: dict
        dictionary that contains the correspondance to the array
        "chosen particles" indices

    direction: string
        transverse directions. Can be either "x" or "y"

    lsavefig: boolean
        Save figure of the emittance distribution if True. Default: False

    lwrite: boolean
        Save data of the emittance distribution if True. Default: False

    num_bins: int
        bin number for the histogram plot of the phase space plot. Default None.

    Returns:
    --------
    weighted_emittance: float
        beam emittance
    """

    try:
        #do analysis according to the direction
        if direction == "x":
            x = chosen_particles[qdict["x"]]
            ux = chosen_particles[qdict["ux"]]

        elif direction =="y":
            x = chosen_particles[qdict["y"]]
            ux = chosen_particles[qdict["uy"]]

        else:
            raise "Invalid direction"

        w = chosen_particles[qdict["w"]]

        w_x = np.mean(x)
        w_ux = np.mean(ux)
        xux = np.sum(x*ux)/(np.shape(x)[0])
        variance_x = np.var(x)
        variance_ux = np.var(ux)
        covariance_xux = xux - w_x*w_ux
        xuxw = [[variance_x,covariance_xux],[covariance_xux,variance_ux]]
        weighted_emittance = np.sqrt(np.linalg.det(xuxw))

        if histogram:
            if num_bins == None:
                num_bins = int(1e6* (np.max(ux) - np.min(ux))\
                                    *(np.max(x) - np.min(x)))

            H, xedges, yedges = np.histogram2d( ux, x,
                                              bins = num_bins, weights = w)
            H = np.rot90(H)
            H = np.flipud(H)
            Hmasked = np.ma.masked_where( H == 0, H )
            Hnorm = Hmasked/np.amax(np.amax( Hmasked ))
            extent = np.array([ np.min(x), np.max(x), np.min(ux), np.max(ux) ])

            # 1D emittances
            #n_x, bin_x, n_ux, bin_ux = emittance_1D ( x, ux, w )

            if lwrite:
                #Writing histogram data
                qname_particle = [ "emittance" , "extent" ]
                fh = FileWriting( qname_particle , "Histogram_emittance_%s_%d" \
                                    %( direction,frame_num ))
                data = [Hnorm.data] + [extent]
                fh.write( data, np.shape(data) , attrs = ["m", "m_e*c"])

                #Writing 1D data

                #position_data = np.stack((bin_x, n_x), axis = 0)
                #momentum_data = np.stack((bin_ux, n_ux), axis = 0)
                #qname_particle = [ "x" , "y" ]
                #fp = FileWriting( qname_particle , "1D_emittance_position_%s_%d" \
                #                    %( direction,frame_num ))
                #fp.write( position_data, np.shape(position_data),
                #        attrs = ["m", "arb.units"])
                #fm = FileWriting( qname_particle , "1D_emittance_momentum_%s_%d" \
                #                    %( direction,frame_num ))
                #fm.write( momentum_data, np.shape(momentum_data),
                #        attrs = ["m_e*c", "arb.units"])

            if lplot:
                if 'inline' in matplotlib.get_backend():
                    fig, ax = plt.subplots( dpi=150 )
                else:
                    fig, ax = plt.subplots( dpi=500 )

                fig.patch.set_facecolor('white')
                cm_peak = cubehelix.cmap( rot = -0.8,  reverse = True )
                plot_extent = [ np.min(x)*1e6, np.max(x)*1e6,
                                np.min(ux), np.max(ux) ]
                sc_peak = (ax.imshow( Hnorm, extent = plot_extent,
                            aspect= "auto", interpolation ='nearest',
                            origin ='lower', cmap = cm_peak))
                colorbar_pos_peak = fig.add_axes([0.9, 0.12, 0.025, 0.78])
                fig.colorbar(sc_peak, cax = colorbar_pos_peak,
                             orientation='vertical')
                #ax.plot(bin_x*1e6, n_x, color="blue",linewidth=2)
                #ax.plot(n_ux, bin_ux, color="blue", linewidth=2)
                ax.set_xlabel(r"$\mathrm{%s\,(\mu m)}$" %direction)
                ax.set_ylabel(r"$\mathrm{p_%s\,(m_{e}c)}$" %direction)
                ax.xaxis.set_tick_params(width=2, length = 8)
                ax.yaxis.set_tick_params(width=2, length = 8)
                font = {'family':'sans-serif'}
                plt.rc('font', **font)

                if lsavefig:
                    fig.savefig( config.result_path + \
                                 "emittance_distribution_%s_%d.png" \
                                 %( direction, frame_num ))

            if not lplot and lsavefig:
                print "Sorry, no plot, no save."

        if math.isnan(weighted_emittance):
            weighted_emittance = 0.0

    except ValueError:
        print "Beam emittance: Analysis is not performed because" + \
              "no particles are detected."
        weighted_emittance = 0.0

    return weighted_emittance
