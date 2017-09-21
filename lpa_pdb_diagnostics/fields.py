import cPickle as pickle
import numpy as np
from scipy.constants import e, c, m_e
from generics import findRoot, savitzkyGolay, wstd
from file_handling import FileWriting

class FieldInstant():

    """
    A class that handles field instants.
    """
    def __init__( self, filename, laser_pol,
     quantities = ["E", "B", "zfield", "densH", "densN5", "densN6",
     "densN7", "rho", "xfield", "J"], **kwargs ):
        """
        initializes the field instant. The fields contain E- and B-fields

        Parameters:
        -----------
        laser_pol: a numpy float
            laser polarization, 0 if in the x-direction, pi/2 if
            in the y-direction

        filename: string
            add the name of the file if pdb, otherwise add the name of
            the folder such as "./diags/hdf5/" if it's a h5

        quantities: a 1D array
            array of the quantities
        """

        print "** Processing ** Fields: Intialisation of "+str(filename)+ " **"

        self.quantities = quantities
        self.it = None

        for arg in kwargs:
            if arg == "it":
                self.it = kwargs[arg]

        pdb_present = False
        # check for the extension of the file
        if filename.endswith('.pdb'):
            with open( filename ) as pickle_file:
                tmf = pickle.load( pickle_file )
            pdb_present = True

        else:
            from opmd_viewer import OpenPMDTimeSeries
            ts = OpenPMDTimeSeries(filename)
            if self.it is None:
                raise 'Please precise the iteration.'

        for quantity in self.quantities:
            if quantity == "E":
                if pdb_present:
                    self.ex = np.array(tmf["ex"])
                    self.ey = np.array(tmf["ey"])
                    self.ez = np.array(tmf["ez"])

                else:
                    self.ex, self.info_ex = ts.get_field( iteration=self.it,  field='E',
                        coord='x' )
                    self.ey, self.info_ey = ts.get_field( iteration=self.it,  field='E',
                        coord='y' )
                    self.ez, self.info_ez = ts.get_field( iteration=self.it,  field='E',
                        coord='z' )

            if quantity == "zfield":
                if pdb_present:
                    self.zfield = np.array(tmf["z"][:-1])
                else:
                    self.zfield = np.array(self.info_ez.z)

            if quantity == "xfield":
                if pdb_present:
                    self.xfield = np.array(tmf["x"])
                else:
                    self.xfield = np.array(self.info_ez.x)

            if quantity == "B":
                if pdb_present:
                    self.bx = np.array(tmf["bx"])
                    self.by = np.array(tmf["by"])
                    self.bz = np.array(tmf["bz"])
                else:
                    self.bx, self.info_bx = ts.get_field( iteration=self.it,  field='B',
                        coord='x' )
                    self.by, self.info_by = ts.get_field( iteration=self.it,  field='B',
                        coord='y' )
                    self.bz, self.info_bz = ts.get_field( iteration=self.it,  field='B',
                        coord='z' )

            if quantity == "densH":
                if pdb_present:
                    self.dens = np.array(tmf["densH"])

            if quantity == "densN5":
                if pdb_present:
                    self.densN5 = np.array(tmf["dens5"])

            if quantity == "densN6":
                if pdb_present:
                    self.densN6 = np.array(tmf["dens6"])

            if quantity == "densN7":
                if pdb_present:
                    self.densN7 = np.array(tmf["dens7"])

            if quantity == "rho":
                if pdb_present:
                    self.rho = np.array(tmf["rho"])
                else:
                    self.rho, self.info_rho = ts.get_field( iteration=self.it,  field='rho')

            if quantity == "J":
                if pdb_present:
                    self.jx = np.array(tmf["jx"])
                    self.jy = np.array(tmf["jy"])
                    self.jz = np.array(tmf["jz"])
                else:
                    self.jx, self.info_jx = ts.get_field( iteration=self.it,  field='J',
                        coord='x' )
                    self.jy, self.info_jy = ts.get_field( iteration=self.it,  field='J',
                        coord='y' )
                    self.jz, self.info_jz = ts.get_field( iteration=self.it,  field='J',
                        coord='z' )



        # self.extent contains information on the row and column
        row, col = np.shape(self.ez)
        self.shape = [ row, col ]

        if pdb_present:
            self.extent = tmf["extent"]

        else:
            self.extent = self.info_ez.imshow_extent

        if laser_pol == np.pi/2:
            self.laser_field = self.ey

        elif laser_pol == 0:
            self.laser_field = self.ex

    def laser_envelop( self , lwrite = False):
        """
        returns the laser envelop

        Returns:
        --------
        z_envelop : 1D numpy array
            the longitudinal position of the laser pulse

        envelop : 1D numpy array
            the amplitude of the envelop with respect to z_envelop
        """

        # we take only  zfield>0
        index = np.compress(self.zfield >=0 , np.arange(len(self.zfield)))
        zfield = self.zfield[index]
        t_laser_field_1D = self.laser_field[int(self.shape[0]/2),:]
        laser_field_1D = t_laser_field_1D[index]

        # looking for zero crossing of the laser field
        roots = findRoot( laser_field_1D, zfield )
        envelop = []
        z_envelop = []

        # for the sake of the symmetry
        if len(roots)%2 == 0:
            begin_index = 1
        else:
            begin_index = 0

        for i in range ( begin_index, len(roots)-1, 2 ):
            ind = np.compress( np.logical_and(zfield >= roots[i],
             zfield <= roots[ i+1 ]), np.arange(len(zfield)) )
            z_temp = np.take( zfield, ind )
            laser_max = np.take( laser_field_1D, ind )
            ind2 = np.argmax( laser_max )
            z_envelop.append( z_temp[ ind2 ] )
            envelop.append( max( np.absolute( laser_max ) ) )

        if lwrite:
            qname = ["z", "envelop"]
            f = FileWriting( qname , "BeamEnvelop" )
            stacked_data = np.vstack( (z_envelop, envelop) )
            f.write( stacked_data, np.shape(stacked_data) , attrs = [ "m", "V/m" ])

        return (z_envelop, envelop)

    def laser_a0( self, w0):
        """
        returns the laser a0

        Returns:
        --------
        a0_max : float
            the maximum a0 of the laser

        zfield_a0_max : float
            corresponding zfield for a0_max
        """
        # We only care about laser propagating forward, in WARP,
        #the convention is left to right
        index = np.compress( self.zfield>=0, np.arange(len(self.zfield)))
        temp_laser_normalized = self.normalizedField(w0,"laser")
        laser_normalized = temp_laser_normalized[index]
        imax = np.argmax(laser_normalized)

        return (max(laser_normalized), self.zfield[imax])

    def laser_ctau( self ):
        """
        Evaluate c*tau of the laser

        Returns:
        --------
        ctau : float
            laser ctau of the laser
        """
        z_env, env = self.laser_envelop()
        # Calculate standard deviation
        sigma = wstd(z_env, env)
        # Return ctau = sqrt(2) * sigma
        return( np.sqrt(2) * sigma )


    def wakefield_zero_crossing( self ):
        """
        Identify zero crossings on the fields, eventually useful to locate the particles contained in the bucket. Buckets are determined by
        the change of sign of the filtered wakefield.

        Returns:
        --------
        buckets : 2D array.
            Ex: buckets[i] gives you the ith bucket, i=0 signifies
                the first bucket.
                buckets[i][j], j represents either the minimum bound or the
                maximum bound of the ith bucket
        """

        index = np.argmax( self.laser_field[self.shape[0]/2,:] )

        # Savitzky-golay filtering is applied to smooth the Ez-field in order
        # to avoid any abrupt change of sign due to the noise
        ez_filtered = savitzkyGolay(self.ez[self.shape[0]/2,:], 51, 3)
        ez_temp = ez_filtered[ 0:index ]
        z_temp = self.zfield[ 0:index ]
        roots = findRoot( ez_temp, z_temp )

        # Length of root_zero
        lrz = np.shape(roots)[0]

        k=-1
        j=0

        buckets=[[] for i in xrange(lrz/2)]

        for i in range(lrz-1,-1,-1):

            if (j%2)==0:
                j=0
                k+=1
                # we want the minimum bound to be on the right
                i-=1
            else:
                i+=1
            buckets[k].append(roots[i])
            j+=1

        return ( buckets )

    def normalizedField( self, w, field_type ):
        """
        returns the normalized fields.

        Parameters:
        -----------
        laser_field: 1D numpy array
            the laser field near axis

        w: float
            Normalizing factor.

        field_type: string
            indicates if it's "laser" or "wake"

        Returns:
        --------
        normalized_laser: 1D numpy array
            normalized laser field
        """

        if field_type == "laser":
            field = self.laser_field[int(self.shape[0]/2)]

        elif field_type =="wake":
            field = self.ez[int(self.shape[0]/2)]

        return field*e/(m_e*c*w)
