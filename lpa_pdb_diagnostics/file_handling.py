import os
import h5py
import numpy as np
import result_path as rp
import pdb

class FileWriting():
    """
    A file handling class for writing data onto disk in h5py.
    All data will be saved in the result directory.
    """
    def __init__( self, dname, file_name, groups = None ):
        """
        Initialize the FileHandling object

        Parameters:
        -----------
        group: a 1D array
            Groups are the container mechanism by which HDF5 files are organized.
            From a Python perspective, they operate somewhat like dictionaries.
            For instance in saving particles, that would be the particle number

        dname : a 1D array
            The quantity name

        file_name : string
            Name of the file wish to be created

        data : data wish to be saved

        size : size of the data
        """

        dir_path = rp.ResultPath()
        self.result_path = dir_path.result_path
        self.dname = dname
        self.groups = groups
        self.create_file( file_name )

        print "A file named %s.hdf5 in %s is created." %(file_name,
              self.result_path)

    def create_file( self, file_name ):
        """
        Creates the file objects

        Parameters:
        -----------
        groups: path string
            Groups are the container mechanism by which HDF5 files are organized.
            From a Python perspective, they operate somewhat like dictionaries.

        file_name: string
            Name of the file to be created
        """

        self.fname = file_name + ".hdf5"
        self.file_object = h5py.File( self.result_path + self.fname , 'w')

        if self.groups is not None:
            self.group_object = []
            for group in self.groups:
                self.group_object.append(self.file_object.create_group(str(group)))

    def write( self, data, size, dset_index_start = None
    , dset_index_stop = None, attrs = None):
        """
        Dump data into the file

        Parameters:
        -----------
        data : data objects
            data to be written in file

        size: tuple
            shape of the data

        dset_index_start : int
            indicates the index where the data should start writing

        dset_index_stop: int
            indicates the index where the data should stop writing
        """

        if len(attrs)!= len(self.dname):
            raise "len(attrs) has to be the same as len(dname)."

        if self.groups is not None:
            for indexg in xrange(len(self.groups)):
                for indexq, quantity in enumerate(self.dname):
                    #print "size of dset", dset
                    #print "size of data", np.shape(data[indexg][indexq][:])
                    #print "current dset size", np.shape(dset[dset_index_start:dset_index_stop])
                    if (dset_index_start is not None) and (dset_index_stop is not None):
                        dset = self.group_object[indexg].require_dataset(quantity,
                                size, dtype = float)
                        dset[dset_index_start:dset_index_stop] = data[indexg][indexq][:]

                    else:
                        dset = self.group_object[indexg].require_dataset(quantity,
                                np.shape(data[indexg][indexq]), dtype = float)
                        dset[:] = data[indexg][indexq][:]

                    if attrs is not None and dset_index_start == 0:
                        dset.attrs["units"] = attrs[indexq]
        else:
            for indexq, quantity in enumerate(self.dname):
                dset = self.file_object.require_dataset( quantity,
                            np.shape(data[indexq]), dtype = float)

                if attrs is not None:
                    dset.attrs["units"] = attrs[indexq]

                dset[:] = data[indexq]

    def read ( self ):

        f = h5py.File(self.result_path + self.fname, 'r')

        return f

class FileReading():

    def __init__( self, file_name ):

        """
        Initialize FileReading object

        Parameters:
        -----------
        file_name : string
            name of the file

        """

        current_path = os.getcwd() + "/result"
        self.file_path = current_path + "/"+ file_name

    def read( self ):
        """
        Read HDF5 file

        Returns:
        --------
        f : file object
            File object to be manipulated
        """

        f = h5py.File(self.file_path,'r')

        return f
