import random
from scipy.constants import e
import math
import numpy as np
from file_handling import FileWriting
import pylab as plt
import config
import pdb
import matplotlib

def quant_concatenate ( array_obj_quant , keep_object_name = False ):

    """
    returns a concatenation of object quantities. Typically used before
    beam analysis  to take into account all particles regardless of their
    species.

    Paramters:
    ----------
    array_obj_quant: ndarray
        an array of objects

    keep_object_name: boolean
        either to keep the origin of the partlcles or not

    Returns:
    --------
    c_object: ndarray
        concatenated object quantities
    """

    c_object = [[] for i in xrange(np.shape(array_obj_quant)[1])]
    #Loop in quantities
    for iquant in xrange(np.shape(array_obj_quant)[1]):
        if not keep_object_name:
            temp = []
        #Loop in number of species
        for index, obj in enumerate(array_obj_quant):
            if keep_object_name:
                if index==0:
                    temp = [obj[iquant]]
                else:
                    temp += [obj[iquant]]
            else:

                temp = np.concatenate((temp,  obj[iquant]), axis=0)
        c_object[iquant] = temp

    return c_object

def ALSS_baseline ( y, lmbda, p ):
    """
    This method is not totally tested yet.
    performs baseline correction with asymmetric least squares smoothing (ALSS).

    Parameters:
    -----------
    y: 1D numpy array
        an array of signal

    lmbda: float value
        parameter for smoothness. Bes to have values in [10^2:10^9]

    p: float value
        parameter for assymetry. Best to have values in [0.001:0.1]

    Returns
    -------
    y_res: 1D numpy array
        an array of signal with its baseline subtracted
    """

    # an approximate solution
    m = len(y)
    diff2 = np.diff( np.eye(m), n=2 )
    weight = np.ones( m ) #weight

    for i in xrange(4):
        weight = np.diag( weight )
        y_array = np.array( y )
        factor =  np.linalg.cholesky(weight + \
                    lmbda*np.dot( diff2, np.transpose(diff2) ))

        z = np.diag(np.linalg.inv(np.dot(factor,
            np.transpose(factor)))*np.dot(weight , y_array))

        z_array = np.array( z )
        weight = p*( y_array>z_array ) + (1 - p)*( y_array<z_array )

    return z

def peak_indexes(y, thres=0.3, min_dist=1):
    """Peak detection routine.

    Finds the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the indexes of the peaks that were detected
    """

    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)

    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        # only take position that is False in the rem array
        peaks = np.arange(y.size)[~rem]

    return peaks

def ROI_by_peak( y, x, xleft, xright, epsilon , plot_ROI_search ):
    """
    returns the ROI that defines the peak based on the variation of the gradient
    of the signal.

    Parameters:
    -----------
    y: 1D numpy array
        y-coordinate of the filtered signal with savitzkyGolay (preferably)

    x: 1D numpy array
        x-coordiante of the signal

    xleft: float
        value of x at FWHM on the LHS of the peak

    xright: float
        value of x at FWHM on the RHS of the peak

    epsilon: float
        value of tolerance. A parameter of control.

    plot_ROI_search: boolean
        either to show how the algorithm search for the acceptable epsilon.

    Returns:
    --------
    xleft: float
        x-coordinate value that indicates the inferior limit of the peak

    xright: float
        x-coordinate value that indicates the superior limit of the peak
    """

    gradient = np.gradient( y ) #calculate the gradient
    abs_norm_grad = np.absolute( gradient/np.max( y ) )
    indexleft = np.array([])
    indexright = np.array([])
    # if indexleft and indexright both have no element, reduce the epsilon and
    # keep looking

    while not(indexleft.size > 0 and indexright.size) > 0:
        indexleft =  np.compress( np.logical_and( x < xleft,
                                abs_norm_grad < epsilon), np.arange(len(x)) )
        indexright =  np.compress( np.logical_and(x > xright,
                                abs_norm_grad < epsilon), np.arange(len(x)) )
        epsilon*=5

    x_interval = [x[indexleft[-1]], x[indexright[0]]]

    if plot_ROI_search: #this plot is for illustrative purpose
        lab = [r"$|diff|$", r"$\epsilon$"]

        if 'inline' in matplotlib.get_backend():
            fig, ax = plt.subplots(dpi=150)
        else:
            fig, ax = plt.subplots( figsize=(10,8) )

        fig.patch.set_facecolor('white')
        ax.semilogy( x, abs_norm_grad, linewidth = 2, label = lab[0] )
        ax.semilogy( x, epsilon*np.ones(len(x)), linewidth = 2,
                    label = lab[1], linestyle = "-." )
        ax.set_xlabel(r"$\mathrm{Energy\, (MeV)}$")
        ax.set_xlim(min(x), max(x))
        ax.xaxis.set_tick_params(width=2, length = 8)
        ax.yaxis.set_tick_params(width=2, length = 8)
        font = {'family':'sans-serif'}
        plt.rc('font', **font)

        if lab is not None:
            # Now add the legend with some customizations.
            legend = plt.legend(loc='best', shadow=True)

            # Set the fontsize
            for label in legend.get_texts():
                label.set_fontsize('large')

            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width

    return x_interval

def values ( inf, sup, period_int, period_ext, Lpdz ):
    """
    returns an array of values needed to generate the file name.

    Parameters:
    -----------
    inf : float
        inferior boundary in terms of longitudinal coordinates

    sup : float
        superior boundary in terms of longitudinal coordinates

    period_int: int
        the period of data dumping between the interior and the superior
        boundary

    period_ext: int
        the period of data dumping outside the interior and the superior
        boundary

    Lpdz: float
        Length of plasma/ dz

    Returns:
    --------
    val: 1D numpy array
        an array of values corresponding to the data dumping period
    """

    distance = 0
    val = []

    while (distance < Lpdz):
        if (distance < inf or distance >= sup):
            period = period_int
        else:
            period = period_ext
        val.append( distance + period )
        distance += period

    return ( val )

def gamma2Energy (gamma):
    """
    Returns energy

    Parameters:
    -----------
    gamma: 1D numpy array
        gamma of the selected particle beams

    Returns:
    --------
    energy: 1D numpy array
        energy of the selected particle beams

    """

    return 0.511*gamma

def randintGenerator( number, fromNum ):
    """
    Returns an array of random int

    Parameters:
    -----------
    Number: int
        Size of the returned array

    fromNum: int
        Size of the pool to choose from

    Returns:
    --------
    randomInt: an array of int
        an array of random numbers
    """

    randomInt =random.sample( range(0 , fromNum), number )

    return randomInt


def leftRightFWHM( yleft, yright, yFWHM, xleft, xright ):
    """
    determines the left and right values of the FWHM value of both left and
    right with the reference of the peak.

    Parameters:
    -----------
    yleft: 1D numpy array
        y values of the LHS of the peak
    yright: 1D numpy array
        y values of the RHS of the peak
    xleft: 1D numpy array
        x values of the LHS of the peak
    xright: 1D numpy array
        x values of the RHS of the peak

    Returns:
    --------
    Positions in x and y with reference to yFWHM

    yleftleft: int
    yleftright: int
    yrightleft: int
    yrightright: int
    xleftleft: int
    xleftright: int
    xrightleft: int
    xrightright: int

    """
    indLeft = np.where(yleft<yFWHM)
    indRight = np.where(yright<yFWHM)

    try:
        yleftleft = yleft[indLeft[0][-1]]
        yleftright = yleft[indLeft[0][-1]+1]
        yrightleft = yright[indRight[0][0]-1]
        yrightright = yright[indRight[0][0]]
        xleftleft = xleft[indLeft[0][-1]]
        xleftright = xleft[indLeft[0][-1]+1]
        xrightleft = xright[indRight[0][0]-1]
        xrightright = xright[indRight[0][0]]
    except IndexError:
        yleftleft = yleftright = yleft[indLeft[0][-1]]
        yrightleft = yrightright = yright[indRight[0][0]]
        xleftleft = xleftright = xleft[indLeft[0][-1]]
        xrightright = xrightleft = xright[indRight[0][0]]

    return yleftleft, yleftright, yrightleft, yrightright,\
    xleftleft, xleftright, xrightleft, xrightright

def bilinearInterpolation( yleft, yright, xleft, xright, yreal):
    """
    returns the interpolated value of x, calculated by bilinear interpolation

    Parameters:
    -----------
    yleft: int
        y-value on the left

    yright: int
        y-value on the right

    xleft: int
        x-value on the left

    xright: int
        x-value on the right

    yreal: int
        exact y-value to collapse to

    Returns:
    --------
    xreal: int
        x-value corresponding to yreal after interpolation
    """

    dist = np.absolute( yleft - yright )

    xreal = xleft + ( 1./dist )*(np.absolute( yleft - yreal )\
                 *np.absolute( xright - xleft ))

    return xreal

def w2charge ( w ):
    """
    returns the charge of a each particle.

    Parameters:
    -----------
    w : 1D numpy array
        weight of the particles

    Returns:
    --------
    charge: 1D numpy array
    """

    return w*e

def findRoot( y, x ):
    """
    finds the zero crossing or the roots of the input signal, the input signal
    should be somewhat smooth.

    Parameters:
    -----------
    x: 1D numpy array
        values in x-coordinates

    y: 1D numpy array
        values in y-coordinates

    Returns
    -------
    roots : 2D numpy array
    """
    #displaying the sign of the value
    l = np.shape(y)[0]
    s = np.sign(y)
    index = []

    for i in xrange ( l-1 ):
        if (s[i+1]+s[i] == 0 ):
            # when the sum of the signs ==0, that means we hit a 0
            index.append(i)

    roots = np.take( x , index ).tolist()
    lrz = len(roots)
    # if there's only one root found,
    # there should be an end to it, we consider the min z
    # as the the limit

    # insert a z value at the first index
    if lrz == 1:
        roots.insert( 0, min(x) )
    # if length of root is not pair, we remove the first value
    if np.shape(roots)[0]%2 != 0:
        roots = np.delete( roots, 0 )

    return roots

def wavg( a , weights ):
    """
    Calculate the weighted average.

    Paramters:
    ----------
    a : array_like
        Calculate the weighted average for these a.

    weights : array_like
        An array of weights for the values in a.

    Returns:
    --------
    w_avg: float
        Weighted average of
    """
    a = np.array(a)
    weights = np.array(weights)
    # Check if input contains data
    if not np.any(weights) and not np.any(a):
        # If input is empty return NaN
        return np.nan
    else:
        # Calculate the weighted standard deviation
        indices = ~np.isnan(a)
        w_avg = np.average(a[indices], weights = weights[indices])

        return( w_avg )

def wstd( a, weights ):
    """
    Calcualte the weighted standard deviation.

    Parameters
    ----------
    a : array_like
        Calculate the weighted standard deviation for these a.

    weights : array_like
        An array of weights for the values in a.

    Returns
    -------
    Float with the weighted standard deviation.
    Returns nan if input array is empty
    """
    a = np.array(a)
    weights = np.array(weights)
    # Check if input contains data
    if not np.any(weights) and not np.any(a):
        # If input is empty return NaN
        return np.nan
    else:
        # Calculate the weighted standard deviation
        indices = ~np.isnan(a)
        average = np.average(a[indices], weights = weights[indices])
        variance = np.average((a[indices]-average)**2, weights = weights[indices])

        return( np.sqrt(variance) )

def savitzkyGolay( y, window_size, order, deriv=0, rate=1 ):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve( m[::-1], y, mode='valid')

def charge_density( x, gamma, w, reduction_factor = None, bin_num = None):
    """
    returns the histogram weighted by charge.

    Parameters:
    -----------
    x: 1D numpy array
        distribution in position (can be x, y or z)

    gamma: 1D numpy array
        distribution in momentum (can be ux, uy, uz)

    w: 1D numpy array
        weight distribution of the particles

    reduction_factor: int
        reduction factor for energy so that the max(energy) will be resized to
        the max(field). Default: None

    bin_num : int
        required bin number for hisogram plot

    Returns:
    --------
    Hmasked: 2D numpy array
        2D distribution of the charge

    extent: 1D array
        necessary information on the limits in both x and y for reconstuction
        purpose

    """
    charge = w2charge( w )
    energy = gamma2Energy( gamma )

    if bin_num == None:
        bin_num = int((max(energy)-min(energy))*((max(x) - min(x))/1e-6))

    if reduction_factor is not None:
        energy *= reduction_factor

    H, xedges, yedges = np.histogram2d(energy, x,
                        bins = bin_num, weights = charge)
    Hmasked = np.ma.masked_where(H == 0,H)
    extent = [ min(x), max(x), min(energy), max(energy) ]

    return Hmasked, extent

def bigPicture( frame_num, p_z, p_gamma, p_w, f_z, f_wake, f_laser,
                lsavefigure = True, lwrite = False, reduction_factor = None,
                bin_num = None):
    """
    Plots the big picture.

    Parameters:
    -----------
    frame_num: int
        frame number for writing purpose

    p_z: array of floats
        position in z of particles

    p_gamma: array of floats
        gamma values of particles

    p_w: array of weights
        weight of particles

    f_z: array of floats
        position in z of the field

    f_wake: array of floats
        normalized ez field

    f_laser: array of floats
        normalized laser field

    lsavefigure: boolean
        if True, the figure will be saved. Default: True

    lwrite: boolean
        if True, data will be saved. Default: False

    reduction_factor: int
        reduction factor for energy so that the max(energy) will be resized to
        the max(field). Default: None

    bin_num: int
        required bin_num for histo plot

    """

    if 'inline' in matplotlib.get_backend():
        fig, ax = plt.subplots( dpi = 150 )
    else:
        fig,ax = plt.subplots( figsize=( 10, 8 ) )

    fig.patch.set_facecolor('white')
    cm_peak = plt.cm.get_cmap('RdBu')

    try:
        if reduction_factor is None:
            reduction_factor = 2*max(f_laser)/max(p_gamma)
        Hmasked, extent = charge_density( p_z, p_gamma, p_w, reduction_factor,
                            bin_num )

        sc_peak = ax.imshow( Hmasked, extent = extent, interpolation='nearest',
                        origin='lower', cmap=cm_peak, aspect = "auto")
        colorbar_pos_peak = fig.add_axes([0.9,0.118,.025,.782])
        fig.colorbar(sc_peak, cax =colorbar_pos_peak,
                            orientation = 'vertical')

        # Writing the particle
        if lwrite:
            qname_particle = [ "charge_density" , "extent" ]
            fp = FileWriting( qname_particle , "Charge_density_%d" %frame_num )
            data = [Hmasked.data] + [extent]
            fp.write( data, np.shape(data) , attrs = ["C/m^2", "m"])

    except ValueError:
        reduction_factor = 0
        print ("No particles are detected for frame %d" %frame_num)

    ax.plot( f_z, f_wake, color="red" )
    ax.plot( f_z, f_laser, color = "#87CEEB" )
    ax.xaxis.set_tick_params(width=2, length = 8)
    ax.yaxis.set_tick_params(width=2, length = 8)
    ax.set_xlim(min(f_z), max(f_z))
    ax.set_xlabel(r"$\mathrm{z\,(m)}$")
    if reduction_factor !=0:
        ax.set_ylabel(r"$\mathrm{Norm.\, amp.}$"+"\n"+ \
                        r"$\mathrm{Energy/%d\,(MeV)}$" \
                        %int(1/reduction_factor))
    else:
        ax.set_ylabel(r"$Norm.\, amp.$")
    plt.setp(ax.get_xticklabels()[::2], visible=False)
    font = {'family':'sans-serif'}
    plt.rc('font', **font)

    if lsavefigure:
        fig.savefig( config.result_path + "bigPicture_%d.png" %frame_num)

    if lwrite:
        # Writing the field
        gname_field = ["Laser", "Wake"]
        qname_field = ["z", "field_amp"]
        ff = FileWriting( qname_field, "Normalized_Fields_%d" %frame_num,
                        groups = gname_field)
        zfield = [f_z]*2
        field = [f_laser] + [f_wake]
        stacked_data = np.stack( (zfield, field  ), axis = 1 )
        ff.write(stacked_data, np.shape(stacked_data),
        attrs = ["m", "normalized_unit (by w0 for laser, wp for wakefield)"])
