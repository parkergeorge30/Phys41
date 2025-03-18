import astropy.io.fits as fits
import astropy.units as u
import numpy as np
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1 import make_axes_locatable

def zeros(x, dtype=int):
    return [dtype(0.0)] * int(x)

def my_sum(x):
    tot = 0.
    for i in x:
        tot += i
    return tot

def my_avg(x):
    return my_sum(x) / len(x)

def get_data(files):
    """
    grabs data from name of fits file

    Usage:
        files = ['data1.fits', 'data2.fits']
        data_array = get_data(files)

    :param files: str or list of strs, name of files to get data from
    :return: array, data array
    """
    # checks to see if files is a list, if not, make it one
    if type(files) is str:
        files = [files]
    n = len(files)

    # if only 1 item in list, just return the data for that str
    if n == 1:
        return fits.getdata(files[0])

    # otherwise initialize empty array and fill it with data
    data = zeros(n, dtype=float)
    for i, file in enumerate(files):
        data[i] = fits.getdata(file)

    return data

def my_bias(x):
    """
    Creates a master bias frame to use for data reduction
    """
    n = len(x)
    tot = zeros(n, dtype=float)

    for i, file in enumerate(x):
        arr = fits.getdata(file)
        tot[i] = arr
    avg = my_avg(tot)
    return avg

def ascend_str(ls, idxs):
    """
    Orders the numbers in the files from least to greatest numerically
    """
    idx = 0
    idx_slice = slice(idxs[0], idxs[1])  # make slice for desired indices
    while idx < len(ls):
        for i, x in enumerate(ls):
            if i > idx and int(x[idx_slice]) < int(ls[idx][idx_slice]):  # check for lowest integer at given indices
                ls[i], ls[idx] = ls[idx], x

        idx += 1
    return None

def set_negatives_to_zero_nd(tensor):
    """
    sets negative values to 0 inplace for a rank n tensor
    """
    # check for rank 1
    ele = tensor[0]
    if isinstance(ele, np.ndarray):
        # not inside rank 1 yet so recursively loop with self call
        for sub in tensor:
            set_negatives_to_zero_nd(sub)
    else:
        # we are inside the rank 1 now
        for i, val in enumerate(tensor):
            if val < 0:                         # if less than zero
                tensor[i] = 0                   # set to zero
    return None

def linear_least_squares(x, y, weights=None):
    """
    Linear Least Square Fit

    :param x: np.array
    :param y: np.array
    :param weights: np.array, optional error associated with x
    :return: m, c tuple representing (slope, intercept)
    """
    # ensure numpy
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # set weights if provided
    w = 1. / (np.asarray(weights, dtype=np.float64) ** 2) if weights is not None else np.ones_like(x)

    # construct design matrix B: each row is [x_i, 1]
    B = np.column_stack((x, np.ones_like(x)))

    # compute weighted normal matrix
    W = np.diag(w)
    BTW = B.T @ W
    Cov = np.linalg.inv(BTW @ B)  # 2x2 covariance matrix

    # finalize
    params = Cov @ (B.T @ (w * y))
    m, c = params[0], params[1]
    return m, c, Cov

def remove_bad_cols(x, bad_cols):
    """
    removes bad columns from x by setting them to background

    :param x: 2d array, data with bad columns
    :param bad_cols: int or list, index or indices of bad columns
    :return: 2d array, frame with bad columns set to background
    """
    x[:, bad_cols] = np.median(x)
    return x

def get_hdr_data(file, entry):
    """
    get header value for a file

    Usage:
        exp = get_hdr_data('data1.fits', 'EXPTIME')

    :param file: str, name of file
    :param entry: str, name of header entry storing desired value
    :return: object, value of header entry
    """
    with fits.open(file) as hdu:
        hdr = hdu[0].header

    return hdr[entry]

def load_headers_all_files(headers, data_files=None, data_dir=None):
    """
    load values for each header entry for all files in a 2D array
    rows being headers, and cols being data_file

    :param headers: list, list of header entries
    :param data_files: list, list of data files
    :param data_dir: str, optional prefix for all data files
    :return: np.array, 2d array of values for each header for each file
    """
    if data_dir is None or not isinstance(data_dir, str):
        data_dir = globals().get('data_dir')
        if data_dir is None:
            data_dir = ""

    if data_files is None:
        data_files = globals().get('data_files')
        if data_files is None:
            raise ValueError("No available 'data_files' has been defined.")

    # All loaded headers will be saved here. np.full() creates a NumPy array of a given shape (first argument)
    # filled with a constant value (second argument, empty string in this case). "dtype = object" will allow
    # the array to store data of any type (some headers may be numbers, not strings).
    output = np.full([len(headers), len(data_files)], "", dtype=object)

    # Now read the headers from all files
    for i, hdr in enumerate(headers):
        for j, file in enumerate(data_files):
            output[i, j] = get_hdr_data(data_dir + file, hdr)

    return output

def load_frame_overscan_remove_bias_subtract(filename, bias, overscan=32, bad_col_idx=None):
    """
    load frame and subtract the bias and remove overscan

    :param filename: str or list, name or list of names of data_file
    :param bias: array, bias frame
    :param overscan: int, overscan value
    :param bad_col_idx: int, index of bad column
    :return: 2d or 3d np.array, clean frame or frames
    """
    if isinstance(filename, str):
        frame = get_data(filename)
        image = frame - bias
        set_negatives_to_zero_nd(image)

        # remove overscan (right black bar region)
        image = image[:, :int(np.shape(image)[1] - overscan)]

        # set bad col to background noise
        if isinstance(bad_col_idx, int):
            image[:, bad_col_idx] = np.median(image)

        return image

    elif isinstance(filename, list):
        images = zeros(len(filename), np.float32)
        for i, fname in enumerate(filename):
            frame = get_data(fname)
            images[i] = frame - bias
            set_negatives_to_zero_nd(images[i])

            # remove overscan (right black bar region)
            images[i] = images[i][:, :int(np.shape(images[i])[1] - overscan)]
        return images

def load_reduced_science_frame(filename, flat, bias):
    """
    load reduced science frame

    :param filename: str or list, path to frame file
    :param flat: array, cleaned flat frame
    :param bias: array, bias frame
    :return: array, normalized reduced frame
    """
    # bias sub data
    data_clean = load_frame_overscan_remove_bias_subtract(filename, bias)

    # norm with clean flat
    norm = data_clean / (flat + 1e-2)

    return norm

def plot_im(ax, data, xlabel='x [pixel]', ylabel='y [pixel]', title='', **imshow_kwargs):
    """
    Plot an image using matplotlib imshow.

    :param ax: plt.Axes, axes to plot on
    :param data: array, data to imshow
    :param xlabel: str, x label
    :param ylabel: str, y label
    :param title: str, title
    :param imshow_kwargs: iterable, imshow kwargs including colorbar axis kwargs 'pad' and 'size'
    :return: None, plots data using imshow
    """
    # extract cax kwargs from imshow_kwargs
    cax_keys = ['pad', 'size']
    cax_kwargs = {key: imshow_kwargs.pop(key) for key in cax_keys if key in imshow_kwargs}
    cax_kwargs.setdefault('size', "5%")
    cax_kwargs.setdefault('pad', 0.05)

    fig = ax.get_figure()
    ax.imshow(data, **imshow_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cax_kwargs['size'], pad=cax_kwargs['pad'])
    fig.colorbar(ax.images[0], cax=cax)

def find_star_locs(im_data, n_size=10, bright_count_thresh=10, background_factor=2):
    """
    ARGUMENTS:
    ================================================================
    im_data - Reduced 2D FITS Data
    n_size  - Neighborhood size to consider for windowing (pixels)
    bright_count_thresh - Threshold for number of 'bright'pixels in
                          neighborhood to be considered a star.
                          (Proportional to size of Star blob)
    background_factor - factor to multiply background by to set
                        definition of bright pixel
    RETURNS:
    ================================================================
    [[x_positions_of_star_center, y_positions_of_star_center]]
    i.e., a list of x and y coordinate of star centers
    """

    # set definition of a 'bright' pixel to be 3 times the background
    background = np.median(im_data)
    count_threshold = background_factor * background

    star_centers = []

    # indexing uses x=rows, y=cols
    # but for analysis we want x=cols and y=rows, so we must swap when referencing
    ny, nx = im_data.shape

    for y in range(0, ny, n_size):
        for x in range(0, nx, n_size):
            # check to see if we're in either left corner (bad corners), if so, skip
            if (y > 979 or y < 50) and x < 50:
                continue

            # set window
            window = im_data[y:y + n_size, x:x + n_size]

            count_bright = np.sum(window > count_threshold)
            if count_bright >= bright_count_thresh:
                center_y = y + window.shape[0] // 2
                center_x = x + window.shape[1] // 2

                star_centers.append([center_y, center_x])

    return star_centers

def make_pos_array(image):
    """
    makes image position array with same shape as image

    :param image: 2D array, science frame
    :return: position array, same shape as image
    """
    row_dim = np.shape(image)[0]
    row_pos = np.array(range(row_dim))[np.newaxis]
    pos_xarr = np.tile(row_pos.T, (1, row_dim))
    pos_yarr = np.tile(row_pos, (row_dim, 1))
    pos_arr = np.dstack((pos_xarr, pos_yarr))

    return pos_arr

def collapse_subsets(arr_list):
    """
    remove all subsets from a list, only keep the super sets (sets that are not subsets of another set)

    :param arr_list: list or array-likes
    :return: list of super sets present in arr_list
    """
    # convert arrays to sets
    sets = [set(arr) for arr in arr_list]
    keep = [True] * len(arr_list)

    # check for subsets
    for i, s in enumerate(sets):
        for j, t in enumerate(sets):
            if i != j:
                # if s is a strict (not duplicate) subset of t, mark it to be removed.
                if s.issubset(t) and len(s) < len(t):
                    keep[i] = False
                    break

    # remove duplicates
    filtered = []
    for flag, arr in zip(keep, arr_list):
        if flag and arr not in filtered:
            filtered.append(arr)

    return filtered

def calc_centroids_2d(intarr, posarr, loc_list, window_max=20):
    """
    given intensities, positions, locations, and a window size, calculate the
    centroid position value for each window at the specified location

    PARAMETERS:
    ==================================================================================
    intarr - array of intensities (image), shape [y, x]
    posarr - array of positions, shape [y, x, 2]
    loc_list - list of [y, x] (NOT [x, y]!) coords to calculate centroids around
    window_max - Size of Window to consider to find max pos of each star (in pixels)

    RETURNS:
    ==================================================================================
    centroids - List of centroid coordinates and corresponding uncertainities
                Format: [[xc, yc, unc_xc, unc_yc]]
    """

    centroids = []

    window_size = window_max // 2

    for i, (y, x) in enumerate(loc_list):
        # check edges
        if x < window_size or y < window_size or y > np.shape(intarr)[0] - window_size or x > np.shape(intarr)[1] - window_size:
            # centroids.append([float('NaN')]*4)
            continue

        # window off region
        y_slice = slice(y - window_size, y + window_size)
        x_slice = slice(x - window_size, x + window_size)
        region_ints = intarr[y_slice, x_slice]
        region_pos = posarr[y_slice, x_slice, :]

        # denominator
        tot_int = np.sum(region_ints)

        # matrix version of equation above
        centroid = np.einsum('ijk,ij->k', region_pos, region_ints) / tot_int

        # error prop
        diff = region_pos - centroid

        # matrix version of eq above (diff transposed is represented by the different indices in einsum)
        cov = np.einsum('ij,ijk,ijl->kl', region_ints, diff, diff) / tot_int ** 2
        sig_y, sig_x = cov[0, 0] ** .5, cov[1, 1] ** .5

        # round pixels to whole numbers
        centroid_full = [round(centroid[0]), round(centroid[1]), sig_y, sig_x]

        centroids.append(centroid_full)

    centroids = np.array(centroids, dtype='object')

    # remove marks for same cluster
    # set threshold window to half window_max in pixels
    threshold = window_max // 2
    all_neighbor_indices = []
    for i, (y, x, sy, sx) in enumerate(centroids):
        # check threshold and make mask for those that cross it
        diff = np.abs(centroids - centroids[i])
        mask = (diff[:, 0] <= threshold) & (diff[:, 1] <= threshold)
        neighbor_indices = np.where(mask)[0]

        # only store clusters, not single star locs
        if len(neighbor_indices) > 1:
            all_neighbor_indices.append(list(neighbor_indices))

    # remove subsets to only get the main clusters
    clusters = collapse_subsets(all_neighbor_indices)
    keep = [True] * len(centroids)
    for cluster in clusters:
        # collapse along vertical dimension to average all values
        collapsed_cluster = np.mean(centroids[cluster], axis=0)

        # set y, x to be integers as they are pixel nums
        collapsed_cluster[0:2] = round(collapsed_cluster[0]), round(collapsed_cluster[1])

        # overwrite first occurence of cluster with the averaged version of its neighbors
        centroids[cluster[0]] = collapsed_cluster

        # mark neighbors for removal (everything after first occurence of cluster)
        for idx in cluster[1:]:
            keep[idx] = False

    centroids = centroids[keep]

    return centroids

def local_pixel_size(ra_deg, dec_deg, center_coord, focal_length=16480, pixel_size=0.030, offset=512, standard=False):
    """
    Converts RA and DEC from degrees to x pixel and y pixel using plate constants

    Params:
    ra_deg: Right Ascension of the object in degrees
    dec_deg: Declination of the object in degrees
    center_coord: AstroPy SkyCoord object, coordinate of center of image
    standard: bool, if True, returns standard coordinates only, no local conversion
    focal_length: focal length in mm
    pixel_size: pixel size in mm
    offset: pixel offset in pixels

    Returns: the x pixel and y pixel locations of the object
    """
    # convert all values from deg to radians
    ra = ra_deg * np.pi / 180
    dec = dec_deg * np.pi / 180
    ra_0 = center_coord.ra.value * np.pi / 180
    dec_0 = center_coord.dec.value * np.pi / 180

    # calculate common denominator beforehand
    denom = (np.cos(dec_0)*np.cos(dec)*np.cos(ra-ra_0) + np.sin(dec) * np.sin(dec_0))

    # standard coordinates from ra and dec
    X = - np.cos(dec) * np.sin(ra-ra_0)/denom
    Y = - (np.sin(dec_0)*np.cos(dec)*np.cos(ra-ra_0) - np.cos(dec_0)*np.sin(dec))/denom

    if standard:
        return X, Y

    # convert to pixels using plate constants and center using offset
    x = focal_length*X/pixel_size + offset
    y = focal_length*Y/pixel_size + offset

    return x, y

def local_plate_scale(ra_deg, dec_deg, center_coord, plate_scale=0.368, offset=512):
    """
    Converts RA and DEC from degrees to x pixel and y pixel using plate scale

    Params:
    ra_deg: Right Ascension of the object in degrees
    dec_deg: Declination of the object in degrees
    center_coord: AstroPy SkyCoord object, coordinate of center of image
    plate_scale: plate scale in as/px
    offset: pixel offset in pixels

    Returns: the x pixel and y pixel locations of the object
    """
    # get relative to center coord and change from deg to arcsec
    ra = (ra_deg - center_coord.ra.value) * 3600
    dec = (dec_deg - center_coord.dec.value) * 3600

    # convert from as to px using plate scale
    x = ra / plate_scale
    y = dec / plate_scale

    # centering by offset
    x += offset
    y += offset

    return x, y

def sky_query(dataframe, filename, fov_width='6.3m', fov_height='6.3m', magnitude_limit=18):
    """
    vizier query the sky for objects around center of a file

    :param dataframe: pandas.DataFrame, dataframe containing the data of file, must have columns
                            ['FILE NAME'] ['RA'] ['DEC'] ['DATE-BEG'] ['RADECSYS']
    :param filename: str, name of file in dataframe
    :param fov_width: str, width of field of view in arcs, '6.3m' is 6.3 arcmin
    :param fov_height: str, height of field of view in arcs, '6.3m' is 6.3 arcmin
    :param magnitude_limit: float, R2 magnitude limit
    :return: ra, dec arrays of queried objects
    """
    # grab necessary values from provided dataframe
    ra_center, dec_center, yr = dataframe.loc[dataframe['FILE NAME'] == filename, ['RA', 'DEC', 'DATE-BEG']].values[0]
    reference_frame = dataframe.loc[dataframe['FILE NAME'] == filename, 'RADECSYS'].values[0].lower()

    center_coord = SkyCoord(ra=ra_center, dec=dec_center, unit=(u.hour, u.deg), frame=reference_frame)

    vizier = Vizier(column_filters={"R2mag": f"<{magnitude_limit}"})
    result_table = vizier.query_region(center_coord, width=fov_width, height=fov_height, catalog="USNO-B1")

    # Extract required data from obtained query results
    ra_cat = np.array(result_table[0]["RAJ2000"])  # this is the stars' RA in the 2000 epoch
    dec_cat = np.array(result_table[0]["DEJ2000"])  # this is the stars' Dec in the 2000 epoch
    pm_ra = np.array(result_table[0]["pmRA"])  # this is the RA proper motion of the stars
    pm_dec = np.array(result_table[0]["pmDE"])  # this is the Dec proper motion of the stars
    mag = np.array(result_table[0]["R2mag"])

    # convert mas/yr to deg/yr
    pm_ra = pm_ra / 1000 / 3600
    pm_dec = pm_dec / 1000 / 3600

    # time in years since epoch (2000)
    dt = yr - 2000

    # add proper motion to epoch coordinates
    ra_cat = ra_cat + pm_ra * dt
    dec_cat = dec_cat + pm_dec * dt

    return ra_cat, dec_cat, center_coord

def nearest_neighbor_match(a, b):
    """
    Matches entries by Euclidean distance

    Parameters:
        a : ndarray of shape (N,2)
            Array containing x and y positions [x, y].
        b : ndarray of shape (M,2)
            Array containing x and y positions [x, y].

    Returns:
        matches : list of tuples
            Each tuple is (a_index, b_index) indicating a match.
    """
    # Ensure inputs are numpy arrays
    a = np.array(a).astype(np.float64)
    b = np.array(b).astype(np.float64)

    N = a.shape[0]
    matches = []
    used_centroids = set()

    for i in range(N):
        a_i = a[i]  # This should be a (2,) array

        # Euclidean distance:
        diff = b - a_i  # Shape (M,2)
        distances = np.sqrt(np.sum(diff**2, axis=1))  # Shape (M,)

        sorted_indices = np.argsort(distances)
        for j in sorted_indices:
            if j not in used_centroids:
                matches.append((i, j))
                used_centroids.add(j)
                break

    return matches

def get_1_sigma_region(x, cov):
    """
    Gets 1 sigma region from covariance matrix

    :param x: np.ndarray, independent variable
    :param cov: np.ndarray, covariance matrix
    :return: np.ndarray, 1-sigma region
    """
    # ensure numpy

    sigma_m = cov[0, 0] ** .5
    sigma_c = cov[1, 1] ** .5
    cov_mc = cov[0, 1]
    sigma_y = (x ** 2 * sigma_m ** 2 + sigma_c ** 2 + 2 * x * cov_mc) ** .5
    return sigma_y