# Functions to extract video/image features.
# @Qiuhan Jin, 15/08/2023


import numpy as np
import os, copy, cv2, time
from datetime import datetime
from skimage.feature import hog, local_binary_pattern
from skimage.exposure import rescale_intensity
from skimage.draw import line_aa
from scipy.ndimage import uniform_filter, gaussian_filter1d


def get_unit_vector(vector):
    """ Return the unit vector of the vector.  
    """
    return vector / np.linalg.norm(vector)


def compute_vector_angle(vector_1, vector_2):
    """ Return the angle in radians between vectors 'vector_1' and 'vector_2':: 
        Direction = vector_2 -> vector_1, vector_2 is the starting vector.
    """
    v1_u = get_unit_vector(vector_1)
    v2_u = get_unit_vector(vector_2)
    angle = np.arctan2(v1_u[1], v1_u[0]) - np.arctan2(v2_u[1], v2_u[0])
    # np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle


def get_luminance_contrast(image):
    """ Return luminance (mean) and contrast (std) of a grayscale image.
    """
    assert len(image.shape) == 2, 'Input image should be grayscale.'
    image_luminance = np.mean(image)
    image_contrast = np.std(image)
    return image_luminance, image_contrast


def get_Sobel_gradients(image, smooth_kernel_size=3):
    """ Calculate Sobel gradients of a grayscale image. 
    Sobel:
        cv2.Sobel() (Gaussian gradients) is adopted in HOG instead of Laplacian gradients.
        https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
        https://docs.opencv.org/3.4/d5/d0f/tutorial_py_gradients.html
        https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/_hog.py#L48-L302
            In scikit-image hog, first-order image spatial gradients are computed just as 
            using a Sobel operator (Gaussian derivative).
    """
    assert len(image.shape) == 2, 'Input image should be grayscale.'
    # Sobel gradients
    h, w = image.shape[:2]
    image_Sobel_gradients = np.zeros((h, w, 2))
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=smooth_kernel_size)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=smooth_kernel_size)
    image_Sobel_gradients[:, :, 0] = sobelx
    image_Sobel_gradients[:, :, 1] = sobely
    return image_Sobel_gradients


def get_Canny_edges(image, min_thr=50, max_thr=200, smooth_kernel_size=3):
    """ Calculate Canny edges of a gayscale image.
    Canny:
        Canny edge detector is a multi-stage edge detector. 
        Image is smoothed by a Gaussian kernel first to reduce noise.
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html
    Sobel and Canny edge detectors are both favorite edge detectors with enhancement.
        https://medium.com/@haidarlina4/sobel-vs-canny-edge-detection-techniques-step-by-step-implementation-11ae6103a56a
        Canny is more enhanced and complex.
    """
    assert len(image.shape) == 2, 'Input image should be grayscale.'
    # Canny edges
    image_Canny_edges = cv2.Canny(image, min_thr, max_thr, smooth_kernel_size)
    # Laplacian edges
    # image_laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Sobel edges
    # image_Sobel_edges = filters.sobel(image)  # Sobel gradients + smoothing = edges
    return image_Canny_edges


def get_FAST_corners(image, visualize=True, fast_thr=10, fast_nonmaxSuppression=True):
    """ https://docs.opencv.org/4.5.4/df/d0c/tutorial_py_fast.html
        FAST corner detection on a grayscale image.
    """
    assert len(image.shape) == 2, 'Input image should be grayscale.'
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create(threshold=fast_thr, nonmaxSuppression=fast_nonmaxSuppression)
    # Find and draw the keypoints
    kp = fast.detect(image, None)
    list_FAST_corners = [x.pt for x in kp]
    mask_FAST_corners = np.zeros_like(image)
    for kp_tuple in list_FAST_corners:
        kp_x = int(kp_tuple[0])
        kp_y = int(kp_tuple[1])
        mask_FAST_corners[kp_y, kp_x] = 1
    if visualize:
        vis_FAST_corners = cv2.drawKeypoints(image, kp, None, color=(255,0,0))
    else:
        vis_FAST_corners = None
    return list_FAST_corners, mask_FAST_corners, vis_FAST_corners


def get_hog(image, visualize=True, num_orientations=9, num_pixels_per_cell=(8, 8),
            num_cells_per_block=(2, 2), normalize_range=(0, 10)):
    """ Calculate histogram of image gradients of a (multi-D) image and 
        return HOG feature vector, HOG feature array, and normalized HOG image.
        https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html

        Default parameters are chosen considering both missing rate and speed.
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467360
    """
    num_channels = len(image.shape)
    sy, sx = image.shape[:2]  # image size in pixels
    cx, cy = num_pixels_per_cell  # cell size in pixels
    bx, by = num_cells_per_block  # block size in cells
    n_cellsx = int(np.floor(sx / cx))  # number of cells in image x axis
    n_cellsy = int(np.floor(sy / cy))  # number of cells in image y axis
    n_blocksx = (n_cellsx - bx) + 1  # number of blocks in image x axis
    n_blocksy = (n_cellsy - by) + 1  # number of blocks in image y axis
    if np.sum(image) == 0:
        hog_vector = np.zeros((n_blocksy, n_blocksx, by, bx, num_orientations))
        if visualize:
            hog_image_rescaled = np.zeros((sy, sx), dtype=float)
        else:
            hog_image_rescaled = None
    else:
        image_data = image
        # if image.shape[0] < 100 or image.shape[1] < 100:
        #     image_data = copy.deepcopy(image)
        #     image_data = cv2.resize(image_data, (100, 100), interpolation = cv2.INTER_AREA)
        if visualize:
            if num_channels > 2:
                hog_vector, hog_image = hog(
                    image_data, orientations=num_orientations, pixels_per_cell=num_pixels_per_cell,
                    cells_per_block=num_cells_per_block, visualize=True, channel_axis=-1
                )
            else:
                hog_vector, hog_image = hog(
                    image_data, orientations=num_orientations, pixels_per_cell=num_pixels_per_cell,
                    cells_per_block=num_cells_per_block, visualize=True
                )
            hog_image_rescaled = rescale_intensity(hog_image, in_range=normalize_range)
        else:
            if num_channels > 2:
                hog_vector = hog(
                    image_data, orientations=num_orientations, pixels_per_cell=num_pixels_per_cell,
                    cells_per_block=num_cells_per_block, visualize=False, channel_axis=-1
                )
            else:
                hog_vector = hog(
                    image_data, orientations=num_orientations, pixels_per_cell=num_pixels_per_cell,
                    cells_per_block=num_cells_per_block, visualize=False
                )
            hog_image_rescaled = None
    # hog_vector may be empty if image is too small, replace with (9,) zeros
    if len(hog_vector) == 0:
        hog_vector = np.zeros((n_blocksy, n_blocksx, by, bx, num_orientations)).flatten()
    # Recover feature array of shape (celly, cellx, orientation)
    normalized_blocks = hog_vector.reshape((n_blocksy, n_blocksx, by, bx, num_orientations))
    hog_array = np.zeros((n_blocksy*by, n_blocksx*bx, num_orientations))
    for y in range(n_blocksy):
        for x in range(n_blocksx):
            for b in range(by):
                for a in range(bx):
                    hog_array[y*by+b,x*bx+a,:] = normalized_blocks[y,x,b,a,:]
    return hog_vector, hog_array, hog_image_rescaled


def get_local_binary_pattern(image, radius=3, num_pixels=8, method='uniform'):
    """ Calculate local binary patterns of a grayscale image.
        https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
        Rotation-invariant, uniform combinations: Considering rotation invariance, 
            there are 9 unique, uniform combinations:
            00000000, 00000001, 00000011, 00000111, 00001111, 00011111, 00111111, 01111111, 11111111
        The values of img_lbp range from 0 to 9, corresponding to the 9 patterns.
        After getting the lbp of all pixels, we need to summarize their patterns into a histogram.
    """
    assert len(image.shape) == 2, 'Input image should be grayscale.'
    # method: 
    #   type of LBP ['flat', 'flat', 'edge', 'corner', 'non-uniform']
    #   uniform = at most two circular 0-1 or 1-0 transitions
    # radius = distance between central pixels and comparison pixels
    num_points = num_pixels * radius  # define number of comparison pixels
    image_lbp = local_binary_pattern(image, num_points, radius, method)  # same shape as input image
    # print(img_lbp.shape)
    num_bins = int(image_lbp.max()+1)
    image_lbp_hist, hist_bin_edges = np.histogram(
        image_lbp.ravel(), density=True, bins=num_bins, range=(0, num_bins)
    )
    # print(img_lbp_hist, hist_bin_edges)
    return image_lbp, image_lbp_hist, hist_bin_edges


def generate_2d_Gabor_bank(parameter_set):
    """ Return a gabor bank (a dict of gabor kernels, every set is two groups with opposite phase_offset,
            gabor kernels in both groups share the same wavelength and orientation (specified in first level key)).
        cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
                           ksize, bandwidth, orientation, wavelength, gamma, phase_offset
    
    parameter_set is a dict with the following fields:
        ksize, wavelength, orientation, phase, gamma, bandwidth
    Each field corresponds to a list of parameter values.
    https://www.cs.rug.nl/~imaging/simplecell.html
    https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97
    
    Wavelength (lambda): wavelength of the cosine factor of the Gabor function, given in pixels, valid values betwween 2 and image length.
        In 3D gabor function, wavelength is associated with preffered speed of the filter.
    Orientation (theta): orientation of the normal to the parallel stripes of the Gabor function, given in degrees, valid values between 0 and 180.
    Phase offset (phi): phase offset of the cosine factor, given in degrees, valid values between -180 and 180.
        The values 0 and 180 correspond to symmetric 'centre-on'  and 'centre-off' functions, respectively, 
        while -90 and 90 correspond to antisymmetric functions.
    Aspect ratio (gamma): ellipticity of the Gaussian factor, 
        values typical of the receptive fields of simple cells lie between 0.2 and 1, typical use of 0.5.
    Bandwidth (b ~ sigma/lambda): spatial-frequency bandwidth (half-response spatial frequency) of the Gabor function, given in octaves, 
        values typical of the receptive fields of simple cells lie between 0.4 and 2.6, median 1.4 in macaques.
    kernelsize: tuple, size of the Gabor kernel, in odd number and square, small for extracting textures and big for segmentation.
    ktype: type and range of values that each pixel in the gabor kernel can hold, ktype=cv2.CV_32F

    Example parameter set:
        Image size = 96*96
        ksize = 3*3, 5*5, 9*9, 15*15  # odd number
        wavelength = (2,2,10)  # pixels, within image size
        orientation = (0,15,180)  #
        phase offset = [0,90]  # symmetric/antisymmetric
        gamma = (0.2,0.1,1)  # [0.2,1]
        bandwidth = (0.4,0.1,2.6)  # ~1.4 for primate V1
    """
    assert isinstance(parameter_set, dict), 'parameter_set must be a dict.'
    list_ksize = parameter_set['ksize']
    list_bandwidth = parameter_set['bandwidth']
    list_aspect_ratio = parameter_set['gamma']
    list_wavelength = parameter_set['wavelength']
    # list_phase_offset = parameter_set['phase']
    list_orientation = parameter_set['orientation']
    Gabor_bank = {}  # output dict of two levels of keys
    count = 0
    for wavelength in list_wavelength:
        for orientation in list_orientation:
            key1_name = 'lambda_{}_theta_{}'.format(wavelength, orientation)
            Gabor_bank[key1_name] = {}
            for phase_offset in [0, 90]:
                key2_name = 'phase_{}'.format(phase_offset)
                Gabor_bank[key1_name][key2_name] = []
                for kernel_size in list_ksize:
                    for bandwidth in list_bandwidth:
                        # bsigma is related to bandwidth
                        bsigma = (wavelength / np.pi) * np.sqrt((np.log(2)) / 2) * ((2**bandwidth) + 1) / ((2**bandwidth) - 1)
                        for aspect_ratio in list_aspect_ratio:
                            gabor_kernel = cv2.getGaborKernel(
                                kernel_size, bsigma, orientation, wavelength, aspect_ratio, phase_offset, 
                                ktype=cv2.CV_64F
                            )
                            Gabor_bank[key1_name][key2_name].append(gabor_kernel)
                            count += 1
    print(count, '2D Gabor kernels generated:', len(Gabor_bank), 'combinations of wavelength/orientation.')
    return Gabor_bank


def get_2d_Gabor_features(image, Gabor_bank, image_shape=None, visualize=False):
    """ Return Gabor features of a grayscale image, given a Gabor bank.
        Three types of Gabor features are returned: 
        (1) response magnitude and variance of each Gabor kernel, i.e. simple_features;
        (2) mean response (image) of all Gabor kernels, i.e. mean_of_all_responses;
        (3) mean response (image) of each pair of Gabor kernels with opposite phase offsets, i.e. Gabor_energy_dict.
    """
    assert len(image.shape) == 2, 'Input image should be grayscale.'
    if image_shape is None:
        image_shape = (96, 96)
    if image.shape != image_shape:
        image = cv2.resize(image, image_shape, interpolation = cv2.INTER_AREA)
    Gabor_bank_groups = list(Gabor_bank.keys())
    num_groups = len(Gabor_bank_groups)
    phase_list = list(Gabor_bank[Gabor_bank_groups[0]].keys())
    num_phases = len(phase_list)
    num_kernels_per_phase = len(Gabor_bank[Gabor_bank_groups[0]][phase_list[0]])
    num_kernels = num_groups * num_phases * num_kernels_per_phase
    # Filter image with Gabor kernels
    simple_features = np.zeros((num_kernels, 2), dtype=np.double)
    mean_of_all_responses = np.zeros_like(image, dtype=np.float64)
    Gabor_energy_dict = {}
    count = 0
    response_list = []
    if np.sum(image) == 0:
        for group_name, gabor_kernel_set in Gabor_bank.items():
            Gabor_energy_dict[group_name] = np.zeros_like(image, dtype=np.float64)
    else:
        for group_name, gabor_kernel_set in Gabor_bank.items():
            Gabor_energy_dict[group_name] = np.zeros_like(image, dtype=np.float64)
            for kernelI in range(num_kernels_per_phase):
                kernel_energy_square = np.zeros_like(image, dtype=np.float64)
                for phaseI in range(num_phases):
                    phase_name = phase_list[phaseI]
                    gabor_kernel = gabor_kernel_set[phase_name][kernelI]
                    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=gabor_kernel)  # cv2.CV_64F
                    if visualize:
                        response_list.append(filtered_image)
                    # Extract features
                    simple_features[count, 0] = np.mean(filtered_image)
                    simple_features[count, 1] = np.std(filtered_image)
                    mean_of_all_responses += filtered_image / num_kernels
                    kernel_energy_square += np.square(filtered_image)
                    count += 1
                kernel_energy = np.sqrt(kernel_energy_square)
                Gabor_energy_dict[group_name] += kernel_energy / num_kernels_per_phase
    return simple_features, mean_of_all_responses, Gabor_energy_dict, response_list


def get_motion_history_image(image_dict, motion_threshold=32, mhi_duration=15):
    """ Compute a single image of motion history from the given list of grayscale images.
            The input dict keys should be consecutive integers.
        motion_threshold is threshold for frame difference, range from 0 to 255, default 32.
        mhi_duration is the time window in num_frames to stack motion history, default 0.5s = 8/15.
        https://github.com/opencv/opencv_contrib/blob/master/modules/optflow/samples/motempl.py
    """
    assert isinstance(image_dict, dict), 'Input should be a dict of images.'
    frame_indices = sorted(list(image_dict.keys()))
    assert len(image_dict[frame_indices[0]].shape) == 2, 'Input image should be grayscale.'
    h, w = image_dict[frame_indices[0]].shape[:2]
    motion_history = np.zeros((h, w), np.float32)
    normalized_MHI_dict = {}
    framestamp = 0
    for frameI in frame_indices:
        if frameI-1 in frame_indices:
            diffI = cv2.absdiff(image_dict[frameI], image_dict[frameI-1])
        else:
            diffI = cv2.absdiff(image_dict[frameI], image_dict[frameI])
        # Apply a threshold to detect motion
        # frameI_max = dict_of_frames[frameI].max()
        _, motion_mask = cv2.threshold(diffI, motion_threshold, 1, cv2.THRESH_BINARY)
        framestamp += 1
        assert motion_mask.dtype == np.uint8, 'Motion mask should be 8-bit single-channel.'
        assert motion_history.dtype == np.float32, 'Motion history should be 32-bit floating-point.'
        cv2.motempl.updateMotionHistory(motion_mask, motion_history, framestamp, mhi_duration)
        # Normalize motion history
        normalized_MHI = np.uint8(np.clip((motion_history - (framestamp - mhi_duration)) / mhi_duration, 0, 1) * 255)
        # print('framestamp', framestamp, 'MHI max', normalized_MHI.max())
        normalized_MHI_dict[frameI] = normalized_MHI
    return normalized_MHI_dict


def get_hof(flow, visualize, num_orientations=9, num_pixels_per_cell=(8, 8),
            num_cells_per_block=(2, 2), motion_threshold=0.5):
    """ Return Histogram of Oriented Optical Flow of a flow image.
    Default parameters are chosen considering both missing rate and speed.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1467360
    
    github: pyKinectTools/algs/HistogramOfOpticalFlow.py
    
    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA
    
    Parameters
    ----------
    Flow : (M, N) ndarray
        Input flow image (x(v) and y(u) flow images).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int, int)
        Number of cells in each block.
        The last block is the no motion block.
    visualise : bool, optional
        Also return an image of the hof.
    normalise : bool, optional
        Apply power law compression to normalise the flow image before processing.
    motion_threshold : threshold for no motion
    
    Output is a flattened feature vector of optical flow histogram 
    (weighted sum of optical flow magnitude for each orientation block).
    
    """
    assert len(flow.shape) == 3, 'Input optical flow image should have three channels.'
    assert flow.shape[-1] == 2, 'Input optical flow image should have u and v values.'
    
    gx = flow[:, :, 1]
    gy = flow[:, :, 0]
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180 # output range [0,180]; without % 180, [-180,180]
    sy, sx = flow.shape[:2]
    cx, cy = num_pixels_per_cell
    bx, by = num_cells_per_block
    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    # if all flow is zero, return zero vector
    if np.sum(magnitude) == 0:
        normalized_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, num_orientations))
        if not visualize:
            hof_image = None
        else:
            hof_image  = np.zeros((sy, sx), dtype=float)
    else:
        # Generate optical flow orientation histogram
        orientation_histogram = np.zeros((n_cellsy, n_cellsx, num_orientations))
        subsample = np.index_exp[int(cy/2): int(cy*n_cellsy) : int(cy), int(cx/2) : int(cx*n_cellsx) : int(cx)]
        for i in range(num_orientations-1):
            # create new integral image for this orientation
            # and isolate orientations in this range
            temp_ori = np.where(orientation < 180 / num_orientations * (i + 1), orientation, -1)
            temp_ori = np.where(orientation >= 180 / num_orientations * i, temp_ori, -1)
            # select magnitudes for those orientations
            cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
            temp_mag = np.where(cond2, magnitude, 0)
            temp_mag = temp_mag.astype('float32')
            temp_filt = uniform_filter(temp_mag, size=(cy, cx))
            orientation_histogram[:, :, i] = temp_filt[subsample]
        # Compute the no-motion bin
        temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)
        temp_mag = temp_mag.astype('float32')
        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, -1] = temp_filt[subsample]
        
        # Normalize histogram
        normalized_blocks = np.zeros((n_blocksy, n_blocksx, by, bx, num_orientations))
        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = orientation_histogram[y:y+by, x:x+bx, :]
                eps = 1e-5
                normalized_blocks[y, x, :] = block / np.sqrt(block.sum()**2 + eps)
        
        # Visualize
        if not visualize:
            hof_image = None
        else:
            radius = min(cx, cy) // 2 - 1
            hof_image = np.zeros((sy, sx), dtype=float)
            for x in range(n_cellsx):
                for y in range(n_cellsy):
                    for o in range(num_orientations-1):
                        center = tuple([y * cy + cy // 2, x * cx + cx // 2])
                        dx = int(radius * np.cos(float(o) / num_orientations * np.pi))
                        dy = int(radius * np.sin(float(o) / num_orientations * np.pi))
                        rr, cc, _ = line_aa(center[0] - dy, center[1] - dx,
                                            center[0] + dy, center[1] + dx)
                        hof_image[rr, cc] += orientation_histogram[y, x, o]
    
    # Flatten to feature vector
    hof_vector = normalized_blocks.flatten()

    # Recover feature array of shape (celly, cellx, orientation)
    # normalized_blocks = block_y, block_x, cell_in_block_y, cell_in_block_x, (9,) histogram
    hof_array = np.zeros((n_blocksy*by, n_blocksx*bx, num_orientations))
    for y in range(n_blocksy):
        for x in range(n_blocksx):
            for b in range(by):
                for a in range(bx):
                    hof_array[y*by+b,x*bx+a,:] = normalized_blocks[y,x,b,a,:]
    # print(hof_array.shape)

    # return magnitude, orientation, normalised_blocks, res_hof, hof_image
    return hof_vector, hof_array, hof_image


def crop_content_from_image(image, mask=None, output_crop_bbox=False):
    """ Crop a (multi-D) image around its non-zero pixel values, i.e. remove empty marginals.
        This function finds the first and last row/column with non-zero pixels, a bounding box,
            then crops the image with this bounding box.
        If a crop_mask is given, it's used as the crop bounding box.
    """
    if mask is None:
        image_data = np.asarray(image)
        background = np.abs(image_data)
    else:
        image_data = np.asarray(image)
        background = np.abs(np.asarray(mask))
        assert image_data.shape[:2] == background.shape[:2], 'Image and mask should have same size.'
    # image_data_abs is used for determining pixels to keep
    if (background > 0).any():
        if len(background.shape) == 3:  # RGB/BGR image or other multi-channel image like optical flow
            image_data_bw = background.max(axis=2)
            non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
            non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
            crop_bbox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
            # crop_bbox contains pixels to keep
            image_cropped = image_data[crop_bbox[0]:crop_bbox[1]+1, crop_bbox[2]:crop_bbox[3]+1, :]
        elif len(background.shape) == 2:  # grayscale image
            image_data_bw = copy.deepcopy(background)
            non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
            non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
            crop_bbox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
            # crop_bbox contains pixels to keep
            image_cropped = image_data[crop_bbox[0]:crop_bbox[1]+1, crop_bbox[2]:crop_bbox[3]+1]
    else:
        crop_bbox = (0, 0, 0, 0)
        image_cropped = image_data
        # print('Image margin cannot be cropped: empty image')
    if output_crop_bbox:
        return crop_bbox
    else:
        return image_cropped


def compute_shape_moments(image, contours=None):
    """ Compute translation, rotation and scale invariant Humoments from a shape image.
    """
    if contours is not None:
        if len(contours) > 0:
            contour = contours[0]
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log scale hu_moments
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        else:
            hu_moments = np.zeros(7)
    else:
        image_data = np.asarray(image, dtype=np.uint8)
        if np.sum(image_data) > 0:
            # Compute moments
            contours, _ = cv2.findContours(image_data, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log scale hu_moments
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))  # 0 in moments will be nan
        else:
            hu_moments = np.zeros(7)
        # for i in range(7):
        #     if np.isnan(hu_moments[i]):
        #         hu_moments[i] = np.inf
    return hu_moments


def get_mask_info(mask, draw_contour=False):
    """ Given a segmentation mask, return its centroid, area, contours, contour_image.
    """
    assert len(mask.shape) == 2, 'Input mask should be a binary image.'
    mask_data = np.asarray(mask, dtype=np.uint8)
    mask_area = cv2.countNonZero(mask_data)
    if mask_area == 0:
        mask_center = np.array([0, 0])
        mask_contours = []
        mask_contour_image = np.zeros(mask_data.shape, dtype=np.uint8)
        mask_Hu_moments = np.zeros(7)
    else:
        center_row_col = np.array([np.average(indices) for indices in np.nonzero(mask_data)])  #, dtype=np.int
        mask_center = np.flip(center_row_col, axis=0)
        mask_contours, _ = cv2.findContours(mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_contour_image = np.zeros(mask_data.shape, dtype=np.uint8)
        if draw_contour:
            cv2.drawContours(mask_contour_image, mask_contours, -1, 255, 1)
        mask_Hu_moments = compute_shape_moments(mask_data, mask_contours)
    return mask_center, mask_area, mask_contours, mask_contour_image, mask_Hu_moments


def get_mask_dict_from_seg_dict(segmentation_dict, segment_key, video_name, segment_filter=None, contour_smoothing=False, target_shape=None):
    """ Given a dict of segmentation results, return a dict of correct masks such that one frame contains an empty or single instance mask.
        Running average is used to fill in empty masks.
        If contour_smoothing, smooth the contour of the mask.
        If target_shape is not None, rescale the mask to the specified size.
        If segment_filter is None, return the first segmentation mask.
            segment_filter is {video set}_{seg name}:
            (1) 'PAO_hand': contour_smoothing 3
                LH & 90/135 & Push/Grasp: max bbox area
                LH & 90/135 & Drop/Scratch: min bbox x1
                LH & 180: max bbox x2
                RH: min bbox y1
            (2) 'PAO_arm': contour_smoothing 8
                LH & 90/135: max bbox area
                LH & 180: max bbox x2
                RH: min bbox y1
            (3) 'PAO_body' & 'PAO_face' & 'PAO_object': contour_smoothing 8/8/3
                All: max bbox area
        output_mask_dict: { frame_index: mask }, mask is a binary mask with 0/1 values of dtype uint8
    """
    assert isinstance(segmentation_dict, dict), 'Input should be a dict of segmentation results.'
    seg_masks_key = f'{segment_key}_pred_masks'
    seg_boxes_key = f'{segment_key}_pred_boxes'  # pred_boxes contains bounding box coordinates in the format of [x1, y1, x2, y2]
    seg_boxes_area_key = f'{segment_key}_pred_boxes_area'
    assert seg_masks_key in segmentation_dict[list(segmentation_dict.keys())[0]].keys(), 'segment_key not found in segmentation_dict.'
    # print(datetime.now(), 'video_name:', video_name, 'segment_key:', segment_key, 'segment_filter:', segment_filter, 
    #       'contour_smoothing:', contour_smoothing, 'target_shape:', target_shape)
    # Sort frame indices
    sorted_frame_indices = sorted(list(segmentation_dict.keys()))
    # Find the correct segmentation mask for each frame
    filtered_segmentation_dict = {}
    for frame_index in sorted_frame_indices:
        seg_masks = segmentation_dict[frame_index][seg_masks_key]
        seg_boxes = segmentation_dict[frame_index][seg_boxes_key]
        seg_boxes_area = segmentation_dict[frame_index][seg_boxes_area_key]
        # Check number of segmentation masks
        if len(seg_masks) > 1:
            ## Multiple masks found for the single object
            ## If segment_filter is None, find the first segmentation mask
            ## Otherwise, find the correct segmentation mask by segment_filter
            if segment_filter is None:
                target_mask_index = 0
            else:
                if segment_filter == 'PAO_hand':
                    ### Check video_name to determine filter
                    if 'RH' in video_name:
                        if '90' in video_name:
                            #### RH: min bbox y1
                            bbox_y1_list = []
                            for idx in range(len(seg_masks)):
                                bbox_y1_list.append(seg_boxes[idx,1])
                            target_mask_index = bbox_y1_list.index(min(bbox_y1_list))
                        else:
                            #### RH: min bbox x1
                            bbox_x1_list = []
                            for idx in range(len(seg_masks)):
                                bbox_x1_list.append(seg_boxes[idx,0])
                            target_mask_index = bbox_x1_list.index(min(bbox_x1_list))
                    elif '180' in video_name:
                        #### LH & 180: max bbox x2
                        bbox_x2_list = []
                        for idx in range(len(seg_masks)):
                            bbox_x2_list.append(seg_boxes[idx,2])
                        target_mask_index = bbox_x2_list.index(max(bbox_x2_list))
                    elif 'drop' in video_name or 'scratch' in video_name:
                        #### LH & 90/135 & Drop/Scratch: min bbox x1
                        bbox_x1_list = []
                        for idx in range(len(seg_masks)):
                            bbox_x1_list.append(seg_boxes[idx,0])
                        target_mask_index = bbox_x1_list.index(min(bbox_x1_list))
                    else:
                        #### LH & 90/135 & Push/Grasp: max mask size
                        bbox_area_list = []
                        for idx in range(len(seg_masks)):
                            tmp_mask_size = np.sum(seg_masks[idx])
                            bbox_area_list.append(tmp_mask_size)
                        target_mask_index = bbox_area_list.index(max(bbox_area_list))
                elif segment_filter == 'PAO_arm':
                    ### Check video_name to determine filter
                    if 'RH' in video_name:
                        #### RH: min bbox y1
                        bbox_y1_list = []
                        for idx in range(len(seg_masks)):
                            bbox_y1_list.append(seg_boxes[idx,1])
                        target_mask_index = bbox_y1_list.index(min(bbox_y1_list))
                    elif '180' in video_name:
                        #### LH & 180: max bbox x2
                        bbox_x2_list = []
                        for idx in range(len(seg_masks)):
                            bbox_x2_list.append(seg_boxes[idx,2])
                        target_mask_index = bbox_x2_list.index(max(bbox_x2_list))
                    else:
                        #### LH & 90/135: max mask size
                        bbox_area_list = []
                        for idx in range(len(seg_masks)):
                            tmp_mask_size = np.sum(seg_masks[idx])
                            bbox_area_list.append(tmp_mask_size)
                        target_mask_index = bbox_area_list.index(max(bbox_area_list))
                else:
                    #### All: max mask size
                    bbox_area_list = []
                    for idx in range(len(seg_masks)):
                        tmp_mask_size = np.sum(seg_masks[idx])
                        bbox_area_list.append(tmp_mask_size)
                    target_mask_index = bbox_area_list.index(max(bbox_area_list))
            target_mask = np.asarray(seg_masks[target_mask_index], dtype=np.uint8)  # [0,1] binary mask
            if contour_smoothing:
                tmp_mask_0 = target_mask * 255
                if 'object' in segment_key or 'hand' in segment_key:
                    tmp_mask_1 = cv2.GaussianBlur(tmp_mask_0, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
                else:
                    tmp_mask_1 = cv2.GaussianBlur(tmp_mask_0, (0,0), sigmaX=8, sigmaY=8, borderType = cv2.BORDER_DEFAULT)
                target_mask = tmp_mask_1 > 127.5
                target_mask = np.asarray(target_mask, dtype=np.uint8)  # [0,1] binary mask
        elif len(seg_masks) == 1:
            ## Only one mask, use it directly
            target_mask = np.asarray(seg_masks[0], dtype=np.uint8)  # [0,1] binary mask
            if contour_smoothing:
                tmp_mask_0 = target_mask * 255
                if 'object' in segment_key or 'hand' in segment_key:
                    tmp_mask_1 = cv2.GaussianBlur(tmp_mask_0, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
                else:
                    tmp_mask_1 = cv2.GaussianBlur(tmp_mask_0, (0,0), sigmaX=8, sigmaY=8, borderType = cv2.BORDER_DEFAULT)
                target_mask = tmp_mask_1 > 127.5
                target_mask = np.asarray(target_mask, dtype=np.uint8)  # [0,1] binary mask
        else:
            ## No mask found, leave it None for now
            target_mask = None
        filtered_segmentation_dict[frame_index] = target_mask
    # Fill in empty masks by running average
    output_mask_dict = {}
    for frame_index, filtered_mask in filtered_segmentation_dict.items():
        if filtered_mask is None:
            print('No filtered_mask found for frame_index', frame_index, 'in video', video_name, ', fill in mask by running average.')
            ## No mask found, check if there exists a previous frame mask and a following frame mask,
            ## if so, copy the previous mask but move it toward the following mask by a step proportional to frame distance,
            ## if not, leave it None
            pre_mask = None
            if frame_index > 0:
                for pre_frame_index in range(frame_index, 0, -1):
                    if filtered_segmentation_dict[pre_frame_index] is not None:
                        pre_mask = filtered_segmentation_dict[pre_frame_index]
                        print('-- pre_mask searched in frame_index', pre_frame_index, 'mask area', np.sum(pre_mask))
                        break
            post_mask = None
            if frame_index < sorted_frame_indices[-1]:
                for post_frame_index in range(frame_index+1, sorted_frame_indices[-1]+1):
                    if filtered_segmentation_dict[post_frame_index] is not None:
                        post_mask = filtered_segmentation_dict[post_frame_index]
                        print('-- post_mask searched in frame_index', post_frame_index, 'mask area', np.sum(post_mask))
                        break
            if pre_mask is not None and post_mask is not None:
                ### Compute frame_index distance to previous and following masks
                print('---- calculating move step')
                frame_distance_to_pre = frame_index - pre_frame_index
                frame_distance_to_post = post_frame_index - frame_index
                # pre_bbox = cv2.boundingRect(pre_mask)
                # post_bbox = cv2.boundingRect(post_mask)
                pre_bbox = crop_content_from_image(pre_mask, mask=None, output_crop_bbox=True)
                post_bbox = crop_content_from_image(post_mask, mask=None, output_crop_bbox=True)
                pre_bbox_x1, pre_bbox_y1 = pre_bbox[0], pre_bbox[1]
                post_bbox_x1, post_bbox_y1 = post_bbox[0], post_bbox[1]
                ### Compute number of pixels to move along x and y axis
                move_x = int((post_bbox_x1 - pre_bbox_x1) * frame_distance_to_pre / (frame_distance_to_pre + frame_distance_to_post))
                move_y = int((post_bbox_y1 - pre_bbox_y1) * frame_distance_to_pre / (frame_distance_to_pre + frame_distance_to_post))
                print('---- pre_bbox', pre_bbox, 'post_bbox', post_bbox, 'move_x', move_x, 'move_y', move_y)
                ### Move pre_mask from pre_bbox to new_bbox
                new_mask = np.zeros_like(pre_mask)
                count_pixels_moved = 0
                for row in range(pre_mask.shape[0]):
                    for col in range(pre_mask.shape[1]):
                        if pre_mask[row,col] > 0:
                            if row+move_y < 0 or row+move_y >= new_mask.shape[0] or col+move_x < 0 or col+move_x >= new_mask.shape[1]:
                                pass
                            else:
                                new_mask[row+move_y, col+move_x] = 1
                                count_pixels_moved += 1
                print('-- Successfully moved', count_pixels_moved, 'pixels.')
                ### Rescale if target_shape is not None
                if target_shape is not None:
                    new_mask = cv2.resize(new_mask, target_shape, interpolation=cv2.INTER_AREA)
                output_mask_dict[frame_index] = new_mask  # [0,1] binary mask
            else:
                output_mask_dict[frame_index] = None
        else:
            ### Rescale if target_shape is not None
            if target_shape is not None:
                filtered_mask = cv2.resize(filtered_mask, target_shape, interpolation=cv2.INTER_AREA)
            output_mask_dict[frame_index] = filtered_mask  # [0,1] binary mask
    return output_mask_dict


def get_mask_trajectory(mask_dict, smooth=True, smooth_kernel_sigma=2, contour_image=False, 
                        output_folder=None, output_prefix=None, output_suffix=None, mask_label=None):
    """ Given a dict of segmentation results, return a list of mask centers, area, 
            shape, contours, contour_images, shape moments, motion degree and velocity.
    """
    assert isinstance(mask_dict, dict), 'Input should be a dict of segmentation masks.'
    assert len(mask_dict[list(mask_dict.keys())[0]].shape) == 2, 'Input mask should be a binary image.'
    feature_list = ['center', 'size', 'shape', 'contours', 'contour-image', 'shape-moments', 'motion-degree', 'velocity']
    mask_trajectory_features = {key: {} for key in feature_list}
    raw_center_x = []
    raw_center_y = []
    pre_center = np.zeros(2)
    for frame_index, mask in mask_dict.items():
        seg_mask = mask.astype('uint8')
        seg_center, seg_area, seg_contours, seg_contour_image, seg_Hu_moments = get_mask_info(seg_mask, draw_contour=contour_image)
        seg_motion_deg = 0.0
        seg_vel = 0.0
        if frame_index > 0:
            delta_cx = seg_center[0] - pre_center[0]
            delta_cy = seg_center[1] - pre_center[1]
            if any([delta_cx != 0, delta_cy != 0]):
                seg_motion_deg = compute_vector_angle([delta_cx, delta_cy], [1, 0])  # baseline (0 deg) = rightward ([1, 0])
                seg_vel = np.linalg.norm([delta_cx, delta_cy])
        pre_center = copy.deepcopy(seg_center)
        raw_center_x.append(seg_center[0])
        raw_center_y.append(seg_center[1])
        mask_trajectory_features['center'][frame_index] = seg_center
        mask_trajectory_features['size'][frame_index] = seg_area
        mask_trajectory_features['shape'][frame_index] = seg_mask
        mask_trajectory_features['contours'][frame_index] = seg_contours
        mask_trajectory_features['contour-image'][frame_index] = seg_contour_image  # binary mask of the outline
        mask_trajectory_features['shape-moments'][frame_index] = seg_Hu_moments
        mask_trajectory_features['motion-degree'][frame_index] = seg_motion_deg
        mask_trajectory_features['velocity'][frame_index] = seg_vel
    # Smooth trajectory
    if smooth:
        smooth_center_x = gaussian_filter1d(raw_center_x, sigma=smooth_kernel_sigma)
        smooth_center_y = gaussian_filter1d(raw_center_y, sigma=smooth_kernel_sigma)
        # Update motion degree and velocity
        for frame_i, frame_index in enumerate(sorted(list(mask_trajectory_features['center'].keys()))):
            mask_trajectory_features['center'][frame_index] = np.array([smooth_center_x[frame_i], smooth_center_y[frame_i]])
            if frame_index > 0:
                delta_cx = smooth_center_x[frame_i] - smooth_center_x[frame_i-1]
                delta_cy = smooth_center_y[frame_i] - smooth_center_y[frame_i-1]
                if any([delta_cx != 0, delta_cy != 0]):
                    motion_deg = compute_vector_angle([delta_cx, delta_cy], [1, 0])  # baseline (0 deg) = rightward ([1, 0])
                    velocity = np.linalg.norm([delta_cx, delta_cy])
                else:
                    motion_deg = 0.0
                    velocity = 0.0
                mask_trajectory_features['motion-degree'][frame_index] = motion_deg
                mask_trajectory_features['velocity'][frame_index] = velocity
    # Save dict to npz
    if output_folder is not None:
        if output_prefix is None:
            output_prefix = ''
        else:
            output_prefix += '_'
        if output_suffix is None:
            output_suffix = ''
        else:
            output_suffix = '_' + output_suffix
        if mask_label is None:
            mask_label = ''
        else:
            mask_label += '-'
        np.savez(
            os.path.join(output_folder, f'{output_prefix}{mask_label}mask-trajectory{output_suffix}.npz'), 
            Results=mask_trajectory_features
        )
    return mask_trajectory_features


def extract_features(image_dict, feature_list, spatial_window_dict=None, mask_dict=None, mask_label=None, 
                     rescale_factor=None, crop_after_masking=False, 
                     smooth_mask_trajectory=False, flow_dict=None, Gabor_bank=None, 
                     output_folder=None, output_prefix=None, output_suffix=None):
    """ Extract features (feature_list) from images (masked if mask_dict is provided, rescaled if rescale_factor is provided).

        There are two types of mask dict: spatial_window_dict and mask_dict.
            spatial_window_dict is used to crop the image such that the result shape is consistent across videos.
            mask_dict is used to mask the image to get the region of interest feature.
        In practice: 
            For global features, one fixed spatial window is applied to all videos to remove empty background, 
                and there is no other mask to apply, hence: 
                    spatial_window_dict is used, mask_dict = None, 
                    both masking and crop_after_masking use spatial_window_dict.
            For local features, one fixed spatial window is composed of the union of segmentations across videos,
                and each video has its own mask to apply, hence:
                    spatial_window_dict = union of masks, mask_dict = single mask, 
                    masking uses mask_dict, crop_after_masking uses spatial_window_dict.
            If no spatial_window_dict given, only mask_dict give,
                both masking and crop_after_masking use mask_dict.
            If no spatial_window_dict or mask_dict given,
                neither masking nor crop_after_masking is applied.
        
        If Gabor_bank is provided, Gabor features are also extracted.
        If flow_dict is provided, motion features based on optical flow are also extracted.

        Output filename format: {output_prefix}feature-{mask_label}-{feature_name}{output_suffix}.npz
    """
    assert isinstance(image_dict, dict), 'Input should be a dict of images.'
    assert len(feature_list) > 0, 'feature_list should not be empty.'
    assert isinstance(feature_list, list) and isinstance(feature_list[0], str), 'feature_list should be a list of strings.'
    # Initialize output dict, key = 'feature-{mask_label}-{feature_name}'
    output_dict = {}
    if mask_label is None:
        mask_label = 'image'
    for feature_name in feature_list:
        output_dict[f'feature-{mask_label}-{feature_name}'] = {}
    # Extract features
    fullsize_altered_image_dict = {}  # masked but not cropped or rescaled
    altered_image_dict = {}
    altered_mask_dict = {}
    count = 0
    for frame_index, image in image_dict.items():
        image_data = copy.deepcopy(image)
        if flow_dict is not None:
            if frame_index in flow_dict.keys():
                flow_data = copy.deepcopy(flow_dict[frame_index])
            else:
                flow_data = None
        else:
            flow_data = None
        raw_image_shape = image.shape[:2]
        ## Prepare masker to mask image and cropper to crop masked image
        if spatial_window_dict is not None and mask_dict is None:
            ### Mask and crop image by spatial_window_dict
            masker = np.asarray(spatial_window_dict[frame_index], dtype=np.uint8)
            cropper = np.asarray(spatial_window_dict[frame_index], dtype=np.uint8)
        elif spatial_window_dict is not None and mask_dict is not None:
            ### Mask image by mask_dict and crop masked image by spatial_window_dict
            masker = np.asarray(mask_dict[frame_index], dtype=np.uint8)
            cropper = np.asarray(spatial_window_dict[frame_index], dtype=np.uint8)
        elif spatial_window_dict is None and mask_dict is not None:
            ### Mask and crop image by mask_dict
            masker = np.asarray(mask_dict[frame_index], dtype=np.uint8)
            cropper = np.asarray(mask_dict[frame_index], dtype=np.uint8)
        else:
            ### No masker or cropper
            masker = None
            cropper = None
        ## Check if image should be masked
        if masker is not None:
            ### Mask image
            assert masker.shape == image_data.shape, 'Image and masker should have same shape.'
            image_data = np.multiply(image_data, masker)
            fullsize_altered_image_dict[frame_index] = copy.deepcopy(image_data)
            ### Mask optical flow
            if flow_data is not None:
                assert masker.shape == flow_data.shape[:2], 'Flow and masker should have same shape.'
                flow_data = np.multiply(flow_data, np.tile(masker[...,np.newaxis], (1,1,2)))
            ### Save mask (mask does not go through rescaling or cropping to extract trajectory features)
            altered_mask_dict[frame_index] = masker
        ## Check if masked image should be cropped
        if crop_after_masking and cropper is not None:
            if np.sum(cropper) > 0:
                image_data = crop_content_from_image(image_data, mask=cropper)
                if flow_data is not None:
                    flow_data = crop_content_from_image(flow_data, mask=cropper)
        ## Check if masked image should be rescaled to a fixed size (e.g. float rescale_factor * input image shape)
        if rescale_factor is not None:
            ### if rescale_factor is a tuple of float, multiply each image shape by it
            ### if rescale_factor is a tuple of integer, use it as output image shape
            ### if rescale_factor is a float, multiply both image shape by it
            ### if rescale_factor is an integer, use it as the square output image shape
            if isinstance(rescale_factor, tuple):
                if isinstance(rescale_factor[0], float):
                    image_shape = (int(raw_image_shape[1]*rescale_factor[0]), int(raw_image_shape[0]*rescale_factor[1]))  # (width, height)
                elif isinstance(rescale_factor[0], int):
                    image_shape = rescale_factor
            elif isinstance(rescale_factor, float):
                image_shape = (int(raw_image_shape[1]*rescale_factor), int(raw_image_shape[0]*rescale_factor))  # (width, height)
            elif isinstance(rescale_factor, int):
                image_shape = (rescale_factor, rescale_factor)  # (width, height)
            ## Resize image
            image_data = cv2.resize(image_data, image_shape, interpolation = cv2.INTER_AREA)
            ## Resize optical flow
            if flow_data is not None:
                flow_data = cv2.resize(flow_data, image_shape, interpolation = cv2.INTER_AREA)
        ## Save modified image to a dict
        altered_image_dict[frame_index] = image_data
        ## Extract features
        if 'luminance' in feature_list or 'contrast' in feature_list:
            image_luminance, image_contrast = get_luminance_contrast(image_data)
            if 'luminance' in feature_list:
                output_dict[f'feature-{mask_label}-luminance'][frame_index] = image_luminance
            if 'contrast' in feature_list:
                output_dict[f'feature-{mask_label}-contrast'][frame_index] = image_contrast
        if 'Sobel-gradients' in feature_list:
            output_dict[f'feature-{mask_label}-Sobel-gradients'][frame_index] = get_Sobel_gradients(image_data, smooth_kernel_size=3)
            ### Check gradients min, max, mean, std
            if count % 30 == 0:
                print(datetime.now(), '---- frame {}, feature {}, min {}, max {}, mean {}, std {}'.format(
                    frame_index, 'Sobel-gradients',
                    output_dict[f'feature-{mask_label}-Sobel-gradients'][frame_index].min(),
                    output_dict[f'feature-{mask_label}-Sobel-gradients'][frame_index].max(),
                    output_dict[f'feature-{mask_label}-Sobel-gradients'][frame_index].mean(),
                    output_dict[f'feature-{mask_label}-Sobel-gradients'][frame_index].std(),
                ))
        if 'Canny-edges' in feature_list:
            output_dict[f'feature-{mask_label}-Canny-edges'][frame_index] = get_Canny_edges(image_data, min_thr=50, max_thr=200, smooth_kernel_size=3)
            ### Check edges min, max, mean, std
            if count % 30 == 0:
                print(datetime.now(), '---- frame {}, feature {}, min {}, max {}, mean {}, std {}'.format(
                    frame_index, 'Canny-edges',
                    output_dict[f'feature-{mask_label}-Canny-edges'][frame_index].min(),
                    output_dict[f'feature-{mask_label}-Canny-edges'][frame_index].max(),
                    output_dict[f'feature-{mask_label}-Canny-edges'][frame_index].mean(),
                    output_dict[f'feature-{mask_label}-Canny-edges'][frame_index].std(),
                ))
        if 'FAST-corners' in feature_list:
            # list_FAST_corners, mask_FAST_corners, vis_FAST_corners
            _, masked_FAST_corners, _ = get_FAST_corners(image_data, visualize=False, fast_thr=10, fast_nonmaxSuppression=True)
            output_dict[f'feature-{mask_label}-FAST-corners'][frame_index] = masked_FAST_corners
            ### Check number of corners = number of non-zero pixels
            if count % 30 == 0:
                print(datetime.now(), '---- frame {}, feature {}, number of corners: {}'.format(
                    frame_index, 'FAST-corners', np.sum(masked_FAST_corners)
                ))
        if any(['Gabor' in x for x in feature_list]):
            Gabor_simple_features, Gabor_mean_response, Gabor_energy_features, _ = get_2d_Gabor_features(image_data, Gabor_bank, image_shape=(96,96), visualize=False)
            if 'Gabor-simple-features' in feature_list:
                output_dict[f'feature-{mask_label}-Gabor-simple-features'][frame_index] = Gabor_simple_features
            if 'Gabor-mean-response' in feature_list:
                output_dict[f'feature-{mask_label}-Gabor-mean-response'][frame_index] = Gabor_mean_response
            if 'Gabor-energy-features' in feature_list:
                output_dict[f'feature-{mask_label}-Gabor-energy-features'][frame_index] = Gabor_energy_features
            ### Check mean response min, max, mean, std
            if count % 30 == 0:
                print(datetime.now(), '---- frame {}, feature {}, min {}, max {}, mean {}, std {}'.format(
                    frame_index, 'Gabor-mean-response',
                    Gabor_mean_response.min(), Gabor_mean_response.max(),
                    Gabor_mean_response.mean(), Gabor_mean_response.std(),
                ))
        if any(['HOG' in x for x in feature_list]):
            if 'HOG-image' in feature_list:
                HOG_vector, HOG_array, HOG_image = get_hog(image_data, visualize=True)
                output_dict[f'feature-{mask_label}-HOG-image'][frame_index] = HOG_image
            else:
                HOG_vector, HOG_array, _ = get_hog(image_data, visualize=False)
            if 'HOG-vector' in feature_list:
                output_dict[f'feature-{mask_label}-HOG-vector'][frame_index] = HOG_vector
            if 'HOG-array' in feature_list:
                output_dict[f'feature-{mask_label}-HOG-array'][frame_index] = HOG_array
            ### Check HOG vector min, max, mean, std
            if count % 30 == 0:
                if len(HOG_vector) > 0:
                    print(datetime.now(), '---- frame {}, feature {}, min {}, max {}, mean {}, std {}'.format(
                        frame_index, 'HOG-vector',
                        HOG_vector.min(), HOG_vector.max(),
                        HOG_vector.mean(), HOG_vector.std(),
                    ))
                else:
                    print(datetime.now(), '---- frame {}, feature {}, empty vector'.format(frame_index, 'HOG-vector'))
        if flow_data is not None:
            if 'flow' in feature_list:
                output_dict[f'feature-{mask_label}-flow'][frame_index] = flow_data
            if any(['HOF' in x for x in feature_list]):
                if 'HOF-image' in feature_list:
                    HOF_vector, HOF_array, HOF_image = get_hof(flow_data, visualize=True)
                    output_dict[f'feature-{mask_label}-HOF-image'][frame_index] = HOF_image
                else:
                    HOF_vector, HOF_array, _ = get_hof(flow_data, visualize=False)
                if 'HOF-vector' in feature_list:
                    output_dict[f'feature-{mask_label}-HOF-vector'][frame_index] = HOF_vector
                if 'HOF-array' in feature_list:
                    output_dict[f'feature-{mask_label}-HOF-array'][frame_index] = HOF_array
                ### Check HOF vector min, max, mean, std
                if count % 30 == 0:
                    print(datetime.now(), '---- frame {}, feature {}, min {}, max {}, mean {}, std {}'.format(
                        frame_index, 'HOF-vector',
                        HOF_vector.min(), HOF_vector.max(),
                        HOF_vector.mean(), HOF_vector.std(),
                    ))
        count += 1
    if 'MHI' in feature_list:
        ### Check if altered_image_dict images have the same shape, otherwise, use fullsize_altered_image_dict
        altered_image_same_shape = True
        first_frame_index = sorted(list(altered_image_dict.keys()))[0]
        for frame_index, image in altered_image_dict.items():
            if image.shape != altered_image_dict[first_frame_index].shape:
                altered_image_same_shape = False
                break
        if altered_image_same_shape:
            MHI_dict = get_motion_history_image(altered_image_dict, motion_threshold=32, mhi_duration=15)
        else:
            MHI_dict = get_motion_history_image(fullsize_altered_image_dict, motion_threshold=32, mhi_duration=15)
        for frame_index, MHI_image in MHI_dict.items():
            output_dict[f'feature-{mask_label}-MHI'][frame_index] = MHI_image
    if mask_dict is not None:
        if any(['shape' in x for x in feature_list]):
            if 'contour-image' in feature_list:
                mask_trajectory_features = get_mask_trajectory(
                    altered_mask_dict, smooth=smooth_mask_trajectory, contour_image=True, 
                    mask_label=None, output_folder=None, output_prefix=None, output_suffix=None
                )
                for frame_index, mask_contour_image in mask_trajectory_features['contour-image'].items():
                    output_dict[f'feature-{mask_label}-contour-image'][frame_index] = mask_contour_image
            else:
                mask_trajectory_features = get_mask_trajectory(
                    altered_mask_dict, smooth=smooth_mask_trajectory, contour_image=False, 
                    mask_label=None, output_folder=None, output_prefix=None, output_suffix=None
                )
            if 'center' in feature_list:
                for frame_index, mask_center in mask_trajectory_features['center'].items():
                    output_dict[f'feature-{mask_label}-center'][frame_index] = mask_center
            if 'size' in feature_list:
                for frame_index, mask_size in mask_trajectory_features['size'].items():
                    output_dict[f'feature-{mask_label}-size'][frame_index] = mask_size
            if 'shape' in feature_list:
                for frame_index, mask_shape in mask_trajectory_features['shape'].items():
                    output_dict[f'feature-{mask_label}-shape'][frame_index] = mask_shape
            if 'contours' in feature_list:
                for frame_index, mask_contours in mask_trajectory_features['contours'].items():
                    output_dict[f'feature-{mask_label}-contours'][frame_index] = mask_contours
            if 'shape-moments' in feature_list:
                for frame_index, mask_Hu_moments in mask_trajectory_features['shape-moments'].items():
                    output_dict[f'feature-{mask_label}-shape-moments'][frame_index] = mask_Hu_moments
            if 'motion-degree' in feature_list:
                for frame_index, mask_motion_degree in mask_trajectory_features['motion-degree'].items():
                    output_dict[f'feature-{mask_label}-motion-degree'][frame_index] = mask_motion_degree
            if 'velocity' in feature_list:
                for frame_index, mask_velocity in mask_trajectory_features['velocity'].items():
                    output_dict[f'feature-{mask_label}-velocity'][frame_index] = mask_velocity
    # Save features
    if output_folder is not None:
        if output_prefix is None:
            output_prefix = ''
        else:
            output_prefix = f'{output_prefix}_'  # add '_'
        if output_suffix is None:
            output_suffix = ''
        else:
            output_suffix = f'_{output_suffix}'  # add '_'
        for feature_name, feature_dict in output_dict.items():
            output_file_name = f'{output_prefix}{feature_name}{output_suffix}.npz'
            np.savez(os.path.join(output_folder, output_file_name), Results=feature_dict)
    return output_dict


def filter_global_features(global_feature_dict, feature_name, 
                           local_spatial_window_dict=None, local_mask_dict=None, local_mask_label=None,
                           local_rescale_factor=None, local_crop_after_masking=True,
                           output_folder=None, output_prefix=None, output_suffix=None):
    """ Filter global features (global_feature_dict) by local masks. 
        global_feature_dict is a dict of feature_name (e.g. feature-global-HOG-array) extracted from a single video.
        Its keys are frame indices and values are feature arrays (array dimension varies).
    """
    assert isinstance(global_feature_dict, dict), 'Input should be a dict of global features.'
    assert 'global' in feature_name, 'feature_name should contain "global".'  # e.g. feature-global-HOG-array
    if local_rescale_factor is not None:
        raise ValueError('local_rescale_factor is not supported yet.')
    # Check which type of feature first
    # if feature has the same spatial size as the image
    check_str_1 = [substr in feature_name for substr in [
        'Sobel-gradients', 'Canny-edges', 'FAST-corners', 'HOG-image', 'HOF-image', 'flow', 'MHI', 
    ]]
    # if feature is Gabor-mean-response which takes in a rescaled (96, 96) image as input
    check_str_2 = [substr in feature_name for substr in ['Gabor-mean-response']]
    # if feature is HOG-array or HOF-array
    check_str_3 = [substr in feature_name for substr in ['HOG-array', 'HOF-array']]
    # if feature is HOG-vector or HOF-vector
    check_str_4 = [substr in feature_name for substr in ['HOG-vector', 'HOF-vector']]
    # if none of the above contains True element
    if not any(check_str_1 + check_str_2 + check_str_3 + check_str_4):
        raise ValueError('feature_name not recognized.')
    else:
        local_feature_dict = {}
        count = 0
        for frame_index, global_feature_array in global_feature_dict.items():
            if local_mask_dict is not None:
                local_masker = local_mask_dict[frame_index]
            if local_spatial_window_dict is not None:
                local_cropper = local_spatial_window_dict[frame_index]
            empty_local_cropper = False
            if local_cropper is not None:
                ## Check if local_cropper is empty
                if np.sum(local_cropper) == 0:
                    empty_local_cropper = True
            # # Check target shape
            # if local_rescale_factor is not None:
            #     ### if rescale_factor is a tuple of float, multiply each image shape by it
            #     ### if rescale_factor is a tuple of integer, use it as output image shape
            #     ### if rescale_factor is a float, multiply both image shape by it
            #     ### if rescale_factor is an integer, use it as the square output image shape
            #     if isinstance(local_rescale_factor, tuple):
            #         if isinstance(local_rescale_factor[0], float):
            #             target_shape = (int(local_cropper[1]*local_rescale_factor[0]), int(local_cropper[0]*local_rescale_factor[1]))  # (width, height)
            #         elif isinstance(local_rescale_factor[0], int):
            #             target_shape = local_rescale_factor
            #     elif isinstance(local_rescale_factor, float):
            #         target_shape = (int(local_cropper[1]*local_rescale_factor), int(local_cropper[0]*local_rescale_factor))  # (width, height)
            #     elif isinstance(local_rescale_factor, int):
            #         target_shape = (local_rescale_factor, local_rescale_factor)  # (width, height)
            if not empty_local_cropper:
                ## Apply local_masker and local_cropper to filter global feature
                if any(check_str_1):
                    ### Feature is a 2D or 3D array with the first two dimensions being the same as the masked and cropped image
                    if local_masker is not None:
                        ### mask global_feature_array by local_masker
                        if len(global_feature_array.shape) == 2:
                            local_feature_array = np.multiply(global_feature_array, local_masker)
                        elif len(global_feature_array.shape) == 3:
                            local_feature_array = np.multiply(global_feature_array, np.tile(local_masker[...,np.newaxis], (1,1,global_feature_array.shape[2])))
                        else:
                            raise ValueError('global_feature_array should be a 2D or 3D array for feature {feature_name} in frame {frame_index}')
                    else:
                        local_feature_array = np.zeros_like(global_feature_array)
                    if local_crop_after_masking:
                        local_feature_array = crop_content_from_image(local_feature_array, mask=local_cropper)
                    local_feature_dict[frame_index] = local_feature_array
                elif any(check_str_2):
                    ### Gabor-mean-response
                    ### (1) Rescale global_feature_array from (96, 96) to (width, height) of [global_cropper > 0]
                    ### (2) mask rescaled_global_feature_array by local_masker and crop by local_cropper
                    # global_height, global_width = global_feature_array.shape[:2]
                    global_height, global_width = local_cropper.shape[:2]
                    # starttime = time.process_time()
                    rescaled_global_feature_array = cv2.resize(global_feature_array, (global_width, global_height), interpolation = cv2.INTER_NEAREST)
                    # endtime = time.process_time()
                    # print(endtime - starttime, 'seconds spent on rescaling global Gabor-mean-response')
                    if local_masker is not None:
                        local_feature_array = np.multiply(rescaled_global_feature_array, local_masker)
                    else:
                        local_feature_array = np.zeros_like(global_feature_array)
                    if local_crop_after_masking:
                        local_feature_array = crop_content_from_image(local_feature_array, mask=local_cropper)
                    local_feature_dict[frame_index] = local_feature_array
                elif any(check_str_3) or any(check_str_4):
                    ### HOG-array or HOF-array
                    ### Assume HOG or HOF is extracted using the default parameters
                    num_orientations=9
                    num_pixels_per_cell=(8, 8)
                    num_cells_per_block=(2, 2)
                    cx, cy = num_pixels_per_cell
                    bx, by = num_cells_per_block
                    global_height, global_width = local_cropper.shape[:2]
                    ### Pick out cells that are located in local_masker
                    ### Example: global_cropper_content = 125*170, global_blocksx = 14, global_blocksy = 20, global_hogx = 28, global_hogy = 40,
                    ### Given a pixel (x, y) in global_cropper_content, compute the number of blocks in x, y until this pixel,
                    ### Then block index = the number. If the pixel is in local_masker, extract the corresponding block conten = 2x2x9.
                    ### Initialize np array for all possible cells in local_cropper
                    global_cell_pixel_list = {}
                    for y in range(global_height):
                        for x in range(global_width):
                            ### Compute number of cells from top left corner to the current pixel
                            global_n_cellsx = int(np.floor(x // cx))  
                            global_n_cellsy = int(np.floor(y // cy))
                            ### Compute the index of the block to which the current pixel belongs
                            global_block_x = global_n_cellsx - bx
                            global_block_y = global_n_cellsy - by
                            ### Return a list of pixel (y,x) for each cell
                            for b in range(by):
                                for a in range(bx):
                                    global_cell_key = (global_block_y*by+b, global_block_x*bx+a)
                                    if global_cell_key not in global_cell_pixel_list.keys():
                                        global_cell_pixel_list[global_cell_key] = []
                                    else:
                                        global_cell_pixel_list[global_cell_key].append(tuple([y,x]))
                    ### Loop through cells, if cell contains a pixel in local_masker, read its content from global_feature_array
                    ### elif cell contains a pixel only in cropper_masker, write a zero array
                    ### else, do nothing
                    local_cell_content_list = {}
                    for cell_key, cell_pixel_list in global_cell_pixel_list.items():
                        if any([local_masker[pixel[0], pixel[1]] > 0 for pixel in cell_pixel_list]):
                            local_cell_content_list[cell_key] = global_feature_array[cell_key[0], cell_key[1], :]
                        else:
                            if any([local_cropper[pixel[0], pixel[1]] > 0 for pixel in cell_pixel_list]):
                                local_cell_content_list[cell_key] = np.zeros((num_orientations,))
                    ### Reconstruct np array from local_cell_content_list
                    unique_local_cell_0 = sorted(list(set([cell_key[0] for cell_key in local_cell_content_list.keys()])))
                    unique_local_cell_1 = sorted(list(set([cell_key[1] for cell_key in local_cell_content_list.keys()])))
                    local_feature_array = np.zeros((len(unique_local_cell_0), len(unique_local_cell_1), num_orientations))
                    for x_i, x in enumerate(unique_local_cell_0):
                        for y_i, y in enumerate(unique_local_cell_1):
                            if (x,y) in local_cell_content_list.keys():
                                local_feature_array[x_i, y_i, :] = local_cell_content_list[(x, y)]
                            else:
                                local_feature_array[x_i, y_i, :] = np.zeros((num_orientations,))
                    if any(check_str_3):
                        local_feature_dict[frame_index] = local_feature_array
                    else:
                        local_feature_dict[frame_index] = local_feature_array.flatten()
            else:
                ## Empty local_cropper
                local_feature_array = np.zeros_like(global_feature_array)
                if local_crop_after_masking:
                    local_feature_array = crop_content_from_image(local_feature_array, mask=local_cropper)
                local_feature_dict[frame_index] = local_feature_array
            ## Print result
            if count % 20 == 0:
                try:
                    print(datetime.now(), '---- frame {}, filter {}, get min {}, max {}, mean {}, std {}'.format(
                        frame_index, feature_name,
                        np.min(local_feature_dict[frame_index]), np.max(local_feature_dict[frame_index]),
                        np.mean(local_feature_dict[frame_index]), np.std(local_feature_dict[frame_index]),
                    ))
                except:
                    print(datetime.now(), '---- frame {}, filter {}, get empty array'.format(frame_index, feature_name))
            count += 1
    # Save features
    if output_folder is not None:
        if output_prefix is None:
            output_prefix = ''
        else:
            output_prefix = f'{output_prefix}_'
        if output_suffix is None:
            output_suffix = ''
        else:
            output_suffix = f'_{output_suffix}'
        if local_mask_label is None:
            local_mask_label = 'local'
        local_feature_name = feature_name.replace('global', local_mask_label)
        output_file_name = f'{output_prefix}{local_feature_name}{output_suffix}.npz'
        np.savez(os.path.join(output_folder, output_file_name), Results=local_feature_dict)
    return local_feature_dict

