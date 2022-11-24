import numpy as np

from nibabel import Nifti1Image
from nilearn.maskers import NiftiMasker
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker
from nilearn.masking import compute_multi_epi_mask
from nilearn.datasets import fetch_icbm152_brain_gm_mask
from nilearn.image import resample_to_img
from nilearn.image import math_img




def correct_mean_var(tseries):

    '''
    Correct time series to zero mean and unit variance. Adapted from:
    https://github.com/SIMEXP/niak/blob/master/commands/statistics/niak_correct_mean_var.m,
    but for a 1D array with the default 'mean_var' only.
    -------------------------------------
    TSERIES_N = CORRECT_MEAN_VAR(TSERIES)

    INPUT
    TSERIES: (1D array) time series.

    OUTPUT
    TSERIES_N: (1D array) same as data after mean/variance correction.
    '''

    nt = tseries.shape[0]
    tseries = tseries.reshape(nt,-1)

    mean_ts = np.mean(tseries,0)
    std_ts = tseries.std()

    tseries_n = (tseries - mean_ts)/std_ts
    tseries_n = np.ravel(tseries_n)

    return (tseries_n)


def build_size_roi(mask, labels_roi):

    '''
    Extract labels and size of ROIs given a mask. Adapted from:
    https://github.com/SIMEXP/niak/blob/master/commands/SI_processing/niak_build_size_roi.m
    -------------------------------------
    [SIZE_ROI,LABELS_ROI] = BUILD_SIZE_ROI(MASK)

    INPUT
    MASK: (array) voxels belonging to no region are coded with 0, those
    belonging to region I are coded with I (I being a positive integer).

    OUTPUTS
    SIZE_ROI: (vector) SIZE_ROI(I) is the number of voxels in region number I.

    LABELS_ROI: (vector) LABELS_ROI(I) is the label of region I.
    '''

    nb_roi = len(labels_roi)
    size_roi = np.zeros([nb_roi,1])

    for num_r in range(nb_roi):
        size_roi[num_r] = np.count_nonzero(mask==labels_roi[num_r])

    return size_roi

def build_combined_mask(imgs):
    mask_epi = compute_multi_epi_mask(imgs, lower_cutoff=0.2, upper_cutoff=0.85, connected=True, 
                       opening=2, threshold=0.5, target_affine=None, target_shape=None, 
                       exclude_zeros=False, n_jobs=1, memory=None, verbose=0)
    
    mask_mni = fetch_icbm152_brain_gm_mask()
    mask_mni = resample_to_img(source_img=mask_mni, target_img=imgs[0], interpolation='nearest')
    
    mask_combined = math_img('img1 & img2', img1=mask_epi, img2=mask_mni)
    
    return mask_combined

def _simulate_img():
    """ Simulate data with one "spot"
        Returns: img, data
    """
    data = np.zeros([8, 8, 8, 100])
    time_series = np.random.randn(1, 1, 2, data.shape[3])
    data[4, 4, 4, :] = time_series[0, 0, 0, :]
    data[4, 4, 5, :] = time_series[0, 0, 0, :] + time_series[0, 0, 1, :]
    corr = np.corrcoef(data[4, 4, 4, :], data[4, 4, 5, :])[1, 0]
    img = Nifti1Image(data, np.eye(4))
    
    mask_v = np.zeros(data.shape[0:3])
    mask_v[4, 4, 4] = 1
    mask_v[4, 4, 5] = 1
    mask = Nifti1Image(mask_v, np.eye(4))
    return img, mask, corr


def _extract_time_series_voxel(img, mask, confounds=None, smoothing_fwhm=None):
    masker = NiftiMasker(standardize=True, mask_img=mask, smoothing_fwhm=smoothing_fwhm)
    time_series_voxel = masker.fit_transform(img, confounds=confounds)
    return time_series_voxel, masker

def _extract_time_series_parcels(img, mask, parcels, confounds=None, smoothing_fwhm=None):
    # Standardize time series at the voxel level
    time_series_voxel, masker_voxel = _extract_time_series_voxel(img, mask)
    img_standardize = masker_voxel.inverse_transform(time_series_voxel)
    
    # Generate parcel value in voxel space
    parcels_resampled = resample_to_img(parcels, mask, interpolation='nearest')    
    parcels_voxel = NiftiMasker(standardize=False, mask_img=mask).fit_transform(parcels_resampled)
    
    # time series at the parcel level 
    masker_parcels = NiftiLabelsMasker(labels_img=parcels_resampled, standardize=False)
    time_series_parcels = masker_parcels.fit_transform(img_standardize)
    
    return time_series_parcels, masker_parcels, parcels_voxel

def _average_conn(img, mask, parcels, confounds=None, smoothing_fwhm=None):
    time_series_parcels, masker_parcels, parcels_voxel = _extract_time_series_parcels(img, mask, parcels, confounds, smoothing_fwhm)
    var_parcels = time_series_parcels.var(axis=0) 
    var_parcels = np.reshape(var_parcels, [var_parcels.shape[0], 1])
    size_parcels = build_size_roi(parcels_voxel, masker_parcels.labels_)
    mask_empty = (size_parcels == 0) | (size_parcels == 1)
    afc = ((size_parcels * size_parcels) * var_parcels - size_parcels) / (size_parcels * (size_parcels - 1))
    afc[mask_empty] = 0
    return afc, size_parcels