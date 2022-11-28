import nibabel as nib
from nibabel import Nifti1Image
import numpy as np

from nilearn.masking import compute_multi_epi_mask
from nilearn.image import resample_to_img, math_img
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker

def average_conn(imgs, mask_grey, parcel, confounds=None, **kwargs):
    """ Compute average connectivity within parcels.""" 
    mask_brain = _mask_brain(imgs, mask_grey, **kwargs)
    parcel_brain, parcel_voxel = _parcel_brain(parcel, mask_brain)
    # Need to properly loop over datasets
    img_n, time_series_voxel = _load_voxel_data(mask_brain, imgs[0], confounds[0])
    time_series, masker_parcels = _compute_tseries(parcel_brain, mask_brain, img_n)
    afc, size_parcels = _compute_afc(time_series, masker_parcels, parcel_voxel)
    return afc  


def _mask_brain(imgs, mask_grey, **kwargs):
    """Generate a brain mask combining fMRI data with a grey matter segmentation."""
    # Build a group mask for the fMRI data
    mask_epi = compute_multi_epi_mask(imgs, **kwargs)
    # lower_cutoff=0.2, upper_cutoff=0.85, connected=True,
    #                  opening=2, threshold=0.5, target_affine=None, target_shape=None,
    #                  exclude_zeros=False, n_jobs=1, memory=None, verbose=0

    # Resample the grey matter in the space of the functional data
    mask_grey_resampled = resample_to_img(
        source_img=mask_grey, target_img=imgs[0], interpolation="nearest"
    )

    # Combine the brain mask with the grey matter mask
    mask_brain = math_img("img1 & img2", img1=mask_epi, img2=mask_grey_resampled)

    return mask_brain


def _parcel_brain(parcel, mask_brain):
    """Combine brain parcels with a grey matter mask (at target resolution and sampling)."""
    parcel_resampled = resample_to_img(parcel, mask_brain, interpolation="nearest")
    parcel_brain = math_img("img1 * (img2 > 0)", img1=parcel_resampled, img2=mask_brain)
    masker_no_standardize = NiftiMasker(
        standardize=False, mask_img=mask_brain, memory="nilearn_cache"
    )
    parcel_voxel = masker_no_standardize.fit_transform(parcel_brain)

    return parcel_brain, parcel_voxel


def _load_voxel_data(mask_combined, img, confounds):
    # load voxel data for one subject
    masker = NiftiMasker(
        standardize=True,
        mask_img=mask_combined,
        smoothing_fwhm=5,
        memory="nilearn_cache",
    )
    time_series_voxel = masker.fit_transform(img, confounds=confounds)
    img_n = masker.inverse_transform(time_series_voxel)

    return img_n, time_series_voxel


def _compute_tseries(mist_resampled, mask_img, img_n):
    # compute time series using mist resampled
    masker_mist = NiftiLabelsMasker(
        labels_img=mist_resampled,
        mask_img=mask_img,
        standardize=False,
        memory="nilearn_cache",
    )
    time_series_mist = masker_mist.fit_transform(img_n)

    return time_series_mist, masker_mist


def _build_size_roi(mask, labels_roi):

    """
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
    """

    nb_roi = len(labels_roi)
    size_roi = np.zeros([nb_roi, 1])

    for num_r in range(nb_roi):
        size_roi[num_r] = np.count_nonzero(mask == labels_roi[num_r])

    return size_roi


def _compute_afc(time_series_mist, masker_mist, mist_voxel):
    var_mist = time_series_mist.var(axis=0)
    var_mist = np.reshape(var_mist, [var_mist.shape[0], 1])
    size_parcels = _build_size_roi(mist_voxel, masker_mist.labels_)
    mask_empty = (size_parcels == 0) | (size_parcels == 1)
    afc = ((size_parcels * size_parcels) * var_mist - size_parcels) / (
        size_parcels * (size_parcels - 1)
    )
    afc[mask_empty] = 0

    return afc, size_parcels


def _compute_brute_afc(time_series_voxel, mist_voxel, masker_mist):

    brute_afc = []
    for num_parcel in masker_mist.labels_:
        # Extract the voxel time series in the network
        time_series_network = time_series_voxel[:, mist_voxel[0, :] == num_parcel]
        time_series_network.shape

        conn_network = np.corrcoef(time_series_network.transpose())

        brute_afc_parcel = np.mean(
            conn_network[np.tril(np.ones(conn_network.shape), -1) == 1]
        )
        brute_afc.append(brute_afc_parcel)

    size_parcels = _build_size_roi(mist_voxel, masker_mist.labels_)
    mask_empty = (size_parcels == 0) | (size_parcels == 1)

    brute_afc = np.array(brute_afc)
    brute_afc = np.reshape(brute_afc, [brute_afc.shape[0], 1])
    brute_afc[mask_empty] = 0

    return brute_afc


# functions to test software on simulated data


def _simulate_img():
    """Simulate data with one "spot"
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


def _extract_time_series_parcels(
    img, mask, parcels, confounds=None, smoothing_fwhm=None
):
    # Standardize time series at the voxel level
    time_series_voxel, masker_voxel = _extract_time_series_voxel(
        img, mask, confounds, smoothing_fwhm
    )
    img_standardize = masker_voxel.inverse_transform(time_series_voxel)

    # Generate parcel value in voxel space
    parcels_resampled = resample_to_img(parcels, mask, interpolation="nearest")
    parcels_voxel = NiftiMasker(standardize=False, mask_img=mask).fit_transform(
        parcels_resampled
    )

    # time series at the parcel level
    masker_parcels = NiftiLabelsMasker(labels_img=parcels_resampled, standardize=False)
    time_series_parcels = masker_parcels.fit_transform(img_standardize)

    return time_series_parcels, masker_parcels, parcels_voxel


def _average_conn(img, mask, parcels, confounds=None, smoothing_fwhm=None):
    time_series_parcels, masker_parcels, parcels_voxel = _extract_time_series_parcels(
        img, mask, parcels, confounds, smoothing_fwhm
    )
    var_parcels = time_series_parcels.var(axis=0)
    var_parcels = np.reshape(var_parcels, [var_parcels.shape[0], 1])
    size_parcels = _build_size_roi(parcels_voxel, masker_parcels.labels_)
    mask_empty = (size_parcels == 0) | (size_parcels == 1)
    test_afc = ((size_parcels * size_parcels) * var_parcels - size_parcels) / (
        size_parcels * (size_parcels - 1)
    )
    test_afc[mask_empty] = 0
    return test_afc
