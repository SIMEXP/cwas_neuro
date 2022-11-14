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


    def build_size_roi(mask):

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
    labels_roi = np.unique(mask)
    labels_roi = labels_roi[labels_roi!=0]

    nb_roi = len(labels_roi)
    size_roi = np.zeros([nb_roi,1])

    for num_r in range(nb_roi):
        size_roi[num_r] = np.count_nonzero(mask==labels_roi[num_r])

    return (size_roi, labels_roi)


    def mat2lvec(mat):

    '''
    Convert a symmetric matrix into a vector (diagonal elements included). Adapted from:
    https://github.com/SIMEXP/niak/blob/master/commands/formats/niak_mat2lvec.m
    -------------------------------------
    LVEC = MAT2LVEC(MAT)

    INPUT
    MAT: (array) a square matrix. MAT should be symmetric. Diagonal elements
    will be included.

    OUTPUTS
    LVEC: (vector) a vectorized version of mat. Low-triangular and diagonal values are kept.
    '''

    N = mat.shape[1]
    mask_l = np.tril(np.ones(N, dtype=int))
    mask_l = mask_l>0
    lvec = mat[mask_l]

    return (lvec)
