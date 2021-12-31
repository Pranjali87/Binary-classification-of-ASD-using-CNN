def performPreprocess(folder, outfile) :
    from nilearn import image as nli
    from nilearn import plotting
    import numpy as np
    import nibabel as nib
    import os
    from nibabel.testing import data_path
    import pylab as plt
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
    
    example1_filename = folder + "/rest_1/rest.nii.gz"
    bold=nli.load_img(example1_filename)
    
    bold = bold.slicer[..., 5:]
    img = nli.mean_img(bold)#mean of the image
    plotting.view_img(img, bg_img=img)
    
    mean = nli.mean_img(bold) # resample image to computed mean image
    print([mean.shape, img.shape])
    
    resampled_img = nli.resample_to_img(img, mean)
    resampled_img.shape
    
    from nilearn import plotting
    plotting.plot_anat(img, title='original img', display_mode='z', dim=-1,
                       cut_coords=[-20, -10, 0, 10, 20, 30])
    plotting.plot_anat(resampled_img, title='resampled img', display_mode='z', dim=-1,
                       cut_coords=[-20, -10, 0, 10, 20, 30])
    
    for fwhm in range(1, 12, 5):
        smoothed_img = nli.smooth_img(mean, fwhm)
        plotting.plot_epi(smoothed_img, title="Smoothing %imm" % fwhm,
                         display_mode='z', cmap='magma')
    
    TR = bold.header['pixdim'][4]#get TR value of functional image
    
    func_d = nli.clean_img(bold, detrend=True, standardize=False, t_r=TR)
    
    # Plot the original and detrended timecourse of a random voxel
    x, y, z = [31, 14, 7]
    plt.figure(figsize=(12, 4))
    plt.plot(np.transpose(bold.get_fdata()[x, y, z, :]))
    plt.plot(np.transpose(func_d.get_fdata()[x, y, z, :]))
    plt.legend(['Original', 'Detrend']);
    
    func_ds = nli.clean_img(bold, detrend=True, standardize=True, t_r=TR)
    
    plt.figure(figsize=(12, 4))
    plt.plot(np.transpose(func_d.get_fdata()[x, y, z, :]))
    plt.plot(np.transpose(func_ds.get_fdata()[x, y, z, :]))
    plt.legend(['Detrend', 'Detrend+standardize']);
    
    plt.figure(figsize=(12, 5))
    plt.plot(np.transpose(func_d.get_fdata()[x, y, z, :]))
    plt.plot(np.transpose(func_ds.get_fdata()[x, y, z, :]))
    plt.legend(['Det.+stand.', 'Det.+stand.-confounds']);
    
    mean = nli.mean_img(bold)
    
    thr = nli.threshold_img(mean, threshold='95%')
    plotting.view_img(thr, bg_img=img)
    voxel_size = np.prod(thr.header['pixdim'][1:4])  # Size of 1 voxel in mm^3
    
    from nilearn.regions import connected_regions
    cluster = connected_regions(thr, min_region_size=1000. / voxel_size, smoothing_fwhm=1)[0]
    mask = nli.math_img('np.mean(img,axis=3) > 0', img=cluster)
    from nilearn.plotting import plot_roi
    plotting.plot_roi(mask, bg_img=img, display_mode='z', dim=-.5, cmap='magma_r');
    # Apply mask to original functional image
    from nilearn.masking import apply_mask
    
    all_timecourses = apply_mask(bold, mask)
    all_timecourses.shape
    
    from nilearn.masking import unmask
    img_timecourse = unmask(all_timecourses, mask)
    
    mean_timecourse = all_timecourses.mean(axis=1)
    plt.plot(mean_timecourse)
    
    # Import CanICA module
    from nilearn.decomposition import CanICA
    
    # Specify relevant parameters
    n_components = 5
    fwhm = 6.
    
    # Specify CanICA object
    canica = CanICA(n_components=n_components, smoothing_fwhm=fwhm,
                    memory="nilearn_cache", memory_level=2,
                    threshold=3., verbose=10, random_state=0, n_jobs=-1,
                    standardize=True)
    # Run/fit CanICA on input data
    canica.fit(bold)
    # Retrieve the independent components in brain space
    components_img = canica.masker_.inverse_transform(canica.components_)
    
    from nilearn.image import iter_img
    from nilearn.plotting import plot_stat_map
    
    for i, cur_img in enumerate(iter_img(components_img)):
        plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                      cut_coords=[0, 10, 20, 30], colorbar=True, bg_img=img)
    
    from scipy.stats.stats import pearsonr
    
    # Get data of the functional image
    orig_data = bold.get_fdata()
    
    # Compute the pearson correlation between the components and the signal
    curves = np.array([[pearsonr(components_img.get_fdata()[..., j].ravel(),
          orig_data[..., i].ravel())[0] for i in range(orig_data.shape[-1])]
            for j in range(canica.n_components)])
    
    # Plot the components
    fig = plt.figure(figsize=(14, 4))
    centered_curves = curves - curves.mean(axis=1)[..., None]
    plt.plot(centered_curves.T)
    plt.legend(['IC %d' % i for i in range(canica.n_components)])
    
    # Import DictLearning module
    from nilearn.decomposition import DictLearning
    
    # Specify the dictionary learning object
    dict_learning = DictLearning(n_components=n_components, n_epochs=1,
                                 alpha=1., smoothing_fwhm=fwhm, standardize=True,
                                 memory="nilearn_cache", memory_level=2,
                                 verbose=1, random_state=0, n_jobs=-1)
    
    # As before, let's fit the model to the data
    dict_learning.fit(bold)
    
    # Retrieve the independent components in brain space
    components_img = dict_learning.masker_.inverse_transform(dict_learning.components_)
    
    # Now let's plot the components
    for i, cur_img in enumerate(iter_img(components_img)):
        plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                      cut_coords=[0, 10, 20, 30], colorbar=True, bg_img=img)
    
    nib.save(components_img, outfile)

def performPreprocessAnat(folder, outfile) :
    print('Anat preprocessing started...')
    
    import nilearn as ni
    import numpy as np
    import nibabel as nib
    import os
    from nibabel.testing import data_path
    
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1
    
    example_filename = folder + "/anat_1/mprage.nii.gz"
    
    img=nib.load(example_filename)
    t1_hdr = img.header
    t1_data = img.get_fdata()
    
    import numpy as np
    print(np.min(t1_data))
    print(np.max(t1_data))
    
    x_slice = t1_data[9, :, :]
    y_slice = t1_data[:, 19, :]
    z_slice = t1_data[:, :, 2]
    import matplotlib.pyplot as plt
    
    slices = [x_slice, y_slice, z_slice]
    
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    
    img.orthoview()
    affine=img.affine
    x, y, z, _ = np.linalg.pinv(affine).dot(np.array([0, 0, 0, 1])).astype(int)
    
    print("Affine:")
    print(affine)
    print("Center: ({:d}, {:d}, {:d})".format(x, y, z))
    nib.aff2axcodes(affine)
    nib.affines.voxel_sizes(affine)
    nib.aff2axcodes(affine)
    nib.affines.voxel_sizes(affine)
    img.orthoview()
    #to make values between 0-255 change datatype to unsigned 8 bit
    data=img.get_fdata()
    rescaled = ((data - data.min()) * 255. / (data.max() - data.min())).astype(np.uint8)
    new_img = nib.Nifti1Image(rescaled, affine=img.affine, header=img.header)
    orig_filename = img.get_filename()
    img.set_data_dtype(np.uint8)
    # Save image again
    new_img = nib.Nifti1Image(rescaled, affine=img.affine, header=img.header)
    nib.save(new_img, outfile)