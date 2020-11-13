"""
Core functions for data import and analysis
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019-2020
# License: BSD3

import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt

from scipy.optimize import leastsq
from skimage.morphology import label, binary_opening, disk
from skimage.filters import threshold_li, gaussian
from skimage.feature import match_template
import skimage
import skimage.transform
import oiffile

#import javabridge
#import bioformats as bf

from oirpy.oirreader import Oirreader

# fix issue with path for javabridge
#os.environ['JAVA_HOME']+='/'

#javabridge.start_vm(class_path=bf.JARS)


def oif_import(filepath, channels=True):
    """Import oif file format

    Parameters
    ----------
    filepath : str
        path to file
    channels : bool
        recover channel names

    Returns
    -------
    image : ND numpy array
        image

    """

    oibfile = oiffile.OifFile(filepath)
    image = oibfile.asarray()

    all_channels = []
    for x in oibfile.mainfile.keys():
        channel = re.findall("^Channel.*", x)
        if len(channel) > 0:
            all_channels.append(x)

    if channels:
        print([oibfile.mainfile[x]["DyeName"] for x in all_channels])

    return image


def oir_import(filepath):
    """Import oir file format via bioformats

    Parameters
    ----------
    filepath : str
        path to file

    Returns
    -------
    image : ND numpy array
        image

    """

    #with bf.ImageReader(filepath) as reader:
    #    image = reader.read(series=0, rescale=False)

    oirfile = Oirreader(filepath)
    image = oirfile.get_stack()
    return image


def oif_get_channels(filepath):
    """Show channesl present in oif file"""

    oibfile = oiffile.OifFile(filepath)

    all_channels = []
    for x in oibfile.mainfile.keys():
        channel = re.findall("^Channel.*", x)
        if len(channel) > 0:
            all_channels.append(x)

    print([oibfile.mainfile[x]["DyeName"] for x in all_channels])


def segment_nuclei(image, radius=15):
    """Basic threshold based nuclei segmetnation"""

    sigma = radius / np.sqrt(2)
    im_gauss = gaussian(image.astype(float), sigma=sigma)

    # create a global mask where nuclei migth be fused
    th_nucl = threshold_li(im_gauss)
    mask_nucl = im_gauss > 1 * th_nucl
    mask_nucl = binary_opening(mask_nucl, disk(10))

    mask_label = label(mask_nucl)

    """#find local maxima in LoG filterd image
    logim = ndi.filters.gaussian_laplace(im_gauss.astype(float),sigma=sigma)
    peaks = peak_local_max(-logim,min_distance=10,indices=False)
    peaks = peaks*mask_nucl

    #use the blobal and the refined maks for watershed
    im_water = watershed(
        -image, markers=label(peaks), mask=mask_nucl)#, compactness=1)"""

    return mask_label


def segment_nuclei_cellpose(image, model, diameter):
    """Segment nuclei using cellpose

    Parameters
    ----------
    image : 2D numpy array
        image with nuclei
    model : cellpose model
    diameter : estimated nuclei diameter

    Returns
    -------
    mask : 2D numpy array
        labelled mask of nuclei

    """

    masks, _, _, _ = model.eval(
        [image[::1, ::1]], channels=[[0, ]], diameter=diameter
        )
    """mask = skimage.transform.resize(
        masks[0], image.shape, preserve_range=True, order=0)"""

    mask = masks[0]

    return mask


def segment_cells(image):

    entropy = skimage.filters.rank.entropy(
            image, selem=np.ones((10, 10)))
    cell_mask = entropy > 5.5
    cell_mask = skimage.morphology.binary_opening(
        cell_mask, selem=skimage.morphology.disk(10))
    return cell_mask


def segment_bacteria(image, final_mask, n_std, bact_len, bact_width, corr_threshold, min_corr_vol):

    # create template
    rot_templ = -np.ones((bact_len, bact_width))
    rot_templ[:, 1:-1] = 1

    # calculate an intensity threshold by fitting a gaussian on background
    out, _ = fit_gaussian_hist(image[final_mask], plotting=False)
    intensity_th = out[0][1] + n_std * np.abs(out[0][2])

    # rotate image over a series of angles and do template matching
    # this has the advantage that the template is always the same and
    # corresponds to the true mode i.e. bright band with dark borders
    rot_im = []
    to_pad = int(0.5 * image.shape[0] * (2 ** 0.5 - 1))
    im_pad = np.pad(image, to_pad, mode='constant')
    for alpha in np.arange(0, 180, 18):
        im_rot = skimage.transform.rotate(
            im_pad, alpha, preserve_range=True)
        im_match = match_template(im_rot, rot_templ, pad_input=True)
        im_unrot = skimage.transform.rotate(
            im_match, -alpha, preserve_range=True)
        rot_im.append(im_unrot[to_pad:-to_pad, to_pad:-to_pad])
    all_match = np.stack(rot_im, axis=0)

    # keep only regions matching well in the rotational match volume
    rotation_vol = all_match > corr_threshold

    # create negative mask to remove regions clearly between bacteria
    neg_mask = np.max(-all_match, axis=0)
    neg_mask = neg_mask < 0.3
    rotation_vol = rotation_vol * neg_mask

    # create volume labelled with periodic boundary conditions in z
    rotation_vol_label = volume_periodic_labelling(rotation_vol)

    # measure region properties
    rotation_vol_props = pd.DataFrame(
        skimage.measure.regionprops_table(
            rotation_vol_label,
            image * np.ones(rotation_vol_label.shape),
            properties=("label", "area", "mean_intensity"),
        )
    )

    # keep only regions with a minimum number of matching voxels
    new_label_image = select_labels(
        rotation_vol_label,
        rotation_vol_props,
        {"area": min_corr_vol, "mean_intensity": intensity_th},
    )

    # relabel and project. In the projection we assume there are no
    # overlapping regions
    new_label_image_proj = np.max(new_label_image, axis=0)
    new_label_image_proj = new_label_image_proj * final_mask
    if new_label_image_proj.max() > 0:
        remove_small = select_labels(
            new_label_image_proj,
            limit_dict={"area": 2})
    else:
        remove_small = new_label_image_proj

    return remove_small, all_match, rotation_vol_label


def segment_actin(image, cell_mask, bact_len, bact_width, n_std, min_corr_vol):

    # two templates are used: small and large widths
    rot_templ = -np.ones((bact_len, bact_width))
    rot_templ[:, 1:-1] = 1

    rot_templ2 = -np.ones((bact_len, bact_width+3))
    rot_templ2[:, 1:-1] = 1

    # calculate an intensity threshold by fitting a gaussian on background
    out, _ = fit_gaussian_hist(image[cell_mask], plotting=False)
    intensity_th = out[0][1] + n_std * np.abs(out[0][2])

    # rotate image over a series of angles and do template matching
    # this has the advantage that the template is always the same and
    # corresponds to the true mode i.e. bright band with dark borders
    rot_im = []
    to_pad = int(0.5 * image.shape[0] * (2 ** 0.5 - 1))
    im_pad = np.pad(image, to_pad, mode='constant')
    for alpha in np.arange(0, 180, 18):
        im_rot = skimage.transform.rotate(
            im_pad, alpha, preserve_range=True)
        im_match = np.max(np.stack(
            [
                match_template(im_rot, rot_templ, pad_input=True, mode='wrap'),
                match_template(im_rot, rot_templ2, pad_input=True, mode='wrap'),
            ], axis=0), axis=0)
        im_unrot = skimage.transform.rotate(
            im_match, -alpha, preserve_range=True)
        rot_im.append(im_unrot[to_pad:-to_pad, to_pad:-to_pad])
    all_match = np.stack(rot_im, axis=0)
    all_match[:, 0:10, :] = 0
    all_match[:, -10::, :] = 0
    all_match[:, :, 0:10] = 0
    all_match[:, :, -10::] = 0

    # keep only regions matching well in the rotational match volume
    rotation_vol = all_match > 0.4
    rotation_vol_proj = all_match.max(axis=0)

    # create volume labelled with periodic boundary conditions in z
    rotation_vol_label = volume_periodic_labelling(rotation_vol)

    # measure region properties of image
    rotation_vol_props = pd.DataFrame(
        skimage.measure.regionprops_table(
            rotation_vol_label,
            image * np.ones(rotation_vol_label.shape),
            properties=("label", "area", "mean_intensity"),
        )
    )

    # keep only regions with a minimum number of matching voxels
    new_label_image = select_labels(
        rotation_vol_label,
        rotation_vol_props,
        #{"area": 0, "mean_intensity": 0},
        {"area": min_corr_vol, "mean_intensity": intensity_th},
    )

    new_label_image_proj = np.max(new_label_image, axis=0)
    if new_label_image_proj.max() > 0:
        proj_props = pd.DataFrame(
            skimage.measure.regionprops_table(
                new_label_image_proj,
                rotation_vol_proj,
                properties=("label", "area", "mean_intensity"),
            )
        )
        remove_small = select_labels(
            new_label_image_proj,
            proj_props,
            {"mean_intensity": 0.5, "area": 25},
        )
    else:
        remove_small = new_label_image_proj

    remove_small = skimage.morphology.label(remove_small)

    return remove_small


def create_template(length=7, width=3):
    """Create series of rotated bacteria templates

    Parameters
    ----------
    length : int
        bacteria length
    width : int
        bacteria width

    Returns
    -------
    rot_templ : list of 2D numpy arrays
        series of rotated templates

    """

    template = np.zeros((length + 2, length + 2))
    template[
        1: 1 + length,
        int((length + 1) / 2)
        - int((width - 1) / 2): int((length + 1) / 2)
        + int((width - 1) / 2)
        + 1,
    ] = 1

    # create a list of rotated templates
    rot_templ = []
    for ind, alpha in enumerate(np.arange(0, 180, 18)):
        rot_templ.append(skimage.transform.rotate(template, alpha, order=0))

    return rot_templ


def detect_bacteria(image, rot_template, mask=None):
    """Detect bacteria by template matching with series
    of rotated templates and max projection.

    Parameters
    ----------
    image : 2D numpy array
        bacteria image
    rot_template : list of 2D numpy arrays
        list of rotated templates
    mask : 2D numpy array
        masks of regions to conserve

    Returns
    -------
    maxproj : 2D numpy array
        max proj. over template matching for
        all rotation angles

    """
    # do template matching with all rotated templates
    all_match = rotational_matching(image, rot_template)

    # calculate max proj of those images matched with templates
    # at different angles maxarg contains for each pixel the plane index
    # of best match and hence the angle
    maxproj = np.max(all_match, axis=0)

    if mask is not None:
        maxproj = maxproj * (1 - mask)

    return maxproj


def rotational_matching(image, rot_template):
    """Perform template matching with seris of rotated
    templates.

    Parameters
    ----------
    image : 2D numpy array
        bacteria image
    rot_template : list of 2D numpy arrays
        list of rotated templates

    Returns
    -------
    all_match : list of 2D numpy arrays
        list of matched images

    """

    all_match = np.zeros(
        (len(rot_template) + 1, image.shape[0], image.shape[1])
        )
    for ind in range(len(rot_template)):
        all_match[ind + 1, :, :] = match_template(
            image, rot_template[ind], pad_input=True
        )
    return all_match


def remove_small(image, minsize):
    """Use region properties to keep only regions larger
    than a threshold size in a mask and return cleaned mask.

    Parameters
    ----------
    image : 2D numpy array
        mask
    minsize : int
        minimak region size to keep

    Returns
    -------
    mask : 2D numpy array
        cleaned mask

    """

    labeled = skimage.morphology.label(image > 1)
    regions = skimage.measure.regionprops(labeled)
    indices = np.array(
        [0] +
        [x.label if (x.area > minsize) else 0 for x in regions])
    mask = indices[labeled] > 0
    mask = mask + 1

    return mask


def select_labels(im_label, im_properties=None, limit_dict={"label": 0}):
    """Given a labelled image and pairs of properties/thresholds, keep only
    labels of regions with properties above thresholds and return clean mask.

    Parameters
    ----------
    im_label : 2D numpy array
        labelled mask
    im_properties : dataframe
        dataframe of output of
        skimage.measure.regionprops_table
    limit_dict: dictionary
        dictionary of thresholds for multiple properties
        e.g. {'label': 0, 'area': 20}

    Returns
    -------
    clean_labels : 2D numpy array
        cleaned labels

    """

    if im_properties is None:
        im_properties = pd.DataFrame(
            skimage.measure.regionprops_table(
                im_label, properties=["label"] + list(limit_dict.keys())
            )
        )
    # create boolean pandas mask with constrains
    select_lab = im_properties.label > 0
    for k in limit_dict:
        select_lab = select_lab & (im_properties[k] > limit_dict[k])
    sel_labels = im_properties[select_lab].label.values

    # create index array to subselect labels
    indices = np.array(
        [
            i if i in sel_labels else 0
            for i in np.arange(im_properties.label.max() + 1)]
    )

    clean_labels = indices[im_label]
    return clean_labels


def volume_periodic_labelling(rotation_vol):
    """Given a binary volume create a labelled volume with
    periodic boundary conditions along the first dimension.

    Parameters
    ----------
    rotation_vol : 3D numpy array
        mask

    Returns
    -------
    total_vol : 3D numpy array
        relabelled array where periodically connected regions
        along z (first dim) share labels

    """

    original_size = rotation_vol.shape

    # repeat the volume to connect 0deg and 180deg matching
    rotation_vol = np.concatenate((rotation_vol, rotation_vol), axis=0)

    # label volume
    rotation_vol_label = skimage.measure.label(rotation_vol)

    # extract labels at the interface of the two assembled volumes
    new_labels = np.unique(rotation_vol_label[original_size[0], :, :])

    # extract the labels at the same position as the new_labels
    # but in the first layer
    old_labels_to_remove = rotation_vol_label[0, :, :][
        rotation_vol_label[original_size[0], :, :] > 0
    ]

    # construct index list of labels to keep for replacement
    # in the assembled volume
    indices = np.array(
        [
            i if i in new_labels else 0
            for i in np.arange(rotation_vol_label.max() + 1)]
    )

    # construct index list of labels to keep in the original volume
    indices2 = np.array(
        [
            i if i not in old_labels_to_remove else 0
            for i in np.arange(rotation_vol_label.max() + 1)
        ]
    )

    # recover labelled regions in the two labelled volumes and assemble them
    keep_overlapping = indices[rotation_vol_label]
    keep_old = indices2[rotation_vol_label]
    total_vol = (
        keep_old[0: original_size[0], :, :]
        + keep_overlapping[original_size[0]::, :, :]
    )

    return total_vol


def normal_fit(self, x, a, x0, s):
    """Gaussian function"""
    return (a / (s * (2 * np.pi) ** 0.5)) * np.exp(-0.5 * ((x - x0) / s) ** 2)


def fit_gaussian_hist(data, plotting=True, minbin=0, maxbin=4000, binwidth=30):
    """Fit a gaussian to the histogram of a data set.

    Parameters
    ----------
    data : numpy array
        data to fit
    plotting : bool
        show fit output
    minbin : int
        minimum value of bin
    maxbin : int
        maximum value of bin
    binwidth : float
        with of bins


    Returns
    -------
    out : list
        output of fitting procedure
        out[0][0] is amplitud, out[0][1] is mean,
        out[0][2] is covariances
    fig : matplotlib figure object

    """

    def fitfunc(p, x):
        return p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2)

    def errfunc(p, x, y):
        return (y - fitfunc(p, x))

    ydata, xdata = np.histogram(data, bins=np.arange(minbin, maxbin, binwidth))
    xdata = [0.5 * (xdata[x] + xdata[x + 1]) for x in range(len(xdata) - 1)]
    init = [np.max(ydata), xdata[np.argmax(ydata)], np.std(data)]

    out = leastsq(errfunc, init, args=(xdata, ydata))

    fig = []
    if plotting is True:
        fig, ax = plt.subplots()
        plt.bar(x=xdata, height=ydata, width=binwidth, color="r")
        plt.plot(xdata, fitfunc(out[0], xdata))
        plt.plot(
            [
                out[0][1] + 3 * np.abs(out[0][2]),
                out[0][1] + 3 * np.abs(out[0][2])],
            [0, np.max(ydata)],
            "green",
        )
        ax.set_ylabel("Counts")
        ax.set_xlabel("Pixel intensity")
        ax.legend(["Background fit", "Threshold", "Pixel intensity"])
        plt.show()

    return out, fig


def filter_sets(image):
    """Create series of filtered images for ML

    Parameters
    ----------
    image : 2D numpy array
        image to filter

    Returns
    -------
    filter_list : list of 2D numpy arrays
        list of filtered images
    filter_names : list of str
        list of filter names

    """

    im_gauss = skimage.filters.gaussian(image, sigma=10, preserve_range=True)
    im_gauss2 = skimage.filters.gaussian(image, sigma=20, preserve_range=True)
    im_frangi = skimage.filters.frangi(image)
    im_prewitt = skimage.filters.prewitt(image)
    im_meijering = skimage.filters.meijering(image)
    im_gauss_laplace = skimage.filters.laplace(
        skimage.filters.gaussian(image, sigma=5, preserve_range=True), ksize=10
    )
    im_gradient = skimage.filters.rank.gradient(
        image, skimage.morphology.disk(8))
    im_entropy = skimage.filters.rank.entropy(
        image, skimage.morphology.disk(8))
    im_roberts = skimage.filters.roberts(
        skimage.filters.gaussian(image, sigma=5, preserve_range=True)
    )

    filter_list = [
        im_gauss,
        im_gauss2,
        im_frangi,
        im_prewitt,
        im_meijering,
        im_gauss_laplace,
        im_gradient,
        im_entropy,
        im_roberts,
    ]
    filter_names = [
        "Gauss $\sigma$=10",
        "Gauss $\sigma$=20",
        "Frangi",
        "Prewitt",
        "Meijering",
        "Gauss+Laplace",
        "Gradient",
        "Entropy",
        "Roberts",
    ]

    return filter_list, filter_names
