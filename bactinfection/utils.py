import numpy as np
import pandas as pd
import skimage.measure
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from skimage.feature import match_template


def rotation_templat_matching(image, rot_templ):

    # rotate image over a series of angles and do template matching
    # this has the advantage that the template is always the same and
    # corresponds to the true mode i.e. bright band with dark borders
    rot_im = []
    to_pad = int(0.5 * image.shape[0] * (2 ** 0.5 - 1))
    im_pad = np.pad(image, to_pad, mode='reflect')
    for alpha in np.arange(0, 180, 18):
        im_rot = skimage.transform.rotate(
            im_pad, alpha, preserve_range=True)
        im_match = match_template(im_rot, rot_templ, pad_input=True)
        im_unrot = skimage.transform.rotate(
            im_match, -alpha, preserve_range=True)
        rot_im.append(im_unrot[to_pad:-to_pad, to_pad:-to_pad])
    all_match = np.stack(rot_im, axis=0)

    return all_match

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