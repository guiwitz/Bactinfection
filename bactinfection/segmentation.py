import numpy as np
import pandas as pd
import skimage.transform
import skimage.filters
import skimage.morphology
from skimage.segmentation import relabel_sequential

from cellpose import models
from .utils import (fit_gaussian_hist, volume_periodic_labelling,
select_labels, rotation_templat_matching)

def segment_nucl_cellpose(model, image, diameter, model_type="nuclei"):
    """
    Segment image x using Cellpose.
    
    Parmeters
    ----------
    model: cellpose model
    image: 2d array
        image to segment
    diameter: float
        estimated diamter of cells/nuclei
    model_type: str
        'cells' or 'nuclei'

    Returns
    -------
    m: 2d array
        labelled mask

    """

    if model is None:
        model = models.Cellpose(model_type=model_type)
    m, flows, styles, diams = model.eval([image], diameter=diameter, channels=[[0, 0]])
    m = m[0]
    m = m.astype(np.uint8)
    return m

def segment_cell_cellpose(model, image, diameter, model_type="cyto"):
    """
    Segment image x using Cellpose.
    
    Parmeters
    ----------
    model: cellpose model
    image: 2d array
        image to segment
    diameter: float
        estimated diamter of cells/nuclei
    model_type: str
        'cyto' or 'nuclei'

    Returns
    -------
    m: 2d array
        labelled mask

    """

    if model is None:
        model = models.Cellpose(model_type=model_type)
    m, flows, styles, diams = model.eval([image], diameter=diameter, channels=[[0, 0]])
    m = m[0]
    m = m.astype(np.uint8)
    return m

def segment_bacteria(
    image, final_mask, n_std, bact_len, bact_width,
    corr_threshold, min_corr_vol):
    """
    Segment bacteria based on a template
    
    Paramters
    ---------
    image: 2d array
        image to segment
    final_mask: 2d array
        mask to select zones to segment
    n_std: float
        number of standard deviation to set intensity
        threshold compared to background
    bact_len: int
        estimated length of bacteria in px
    bact_width: int
        estimated width of bacteria in px
    corr_threshold: float
        threshold on template matching quality in range [0,1]
    min_corr_vol: float
        minimal number of voxels with matching above threshold
        designed to suppress cases of single bright spots
    
    Returns
    -------
    remove_small: 2d array
        final bacteria labelled mask
    all_match: 3d array
        template matching rotational volume
    rotation_vol_label: 3d array
        labelled template matching rotational volume
    
    """

    image = skimage.filters.median(image, skimage.morphology.disk(2))

    # create template
    rot_templ = -np.ones((bact_len, bact_width))
    rot_templ[:, 1:-1] = 1

    # calculate an intensity threshold by fitting a gaussian on background
    out, _ = fit_gaussian_hist(image[final_mask], plotting=False)
    intensity_th = out[0][1] + n_std * np.abs(out[0][2])

    # rotate image over a series of angles and do template matching
    all_match = rotation_templat_matching(image, rot_templ)

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

    remove_small, fw, inv = relabel_sequential(remove_small)
    return remove_small, all_match, rotation_vol_label

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
