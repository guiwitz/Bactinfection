from cellpose import models
from bactinfection import utils
import pandas as pd
from oirpy.oirreader import Oirreader
import skimage.io
from pathlib import Path
import numpy as np


def single_image_analysis(
    filepath,
    save_folder,
    diameter,
    nucl_channel,
    cell_channel,
    bact_channel,
    actin_channel,
    bact_width,
    bact_len,
    corr_threshold,
    min_corr_vol,
    n_std,
):

    oir_image = Oirreader(filepath)
    channels = oir_image.get_meta()["channel_names"]
    stack = oir_image.get_stack()

    model = None

    # detect nuclei
    im_nucl = stack[:, :, channels.index(nucl_channel)]
    nucl_mask = segment_cellpose(model, im_nucl, diameter)
    save_to = save_folder.joinpath(Path(filepath).stem + "_nucl_seg.tif")
    skimage.io.imsave(save_to, nucl_mask, check_contrast=False)

    # detect cells
    im_cell = stack[:, :, channels.index(cell_channel)]

    cell_mask = utils.segment_cells(im_cell)
    save_to = save_folder.joinpath(Path(filepath).stem + "_cell_seg.tif")
    skimage.io.imsave(save_to, cell_mask.astype(np.uint8), check_contrast=False)

    # detect bacteria
    im_bact = stack[:, :, channels.index(bact_channel)]

    im_bact = skimage.filters.median(im_bact, skimage.morphology.disk(2))
    nucl_mask2 = nucl_mask > 0
    final_mask = cell_mask & ~nucl_mask2
    final_mask = final_mask.astype(bool)
    bact_mask, _, _ = utils.segment_bacteria(
        im_bact,
        final_mask,
        n_std=n_std,
        bact_len=bact_len,
        bact_width=bact_len,
        corr_threshold=corr_threshold,
        min_corr_vol=min_corr_vol,
    )
    save_to = save_folder.joinpath(Path(filepath).stem + "_bact_seg.tif")
    skimage.io.imsave(save_to, bact_mask.astype(np.uint16), check_contrast=False)

    # detect actin tails
    im_actin = oir_image.get_images(channels.index(actin_channel))
    im_actin = skimage.filters.median(im_actin, skimage.morphology.disk(2))
    actin_mask = utils.segment_actin(
        im_actin,
        np.ones(im_actin.shape, dtype=np.bool),
        bact_len,
        bact_width,
        n_std,
        min_corr_vol,
    )
    save_to = save_folder.joinpath(Path(filepath).stem + "_actin_seg.tif")
    skimage.io.imsave(save_to, actin_mask.astype(np.uint16), check_contrast=False)

    # extract signals
    measurements = extract_signals(stack, bact_mask, channels, Path(filepath))
    save_to = save_folder.joinpath(Path(filepath).stem + "_measure.csv")
    measurements.to_csv(save_to, index=False)


def segment_cellpose(model, image, diameter):
    """Segment image x using Cellpose. If model is None, a model is loaded"""

    if model is None:
        model = models.Cellpose(model_type="nuclei")
    m, flows, styles, diams = model.eval([image], diameter=diameter, channels=[[0, 0]])
    m = m[0]
    m = m.astype(np.uint8)
    return m


def extract_signals(stack, bact_labels, channels, filepath):

    if bact_labels.max() > 0:
        dataframes = []
        for x in range(len(channels)):
            if channels[x] is not None:
                measurements = skimage.measure.regionprops_table(
                    bact_labels,
                    stack[:, :, x],
                    properties=("mean_intensity", "label", "area", "eccentricity"),
                )
                dataframes.append(
                    pd.DataFrame(
                        {
                            **measurements,
                            **{"channel": channels[x]},
                            **{"filename": filepath.name},
                        }
                    )
                )

        measure_df = pd.concat(dataframes)
        return measure_df
