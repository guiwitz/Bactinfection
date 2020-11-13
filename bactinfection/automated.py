from cellpose import models
from bactinfection import utils
import pandas as pd
from oirpy.oirreader import Oirreader
import skimage.io
import skimage.morphology
import skimage.measure
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def single_image_analysis(
    filepath,
    save_folder,
    diameter_nucl,
    diameter_cell,
    nucl_channel,
    cell_channel,
    bact_channel,
    actin_channel,
    bact_width,
    bact_len,
    corr_threshold,
    min_corr_vol,
    n_std,
    nucl_model_type="nuclei",
    masking="cell_no_nuclei",
):

    report_file = filepath.joinpath(str(filepath).replace(".oir", "_error.txt"))

    try:
        oir_image = Oirreader(filepath)
        channels = oir_image.get_meta()["channel_names"]
        stack = oir_image.get_stack()
    except:
        with open(report_file, "a+") as f:
            f.write("Loading error")
        return None

    for c in [nucl_channel, bact_channel, actin_channel]:
        if c not in channels:
            with open(report_file, "a+") as f:
                f.write(nucl_channel + "channel not existing")
            return None

    model = None

    try:
        # detect nuclei
        im_nucl = stack[:, :, channels.index(nucl_channel)]
        nucl_mask = segment_nucl_cellpose(
            model, im_nucl, diameter_nucl, model_type=nucl_model_type
        )
        save_to = save_folder.joinpath(Path(filepath).stem + "_nucl_seg.tif")
        skimage.io.imsave(save_to, nucl_mask, check_contrast=False)
        if np.max(nucl_mask) == 0:
            with open(report_file, "a+") as f:
                f.write("No nuclei found")
            return None

        if cell_channel is not None:
            # detect cells
            # im_cell = stack[:, :, channels.index(cell_channel)]

            save_to = save_folder.joinpath(Path(filepath).stem + "_cell_seg.tif")
            # cell_mask = segment_cell_cellpose(model, im_cell, diameter_cell)
            # skimage.io.imsave(save_to, cell_mask.astype(np.uint8), check_contrast=False)
            cell_mask = skimage.io.imread(save_to)

            if np.max(cell_mask) == 0:
                with open(report_file, "a+") as f:
                    f.write("No cell found")
                return None

        # detect bacteria
        im_bact = stack[:, :, channels.index(bact_channel)]

        im_bact = skimage.filters.median(im_bact, skimage.morphology.disk(2))
        nucl_mask2 = nucl_mask > 0
        # exclude detection in the nuclei (mainly for when detecting
        # bacteria in the same channel as nuclei
        if masking == "cell_no_nuclei":
            final_mask = cell_mask & ~nucl_mask2
        elif masking == "cell":
            final_mask = cell_mask
        elif masking == "nuclei":
            final_mask = nucl_mask2
        else:
            with open(report_file, "a+") as f:
                f.write("No appropriate masking found")
            return None

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
        bact_mask = skimage.morphology.label(bact_mask)
        save_to = save_folder.joinpath(Path(filepath).stem + "_bact_seg.tif")
        skimage.io.imsave(save_to, bact_mask.astype(np.uint16), check_contrast=False)

        # detect actin tails
        if actin_channel is not None:
            im_actin = oir_image.get_images(channels.index(actin_channel))
            # im_actin = skimage.filters.median(im_actin, skimage.morphology.disk(2))
            actin_mask = utils.segment_actin(
                im_actin, final_mask, bact_len, bact_width, n_std, min_corr_vol,
            )
            save_to = save_folder.joinpath(Path(filepath).stem + "_actin_seg.tif")
            skimage.io.imsave(
                save_to, actin_mask.astype(np.uint16), check_contrast=False
            )

        # extract signals in nuclei and bacteria
        measurements = extract_signals(stack, bact_mask, channels, Path(filepath))
        measurements_nucl = extract_signals(stack, nucl_mask, channels, Path(filepath))

        # find to which nucleus each bacteria belongs (relevant only if nuclei not excluded)
        nucl_index = pd.DataFrame(
            skimage.measure.regionprops_table(
                bact_mask,
                intensity_image=nucl_mask,
                properties=("label", "max_intensity"),
            )
        )
        nucl_index = nucl_index.rename(columns={"max_intensity": "nucleus_index"})
        # merge signal and bacteria-nuclei info to be able to exclude data from
        # some nuclei later
        measurements = pd.merge(measurements, nucl_index, on="label")

        # export data
        save_to = save_folder.joinpath(Path(filepath).stem + "_measure.csv")
        measurements.to_csv(save_to, index=False)
        save_to = save_folder.joinpath(Path(filepath).stem + "_nucl_measure.csv")
        measurements_nucl.to_csv(save_to, index=False)

    except:
        with open(report_file, "a+") as f:
            f.write("Unknown error happened during segmentation")
            return None


def segment_nucl_cellpose(model, image, diameter, model_type="nuclei"):
    """Segment image x using Cellpose. If model is None, a model is loaded"""

    if model is None:
        model = models.Cellpose(model_type=model_type)
    m, flows, styles, diams = model.eval([image], diameter=diameter, channels=[[0, 0]])
    m = m[0]
    m = m.astype(np.uint8)
    return m


def segment_cell_cellpose(model, image, diameter):
    """Segment image x using Cellpose. If model is None, a model is loaded"""

    if model is None:
        model = models.Cellpose(model_type="cyto")
    m, flows, styles, diams = model.eval([image], diameter=diameter, channels=[[0, 0]])
    m = m[0]
    m = m > 0
    # m = m.astype(np.uint8)
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
    else:
        measure_df = pd.DataFrame(
            index=[],
            columns=[
                "mean_intensity",
                "label",
                "area",
                "eccentricity",
                "channel",
                "filename",
            ],
        )

    return measure_df


def get_seg_image(csvfile, seg_type):
    """Given a measurement file, load the corresponding images."""

    image_name = Path(str(csvfile).replace("_measure.csv", "_" + seg_type + ".tif"))

    image = skimage.io.imread(image_name)
    return image


def count_objects(csvfile, seg_type):
    """Given a measurement file, calculate how many object of 
    seg_type are present in the corresponding image"""

    image = get_seg_image(csvfile, seg_type)
    numobj = len(np.unique(image)) - 1

    return numobj


def plot_number_objects(averaged):
    fig, ax1 = plt.subplots()
    # color = 'tab:red'
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Number of bacteria")  # , color=color)
    ax1.plot(
        averaged.hour,
        averaged.bact_normalized,
        "-o",
        color="black",
        label="Number of bacteria",
    )
    ax1.tick_params(axis="y")  # , labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:blue'
    ax2.set_ylabel(
        "Number of actin tails"
    )  # , color=color)  # we already handled the x-label with ax1
    ax2.plot(
        averaged.hour,
        averaged.actin_normalized,
        "-s",
        color="black",
        label="Number of actin tails",
    )
    ax2.tick_params(axis="y")  # , labelcolor=color)

    fig.legend(loc=(0.15, 0.8))
    plt.xticks(ticks=np.unique(averaged.hour))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return fig
