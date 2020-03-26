"""
Class implementing segmentation tools.
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3

import os, glob, re, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import skimage.measure
import skimage.morphology
from skimage.feature import match_template
import napari
import pickle

# import ipywidgets as ipw

from . import utils

# import warnings
# warnings.filterwarnings('ignore')


class Bact:
    def __init__(
        self,
        channels=None,
        all_files=None,
        folder_name=None,
        corr_threshold=0.5,
        min_corr_vol=3,
        use_ml=False,
        use_cellpose=False,
        model=None,
        saveto='Segmented'
    ):

        """Standard __init__ method.
        
        Parameters
        ----------
        channels : list of str
            list of available channels
        all_files : list of str
            list of files to analyze
        folder_name : str
            path of folder to analyze
        corr_threshold : float
            threshold cross-correlation for template matching
        min_corr_vol : int
            minimal number of voxels in rotational volume matching regions
        use_ml : bool
            use manual annotation and ML for nucleus segmentation
        use_cellpose : bool
            use cellpose for nucleus segmentation
        model : cellpose model
            cellpose model
        saveto : str
            folder name wehre to save the segmentation
        
        
        Attributes
        ----------
            
        nucl_channel = str
            name of nucleus channel
        bact_channel = str
            name of bacteria channel
        cell_channel = str
            name of cell channel
        
        """

        self.channels = channels
        self.all_files = all_files
        self.folder_name = folder_name
        self.corr_threshold = corr_threshold
        self.min_corr_vol = min_corr_vol
        self.use_ml = use_ml
        self.use_cellpose = use_cellpose
        self.model = model
        self.saveto = saveto

        self.current_image = None
        self.current_image_med = None
        self.hard_threshold = None
        self.maxproj = None
        self.bact_mask = None
        self.ml = None
        self.result = None
        self.nucl_channel = None
        self.bact_channel = None
        self.cell_channel = None
        self.minsize = 0
        self.fillholes = False
        self.cellpose_diam = 60

    def initialize_output(self):

        self.nuclei_segmentation = {os.path.split(x)[1]: None for x in self.all_files}
        self.bacteria_segmentation = {os.path.split(x)[1]: None for x in self.all_files}
        self.cell_segmentation = {os.path.split(x)[1]: None for x in self.all_files}
        self.annotations = {os.path.split(x)[1]: None for x in self.all_files}
        self.bacteria_channel_intensities = {
            os.path.split(x)[1]: None for x in self.all_files
        }
        self.bact_measurements = {os.path.split(x)[1]: None for x in self.all_files}

    def import_file(self, filepath):

        if os.path.split(filepath)[0] == "":
            filepath = self.folder_name + "/" + filepath

        self.current_file = os.path.split(filepath)[1]
        filetype = os.path.splitext(self.current_file)[1]

        if filetype == ".oir":
            image = utils.oir_import(filepath)
        else:
            image = utils.oif_import(filepath, channels=False)

        self.current_image = image

    def import_cellpose_model(self):

        import mxnet
        from cellpose import models

        model = models.Cellpose(device=mxnet.cpu(), model_type="nuclei")
        self.model = model

    def calculate_median(self, channel):

        ch = self.channels.index(channel)
        self.current_image_med = skimage.filters.median(
            self.current_image[:, :, ch], skimage.morphology.disk(2)
        )
        self.current_median_channel = channel

    def segment_cells(self, channel):

        ch = self.channels.index(channel)

        back_pix = np.sort(np.ravel(self.current_image[:, :, ch]))[0:10000]
        back_mean = np.mean(back_pix)
        back_std = np.std(back_pix)

        cell_mask = self.current_image[:, :, ch] > back_mean + 10 * back_std
        cell_mask = skimage.morphology.binary_opening(
            cell_mask, selem=skimage.morphology.disk(5)
        )

        # cell_mask = self.current_image[:, :, ch] > skimage.filters.threshold_otsu(self.current_image[:, :, ch])

        self.current_cell_mask = cell_mask
        self.cell_segmentation[self.current_file] = cell_mask

    def segment_nuclei(self, channel):

        ch = self.channels.index(channel)

        nucle_seg = utils.segment_nuclei(self.current_image[:, :, ch], radius=0)
        nucl_mask = nucle_seg > 0

        self.current_nucl_mask = nucl_mask
        self.nuclei_segmentation[self.current_file] = nucl_mask

    def segment_nuclei_cellpose(self, channel):

        ch = self.channels.index(channel)

        if self.model is None:
            print("No cellpose model provided")
        else:
            nucle_seg = utils.segment_nuclei_cellpose(
                self.current_image[:, :, ch], self.model, self.cellpose_diam
            )
            nucl_mask = nucle_seg > 0

            self.current_nucl_mask = nucl_mask
            self.nuclei_segmentation[self.current_file] = nucl_mask

    def segment_nucleiML(self, channel):

        ch = self.channels.index(channel)
        image = self.current_image[:, :, ch]
        im_rescaled = skimage.transform.rescale(image, 0.5)
        filtered, filter_names = utils.filter_sets(im_rescaled)

        # classify all pixels and update the segmentation layer
        all_pixels = pd.DataFrame(index=np.arange(len(np.ravel(im_rescaled))))
        for ind, x in enumerate(filter_names):
            all_pixels[x] = np.ravel(filtered[ind])
        predictions = self.ml.predict(all_pixels)

        predicted_image = np.reshape(predictions, im_rescaled.shape)
        if self.minsize > 0:
            predicted_image = utils.remove_small(predicted_image, self.minsize)

        predicted_image_upscaled = skimage.transform.resize(
            predicted_image, image.shape, order=0, preserve_range=True
        )

        self.nuclei_segmentation[self.current_file] = predicted_image_upscaled == 2

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(self.nuclei_segmentation[self.current_file], cmap = 'gray')
        # plt.show()
        

    def segment_bacteria(self, channel):

        rot_templ = -np.ones((5,5))
        rot_templ[:,1:-1] = 1

        n_std = 20
        # recover nuclei and cell mask
        nucl_mask = self.nuclei_segmentation[self.current_file]
        cell_mask = self.cell_segmentation[self.current_file]

        # remove bright nuclei regions
        cell_mask = cell_mask & ~nucl_mask

        # calculate median filter image
        self.calculate_median(channel)
        image = self.current_image_med

        # calculate an intensity threshold by fitting a gaussian on background
        out, _ = utils.fit_gaussian_hist(image[cell_mask], plotting=False)
        intensity_th = out[0][1] + n_std * np.abs(out[0][2])
        intensity_mask = image > intensity_th

        # rotate image over a series of angles and do template matching
        # this has the advantage that the template is always the same and 
        # corresponds to the true mode i.e. bright band with dark borders
        rot_im = []
        for ind, alpha in enumerate(np.arange(0, 180, 18)):
            im_rot = skimage.transform.rotate(image, alpha,preserve_range=True)
            im_match = match_template(im_rot, rot_templ, pad_input=True)
            im_unrot = skimage.transform.rotate(im_match, -alpha,preserve_range=True)
            rot_im.append(im_unrot)
        all_match = np.stack(rot_im,axis = 0)

        # keep only regions matching well in the rotational match volume
        rotation_vol = all_match > self.corr_threshold
        # label remaining regions
        rotation_vol_label = skimage.measure.label(rotation_vol)
        # merge regions corresponding to angles around 0 and 180 (periodic bounaries)
        for x in range(rotation_vol_label.shape[1]):
            for y in range(rotation_vol_label.shape[2]):
                if rotation_vol_label[0, x, y] > 0 and rotation_vol_label[-1, x, y] > 0:
                    rotation_vol_label[
                        rotation_vol_label == rotation_vol_label[-1, x, y]
                    ] = rotation_vol_label[0, x, y]

        # measure region properties
        rotation_vol_props = pd.DataFrame(
            skimage.measure.regionprops_table(
                rotation_vol_label, properties=("label", "area", "centroid")
            )
        )

        # keep only regions with a minimum number of matching voxels
        sel_labels = rotation_vol_props[
            rotation_vol_props.area > self.min_corr_vol
        ].label.values
        indices = np.array(
            [
                i if i in sel_labels else 0
                for i in np.arange(rotation_vol_props.label.max() + 1)
            ]
        )

        # relabel and project. In the projection we assume there are no
        # overlapping regions
        new_label_image = indices[rotation_vol_label]
        new_label_image_proj = np.max(new_label_image, axis=0)
        new_label_image_proj = new_label_image_proj * cell_mask * intensity_mask
        self.bacteria_segmentation[self.current_file] = new_label_image_proj

        self.all_match = all_match
        self.rotation_vol_label = rotation_vol_label

    def calculate_threshold(self):

        binval, binpos = np.histogram(
            np.ravel(self.current_image_med), bins=np.arange(0, 1000, 10)
        )
        hard_threshold = 1.5 * binpos[np.argmax(binval)]
        self.hard_threshold = hard_threshold

    def bact_calc_intensity_channels(self):

        # bact_labels = skimage.morphology.label(self.bact_mask)
        bact_labels = self.bacteria_segmentation[self.current_file]
        if bact_labels.max() > 0:
            intensities = {
                self.channels[x]: skimage.measure.regionprops_table(
                    bact_labels,
                    self.current_image[:, :, x],
                    properties=("mean_intensity", "label"),
                )["mean_intensity"]
                for x in range(len(self.channels))
                if self.channels[x] is not None
            }

            self.bacteria_channel_intensities[self.current_file] = intensities

    def bact_measure(self):
        # bact_mask = self.bacteria_segmentation[self.current_file]
        # bact_labels = skimage.morphology.label(bact_mask)

        bact_labels = self.bacteria_segmentation[self.current_file]
        if bact_labels.max() > 0:
            dataframes = []
            for x in range(len(self.channels)):
                if self.channels[x] is not None:
                    measurements = skimage.measure.regionprops_table(
                        bact_labels,
                        self.current_image[:, :, x],
                        properties=("mean_intensity", "label", "area", "eccentricity"),
                    )
                    dataframes.append(
                        pd.DataFrame(
                            {
                                **measurements,
                                **{"channel": self.channels[x]},
                                **{"filename": self.current_file},
                            }
                        )
                    )

            measure_df = pd.concat(dataframes)
            self.bact_measurements[self.current_file] = measure_df

    def run_analysis(self, nucl_channel, bact_channel, cell_channel, filepath):

        self.nucl_channel = nucl_channel
        self.bact_channel = bact_channel
        self.cell_channel = cell_channel

        self.import_file(filepath)

        if self.use_ml:
            if self.ml is None:
                print("No ML training available. Load one.")
                return False
            else:
                self.segment_nucleiML(nucl_channel)
        elif self.use_cellpose:
            self.segment_nuclei_cellpose(nucl_channel)
        else:
            self.segment_nuclei(nucl_channel)

        self.segment_cells(cell_channel)
        self.segment_bacteria(bact_channel)
        self.bact_calc_intensity_channels()
        self.bact_measure()

        # self.bacteria_segmentation[self.current_file] = self.bacteria_segmentation[self.current_file]*self.cell_segmentation[self.current_file]
        return True

    def save_segmentation(self):

        if self.folder_name is None:
            print("No folder_name specified")
            return None

        if not os.path.isdir(self.folder_name + "/Segmented/"):
            os.makedirs(self.folder_name + "/Segmented/", exist_ok=True)
        file_to_save = (
            self.folder_name
            + "/Segmented/"
            + os.path.split(self.folder_name)[-1]
            + ".pkl"
        )
        with open(file_to_save, "wb") as f:
            to_export = {
                "bact_channel": self.bact_channel,
                "nucl_channel": self.nucl_channel,
                "cell_channel": self.cell_channel,
                "minsize": self.minsize,
                "fillholes": self.fillholes,
                "nuclei_segmentation": self.nuclei_segmentation,
                "bacteria_segmentation": self.bacteria_segmentation,
                "cell_segmentation": self.cell_segmentation,
                "annotations": self.annotations,
                "bacteria_channel_intensities": self.bacteria_channel_intensities,
                "bact_measurements": self.bact_measurements,
                "channels": self.channels,
                "all_files": self.all_files,
                "ml": self.ml,
                "result": self.result,
            }
            pickle.dump(to_export, f)

    def load_segmentation(self, b=None):

        file_to_load = (
            self.folder_name
            + "/Segmented/"
            + os.path.split(self.folder_name)[-1]
            + ".pkl"
        )
        if not os.path.isfile(file_to_load):
            print("No analysis found")
        else:
            with open(file_to_load, "rb") as f:
                temp = pickle.load(f)

            for k in temp.keys():
                setattr(self, k, temp[k])

            print("Loading Done")

    def show_segmentation(self, local_file):
        filepath = self.folder_name + "/" + local_file
        if self.bacteria_segmentation[local_file] is None:
            print("not yet segmented")

        # else:
        # if local_file != self.current_file:
        self.import_file(filepath)

        viewer = napari.Viewer(ndisplay=2)
        for ind, c in enumerate(self.channels):
            if c is not None:
                image_name = self.current_file + "_" + c
                viewer.add_image(self.current_image[:, :, ind], name=image_name)
        if self.bacteria_segmentation[local_file] is not None:
            viewer.add_labels(
                # skimage.morphology.label(self.bacteria_segmentation[local_file]),
                self.bacteria_segmentation[local_file],
                name="bactseg",
            )
            viewer.add_labels(
                skimage.morphology.label(self.nuclei_segmentation[local_file]),
                name="nucleiseg",
            )
            viewer.add_labels(
                skimage.morphology.label(self.cell_segmentation[local_file]),
                name="cellseg",
            )
        self.viewer = viewer
        self.create_key_bindings()

    def create_key_bindings(self):

        self.viewer.bind_key("w", self.move_forward_callback)
        self.viewer.bind_key("b", self.move_backward_callback)

    def move_forward_callback(self, viewer):

        current_file_index = self.all_files.index(self.current_file)
        current_file_index = (current_file_index + 1) % len(self.all_files)
        self.load_new_view(current_file_index)

    def move_backward_callback(self, viewer):

        current_file_index = self.all_files.index(self.current_file)
        current_file_index = current_file_index - 1
        if current_file_index<0:
            current_file_index = len(self.all_files)
        self.load_new_view(current_file_index)

    def load_new_view(self, current_file_index):

        # local_file = os.path.split(self.all_files[current_file_index])[1]
        local_file = self.all_files[current_file_index]

        if self.bacteria_segmentation[local_file] is None:
            print("not yet segmented")

        else:
            self.import_file(self.folder_name + "/" + local_file)
            for ind, c in enumerate(self.channels):
                if c is not None:
                    layer_index = [
                        x.name.split(".")[1].split("_")[1] if "." in x.name else x.name
                        for x in self.viewer.layers
                    ].index(c)
                    self.viewer.layers[layer_index].data = self.current_image[:, :, ind]
                    self.viewer.layers[layer_index].name = self.current_file + "_" + c

            # self.viewer.layers[-3].data = skimage.morphology.label(
            #    self.bacteria_segmentation[local_file]
            self.viewer.layers[-3].data = self.bacteria_segmentation[local_file]

            self.viewer.layers[-2].data = skimage.morphology.label(
                self.nuclei_segmentation[local_file]
            )
            self.viewer.layers[-1].data = skimage.morphology.label(
                self.cell_segmentation[local_file]
            )

