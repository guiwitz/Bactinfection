"""
Class implementing analysis of segmentation data
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3


import os
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as ipw
import pickle
from sklearn import mixture
import skimage.filters

from . import utils
from .segmentation import Bact
from .folders import Folders


font = {
    "family": "sans-serif",
    "color": "black",
    "weight": "normal",
    "size": 16,
}


class Analysis(Bact):
    def __init__(self):

        """Standard __init__ method.
        
        Parameters
        ----------
        bact : Bact object
            Bact object
        
        
        Attributes
        ----------
            
        all_files = list
            list of files to process
        
        """
        Bact.__init__(self)

        # Recover and handle default settings
        self.notebook_path = os.getcwd()
        self.default_path_text = ipw.Text(
            layout={"width": "600px"}, style={"description_width": "initial"}
        )
        self.default_path_button = ipw.Button(
            description="Update default path",
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.default_path_button.on_click(self.change_default)

        self.default_path_from_browser_button = ipw.Button(
            description="Update using browser",
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.default_path_from_browser_button.on_click(self.change_default_from_browser)

        self.folders = Folders()
        self.folders.file_list.observe(self.get_filenames, names="options")

        self.out = ipw.Output()
        self.out_plot = ipw.Output()
        self.out_plot2 = ipw.Output()

        self.load_button = ipw.Button(description="Load segmentation")
        self.load_button.on_click(self.load_infos)

        self.sel_channel = ipw.SelectMultiple(
            options=[],
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.sel_channel.observe(self.plot_byhour_callback, names="value")

        self.sel_channel_time = ipw.SelectMultiple(
            options=[],
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.sel_channel_time.observe(self.plot_time_curve, names="value")

        self.button_plotbyhour = ipw.Button(description="Plot by hour")
        self.button_plotbyhour.on_click(self.plot_byhour_callback)

        self.bin_width = ipw.IntSlider(min=10, max=1000, value=50)
        self.bin_width.observe(self.plot_byhour_callback, names="value")
        self.hour_select = ipw.Select(options=[], value=None)
        self.hour_select.observe(self.plot_byhour_callback, names="value")

        self.load_analysis_button = ipw.Button(description="Load analysis")
        self.load_analysis_button.on_click(self.load_analysis)

        self.save_analysis_button = ipw.Button(description="Save analysis")
        self.save_analysis_button.on_click(self.save_analysis)

        self.plot_time_curve_button = ipw.Button(description="Plot time-curve")
        self.plot_time_curve_button.on_click(self.plot_time_curve)

        self.save_time_curve_plot_button = ipw.Button(
            description="Save Plot time-curve"
        )
        self.save_time_curve_plot_button.on_click(self.save_time_curve_plot)

        self.GM = None

        # recover default path
        if os.path.isfile("settings.txt"):
            with open("settings.txt", "r") as f:
                default_path = f.readline()
                os.chdir(default_path)
                self.folders.cur_dir = Path(".").resolve()
                self.folders.refresh()
                self.default_path_text.value = default_path
        else:
            with open("settings.txt", "w") as f:
                f.write(os.getcwd())

    def change_default(self, b):

        new_path = os.path.normpath(self.default_path_text.value)
        with self.out:
            if not os.path.isdir(new_path):
                print("Not a correct path")
        with open(os.path.join(self.notebook_path, "settings.txt"), "w") as f:
            f.write(new_path)
            self.folders.cur_dir = Path(".").resolve()
            self.folders.refresh()

    def change_default_from_browser(self, b):

        new_path = self.folders.cur_dir.as_posix()
        self.default_path_text.value = new_path
        with open(os.path.join(self.notebook_path, "settings.txt"), "w") as f:
            f.write(new_path)

    def get_filenames(self, change=None):
        """Initialize file list with lsm files present in folder"""

        # with self.outcheck:
        #    print('here')
        self.all_files = [
            os.path.split(x)[1] for x in self.folders.cur_dir.glob("*.oir")
        ]
        if len(self.all_files) > 0:
            self.current_file = os.path.split(self.all_files[0])[1]

        self.folder_name = self.folders.cur_dir.as_posix()
        self.initialize_output()

    def load_infos(self, b=None):

        self.load_analysis_button.description = "Loading..."
        self.load_button.description = "Loading..."

        with self.out:
            self.load_segmentation()

        self.sel_channel.options = self.channels
        self.sel_channel_time.options = self.channels
        self.create_result()
        self.hour_select.options = self.result.hour.unique()

        self.load_analysis_button.description = "Load segmentation"
        self.load_button.description = "Load segmentation"

    def create_result(self):

        if not hasattr(self, "random_channel_intensities"):
            # self.bact_calc_random_intensity_channels()
            self.bact_calc_mean_background()

        empty_dict = {x: [] for x in self.channels if x is not None}
        empty_dict_ran = {x: [] for x in self.channels if x is not None}

        empty_dict["hour"] = []
        empty_dict["replicate"] = []
        empty_dict["filename"] = []

        empty_dict_ran["hour"] = []
        empty_dict_ran["replicate"] = []
        empty_dict_ran["filename"] = []

        for x in self.bacteria_channel_intensities:
            if self.bacteria_channel_intensities[x] is not None:
                for c in self.channels:
                    if c is not None:
                        numvals = len(self.bacteria_channel_intensities[x][c])
                        empty_dict[c] += list(self.bacteria_channel_intensities[x][c])

                        numvals_ran = len(self.random_channel_intensities[x][c])
                        empty_dict_ran[c] += list(self.random_channel_intensities[x][c])

                empty_dict["hour"] += numvals * [int(re.findall("\_(\d+)h\_", x)[0])]
                empty_dict["filename"] += numvals * [x]

                empty_dict_ran["hour"] += numvals_ran * [
                    int(re.findall("\_(\d+)h\_", x)[0])
                ]
                empty_dict_ran["filename"] += numvals_ran * [x]

                index = re.findall("\_(\d+)\.", x)
                if len(index) == 0:
                    index = 0
                else:
                    index = int(index[0])
                empty_dict["replicate"] += numvals * [index]
                empty_dict_ran["replicate"] += numvals_ran * [index]

        empty_dict = pd.DataFrame(empty_dict)
        empty_dict_ran = pd.DataFrame(empty_dict_ran)
        self.result = empty_dict
        self.result_ran = empty_dict_ran

    def bact_calc_mean_background(self):

        ch = self.channels.index(self.cell_channel)
        self.random_channel_intensities = {}

        for f in self.all_files:

            self.import_file(f)
            random_intensities = {}
            sort_pix = np.sort(np.ravel(self.current_image[:, :, ch]))
            th = np.mean(sort_pix[0 : int(0.2 * len(sort_pix))])
            minfilt = skimage.filters.rank.minimum(
                self.current_image[:, :, ch] > th, skimage.morphology.disk(10)
            )

            mask = minfilt > 0
            mask = mask * (1 - self.bacteria_segmentation[self.current_file])
            mask = mask * (1 - self.nuclei_segmentation[self.current_file])
            mask = mask > 0

            for x in range(len(self.channels)):
                if self.channels[x] is not None:
                    random_intensities[self.channels[x]] = self.current_image[:, :, x][
                        mask
                    ]
                    self.random_channel_intensities[
                        self.current_file
                    ] = random_intensities

    def bact_calc_random_intensity_channels(self):

        num_points = 1000
        np.random.seed(42)

        ch = self.channels.index(self.bact_channel)

        self.random_channel_intensities = {}
        for f in self.all_files:

            self.import_file(f)
            random_intensities = {}

            # pick random points below Otsu threshold and that don't belong to segmentation
            th = skimage.filters.threshold_otsu(self.current_image[:, :, ch])
            binary_mask = self.current_image[:, :, ch] < th
            self.current_image[:, :, ch][~binary_mask] = 0

            ranpos = np.random.randint(
                [0, 0], self.current_image[:, :, ch].shape, (num_points, 2)
            )
            randpos_values = self.current_image[:, :, ch][ranpos[:, 0], ranpos[:, 1]]
            ranpos = ranpos[randpos_values > 0, :]

            for x in range(len(self.channels)):
                if self.channels[x] is not None:
                    randpos_values = self.current_image[:, :, x][
                        ranpos[:, 0], ranpos[:, 1]
                    ]
                    randpos_values = randpos_values[randpos_values > 0]
                    random_intensities[self.channels[x]] = randpos_values

                    self.random_channel_intensities[
                        self.current_file
                    ] = random_intensities

    def pool_intensities(self):
        pooled = {
            k: np.concatenate(
                [
                    self.bacteria_channel_intensities[x][k]
                    for x in self.bacteria_channel_intensities.keys()
                ]
            )
            for k in self.channels
            if k is not None
        }

        return pooled

    def plot_byhour_callback(self, b=None):

        self.out_plot.clear_output()
        with self.out_plot:

            self.plot_split_intensities(
                bin_width=self.bin_width.value,
                channel=self.sel_channel.value,
                hour=self.hour_select.value,
            )

    def plot_split_intensities(
        self, channel=None, hour=None, bin_width=10, min=0, max=3000
    ):
        if (len(channel) == 0) or (hour is None):
            print("select a channel and an hour")
        else:
            grouped = self.result.groupby("hour")
            sel_group = grouped.get_group(hour)
            grouped_ran = self.result_ran.groupby("hour")
            sel_group_ran = grouped_ran.get_group(hour)
            fig, ax = plt.subplots(figsize=(10, 7))
            for c in channel:
                self.split(sel_group[c].values)
                hist_val, xdata = np.histogram(
                    sel_group[c].values,
                    bins=np.arange(min, max, bin_width),
                    density=True,
                )
                xdata = np.array(
                    [0.5 * (xdata[x] + xdata[x + 1]) for x in range(len(xdata) - 1)]
                )
                ind1 = 0
                ind2 = 1
                ax.bar(
                    x=xdata,
                    height=hist_val,
                    width=xdata[1] - xdata[0],
                    color="gray",
                    label="Data",
                )

                # fit background
                out, _ = utils.fit_gaussian_hist(
                    sel_group_ran[c].values, plotting=False
                )
                # show histogram
                hist_val_ran, xdata_ran = np.histogram(
                    sel_group_ran[c].values,
                    bins=np.arange(min, max, bin_width),
                    density=True,
                )
                xdata_ran = np.array(
                    [
                        0.5 * (xdata_ran[x] + xdata_ran[x + 1])
                        for x in range(len(xdata_ran) - 1)
                    ]
                )
                ind1 = 0
                ind2 = 1
                ax.bar(
                    x=xdata_ran,
                    height=hist_val_ran,
                    width=xdata_ran[1] - xdata_ran[0],
                    color="red",
                    alpha=0.5,
                    label="Random",
                )

                ax.plot(
                    xdata,
                    self.normal_fit(
                        xdata,
                        self.GM.weights_[ind1],
                        self.GM.means_[ind1, 0],
                        self.GM.covariances_[ind1, 0, 0] ** 0.5,
                    ),
                    "b",
                    linewidth=2,
                    label="Cat1",
                )
                ax.plot(
                    xdata,
                    self.normal_fit(
                        xdata,
                        self.GM.weights_[ind2],
                        self.GM.means_[ind2, 0],
                        self.GM.covariances_[ind2, 0, 0] ** 0.5,
                    ),
                    "r",
                    linewidth=2,
                    label="Cat2",
                )
                # ax.hist(sel_group[c], label = c, alpha = 0.5, bins = np.arange(min,max,bin_width))
                ax.plot(
                    [out[0][1] + 3 * out[0][2], out[0][1] + 3 * out[0][2]],
                    [0, np.max(hist_val_ran)],
                    "green",
                )
            ax.legend()
            ax.set_title("Hour " + str(hour))
            plt.show()

    def calculage_time_curves(self):

        results = []
        grouped = self.result.groupby("hour")
        grouped_ran = self.result_ran.groupby("hour")

        for hour in self.hour_select.options:

            sel_group = grouped.get_group(hour)
            sel_group_ran = grouped_ran.get_group(hour)

            for c in self.sel_channel.options:
                # fit background
                out, _ = utils.fit_gaussian_hist(
                    sel_group_ran[c].values, plotting=False
                )

                # count events above threshold
                threshold = out[0][1] + 3 * out[0][2]
                num_above = np.sum(sel_group[c].values > threshold)
                newitem = {"channel": c, "hour": hour, "number": num_above}
                results.append(newitem)

        return results

    def plot_time_curve(self, b=None):

        res = self.calculage_time_curves()
        res_pd = pd.DataFrame(res)
        channel_group = res_pd.sort_values(by="hour").groupby("channel")
        self.out_plot2.clear_output()
        with self.out_plot2:

            if len(self.sel_channel_time.value) == 0:
                print("Select at least one channel")

            import itertools

            marker = itertools.cycle(("+", ".", "*", "v"))
            fig, ax = plt.subplots(figsize=(10, 7))
            for x, y in channel_group:
                if x in self.sel_channel_time.value:
                    plt.plot(
                        y.hour,
                        y.number,
                        "k" + next(marker) + "-",
                        label=x,
                        markersize=10,
                    )
            ax.set_xlabel("Time", fontdict=font)
            ax.set_ylabel("Number", fontdict=font)
            ax.legend()
            plt.show()
            self.time_curve_fig = fig

    def save_time_curve_plot(self, b=None):

        if not os.path.isdir(self.folder_name + "/Analyzed/"):
            os.makedirs(self.folder_name + "/Analyzed/", exist_ok=True)

        file_to_save = (
            self.folder_name
            + "/Analyzed/"
            + os.path.split(self.folder_name)[-1]
            + "_timecurve.png"
        )
        self.time_curve_fig.savefig(file_to_save)

    def plot_result_groupedby_hour(self, min=0, max=3000, bin_width=100, channels=None):

        if channels is None:
            channels = self.channels
        grouped = self.result.groupby("hour")
        grouped_ran = self.result_ran.groupby("hour")
        for x, y in grouped:

            fig, ax = plt.subplots(figsize=(10, 7))
            for c in channels:
                if c is not None:
                    self.GM = self.split(y[c].values)
                    hist_val, xdata = np.histogram(
                        y[c].values, bins=np.arange(min, max, bin_width), density=True
                    )
                    xdata = np.array(
                        [0.5 * (xdata[x] + xdata[x + 1]) for x in range(len(xdata) - 1)]
                    )
                    ind1 = 0
                    ind2 = 1
                    ax.bar(
                        x=xdata,
                        height=hist_val,
                        width=xdata[1] - xdata[0],
                        color="gray",
                        label="Data",
                    )

                    ax.plot(
                        xdata,
                        self.normal_fit(
                            xdata,
                            self.GM.weights_[ind1],
                            self.GM.means_[ind1, 0],
                            GM.covariances_[ind1, 0, 0] ** 0.5,
                        ),
                        "b",
                        linewidth=2,
                        label="Cat1",
                    )
                    ax.plot(
                        xdata,
                        self.normal_fit(
                            xdata,
                            self.GM.weights_[ind2],
                            self.GM.means_[ind2, 0],
                            GM.covariances_[ind2, 0, 0] ** 0.5,
                        ),
                        "r",
                        linewidth=2,
                        label="Cat2",
                    )
                    # ax.hist(y[c], label = c, alpha = 0.5, bins = np.arange(min,max,bin_width))

            ax.legend()
            ax.set_title(x)

    def split(self, data):
        X = np.reshape(data, (-1, 1))
        GM = mixture.GaussianMixture(n_components=2)
        GM.fit(X)
        self.GM = GM

    def normal_fit(self, x, a, x0, s):
        return (a / (s * (2 * np.pi) ** 0.5)) * np.exp(-0.5 * ((x - x0) / s) ** 2)

    def plot_hist_single_file(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        for k in self.bacteria_channel_intensities[self.select_file.value].keys():
            ax.hist(
                self.bacteria_channel_intensities[self.select_file.value][k],
                bins=np.arange(0, 3000, 100),
                density=True,
                alpha=0.5,
                label=k,
            )
        ax.legend()
        ax.set_title(self.select_file.value)
        plt.show()

    def save_analysis(self, b=None):
        if not os.path.isdir(self.folder_name + "/Analyzed/"):
            os.makedirs(self.folder_name + "/Analyzed/", exist_ok=True)
        file_to_save = (
            self.folder_name
            + "/Analyzed/"
            + os.path.split(self.folder_name)[-1]
            + ".pkl"
        )
        with open(file_to_save, "wb") as f:
            to_export = {
                "bact_channel": self.bact_channel,
                "nucl_channel": self.nucl_channel,
                "cell_channel": self.cell_channel,
                "bacteria_channel_intensities": self.bacteria_channel_intensities,
                "channels": self.channels,
                "all_files": self.all_files,
                "result": self.result,
            }
            pickle.dump(to_export, f)

    def load_analysis(self, b=None):

        file_to_load = (
            self.folder_name
            + "/Analyzed/"
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

