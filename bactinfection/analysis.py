"""
Class implementing analysis of segmentation data
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019-2020
# License: BSD3

from . import utils
from .segmentation import Bact
from .folders import Folders

import os
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as ipw
from sklearn import mixture
import skimage.filters
from IPython.display import display, clear_output
import napari

import plotnine as pn
from plotnine import ggplot, geom_point, aes, geom_line, labels

pn.theme_set(pn.theme_classic(base_size=18, base_family="Helvetica"))


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

        Attributes
        ----------

        all_files : list
            list of files to process
        result: pandas dataframe
            one line per bacteria
        result_ran : pandas dataframe
            background pixel values for each image
        aggregated : pandas dataframe
            counts of bacteria for each channel
        threshold_global : pandas dataframe
            global background threshold for each channel
        GM : gaussian mixture model
            model for double gaussian fitting (unused)
        folders : Folders object


        """
        Bact.__init__(self)

        # define measurement attributes
        self.result = None
        self.result_ran = None
        self.aggregated = None
        self.aggregated_th = None
        self.threshold_global = None
        self.GM = None

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

        # create browser
        self.folders = Folders()
        self.folders.file_list.observe(self.get_filenames, names="options")

        # option to name saving/loading folder
        self.saveto_widget = ipw.Text(value="Segmented")
        self.saveto_widget.observe(self.update_saveto, names="value")

        # output widgets
        self.out = ipw.Output()
        self.out_plot = ipw.Output()
        self.out_plot2 = ipw.Output()

        # loading segmented data
        self.load_button = ipw.Button(description="Load segmentation")
        self.load_button.on_click(self.load_infos)

        # interactive visualization of detected bacteria in all channels
        self.show_analyzed_button = ipw.Button(
            description="Show selected bacteria",
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.show_analyzed_button.on_click(self.show_interactive_masks)

        ###### Histogram visualization ##################################
        # plotting hisrogram by channel and hour
        self.button_plotbyhour = ipw.Button(description="Plot by hour")
        self.button_plotbyhour.on_click(self.plot_byhour_callback)

        # selection of channel to plot for histogram
        self.sel_channel = ipw.SelectMultiple(
            options=[],
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )

        # selection of bin size for histogram
        self.bin_width = ipw.IntSlider(min=10, max=1000, value=50)

        # selection of hour to plot in histogram
        self.hour_select = ipw.Select(options=[], value=None)

        ###### Time-curve visualization ##################################
        # plot time curve
        self.plot_time_curve_button = ipw.Button(description="Plot time-curve")
        self.plot_time_curve_button.on_click(self.plot_time_curve)

        # selection of channel for time curves
        self.sel_channel_time = ipw.SelectMultiple(
            options=[],
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.sel_channel_time.observe(self.plot_time_curve, names="value")
        self.sel_channel_time2 = ipw.SelectMultiple(
            options=[],
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.sel_channel_time2.observe(self.plot_time_curve, names="value")

        # saving time-curve plot
        self.save_time_curve_plot_button = ipw.Button(
            description="Save Plot time-curve"
        )
        self.save_time_curve_plot_button.on_click(self.save_time_curve_plot)

        # get current file parameters
        self.get_filenames()

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

        # load segmentation data
        # self.load_analysis_button = ipw.Button(description="Load segmentation")
        # self.load_analysis_button.on_click(self.load_analysis)

        # self.save_analysis_button = ipw.Button(description="Save analysis")
        # self.save_analysis_button.on_click(self.save_analysis)

    def change_default(self, b):
        """Update default path using manual text input"""

        new_path = os.path.normpath(self.default_path_text.value)
        with self.out:
            if not os.path.isdir(new_path):
                print("Not a correct path")
        with open(os.path.join(self.notebook_path, "settings.txt"), "w") as f:
            f.write(new_path)
            self.folders.cur_dir = Path(".").resolve()
            self.folders.refresh()

    def change_default_from_browser(self, b):
        """Update default path using current interactive folder setting"""

        new_path = self.folders.cur_dir.as_posix()
        self.default_path_text.value = new_path
        with open(os.path.join(self.notebook_path, "settings.txt"), "w") as f:
            f.write(new_path)

    def update_saveto(self, change):
        self.saveto = change["new"]

    def get_filenames(self, change=None):
        """Initialize file list with oir/oib files present in folder"""

        self.all_files = [
            os.path.split(x)[1] for x in self.folders.cur_dir.glob("*.oir")
        ]
        self.all_files += [
            os.path.split(x)[1] for x in self.folders.cur_dir.glob("*.oib")
        ]
        self.all_files = [x for x in self.all_files if x[0] != '.']

        if len(self.all_files) > 0:
            self.current_file = os.path.split(self.all_files[0])[1]

        self.folder_name = self.folders.cur_dir.as_posix()
        self.initialize_output()
        self.clean_masks = {os.path.split(x)[1]: None for x in self.all_files}

    def load_infos(self, b=None):
        """Callback to Load segmentation data.
        Called by load_button."""

        self.load_button.description = "Loading..."

        with self.out:
            self.load_segmentation()

        self.sel_channel.options = self.channels
        self.sel_channel_time.options = self.channels
        self.sel_channel_time2.options = self.channels
        self.create_result()
        self.hour_select.options = self.nuclei_count.hour.unique()

        self.load_button.description = "Load segmentation"

    def create_result_with_th(self):
        """Collect measurements of all images into dataframe and
        count nuclei"""

        # calculate background values in all channels
        if self.result_ran is None:
            self.result_ran = self.bact_calc_mean_background()

        if self.result is None:
            # calcualate bacteria intensities in all channels
            measurements = pd.concat(
                [
                    self.bact_measurements[x]
                    for x in self.bact_measurements
                    if self.bact_measurements[x] is not None
                ]
            )

            measurements = self.parse_hour_replicates(measurements)

            self.result = measurements
        
        # # count nuclei
        # nuclei_count = self.count_nuclei()
        # self.nuclei_count = nuclei_count

    def create_result(self):
        '''Simple count bacteria, cells, actin tails'''

        self.bact_count = self.count_objects(
            object_dict=self.bacteria_segmentation, name='number_bacteria')
        self.nuclei_count = self.count_objects(
            object_dict=self.nuclei_segmentation, name='number_nuclei')
        self.actin_count = self.count_objects(
            object_dict=self.actin_segmentation, name='number_actin')
        
        self.data_aggregation()

    def count_nuclei(self):
        """Create dataframe with nuclei count for each image"""
        dict_list = []
        for x in self.nuclei_segmentation:
            if self.nuclei_segmentation[x] is not None:
                num_nucl = None
                if self.nuclei_segmentation[x].max() > 0:
                    num_nucl = np.sum(np.unique(self.nuclei_segmentation[x]) > 0)
                dict_list.append({"filename": x, "number_nuclei": num_nucl})
        nuclei_count = pd.DataFrame(dict_list)

        nuclei_count = self.parse_hour_replicates(nuclei_count)

        return nuclei_count

    def count_objects(self, object_dict, name):
        """Create dataframe with object-name count for each image"""
        dict_list = []
        for x in object_dict:
            if object_dict[x] is not None:
                num_obj = None
                if object_dict[x].max() > 0:
                    num_obj = np.sum(np.unique(object_dict[x]) > 0)
                dict_list.append({"filename": x, name: num_obj})
        object_count = pd.DataFrame(dict_list)

        object_count = self.parse_hour_replicates(object_count)

        return object_count

    def parse_hour_replicates(self, dataframe):
        """Use the filename parameter of dataframe to extract the hour
        and replicate values of the file"""

        dataframe["hour"] = dataframe.filename.apply(
            lambda x: int(re.findall("\_(\d+)h.+\_", x)[0])
        )
        dataframe["replicate"] = dataframe.filename.apply(
            lambda x: int(re.findall("\_(\d+)\.", x)[0])
            if len(re.findall("\_(\d+)\.", x)) > 0
            else 0
        )
        return dataframe

    def bact_calc_mean_background(self):
        """Calculate background intensity within cells but outside bacteria"""

        dict_list = []
        for f in self.all_files:

            self.import_file(f)
            random_intensities = {}

            # create mask of cell without bacteria
            cellmask = self.cell_segmentation[self.current_file]
            bactmask = self.bacteria_segmentation[self.current_file]
            cellmask = cellmask & ~bactmask
            cellmask = cellmask.astype(bool)

            for x in range(len(self.channels)):
                if self.channels[x] is not None:
                    int_values = self.current_image[:, :, x][cellmask]
                    out, _ = utils.fit_gaussian_hist(
                        int_values, plotting=False, maxbin=int_values.max(), binwidth=10
                    )
                    random_intensities = {
                        "filename": self.current_file,
                        "mean_intensity": out[0][1],
                        "std_intensity": np.abs(out[0][2]),
                        "channel": self.channels[x],
                        "pixvalues": int_values,
                    }

                    dict_list.append(random_intensities)

        random_measurements = pd.DataFrame(dict_list)

        random_measurements = self.parse_hour_replicates(random_measurements)

        return random_measurements

    def global_threshold(self):
        """Collect all background pixels for each channel and fit a log
        normal around the main peack to estimate background mu and sigma"""

        self.threshold_global = []
        if self.result is None:
            self.create_result_with_th()
        for x in range(len(self.channels)):
            out = None
            if self.channels[x] is not None:
                data = np.concatenate(
                    self.result_ran[
                        (self.result_ran.channel == self.channels[x])
                    ].pixvalues.values
                )
                data = np.log(data)
                ydata, xdata = np.histogram(data, bins=np.arange(0, data.max(), 0.05))
                xdata = [0.5 * (xdata[x] + xdata[x + 1]) for x in range(len(xdata) - 1)]
                # find first point lower than half max after peak
                first_low = (
                    np.argwhere(
                        (ydata > 0.5 * ydata.max())[ydata.argmax()::] == False
                    )[0][0]
                    + ydata.argmax()
                )
                out, _ = utils.fit_gaussian_hist(
                    data[data < xdata[first_low]],
                    plotting=False,
                    maxbin=xdata[first_low] + 0.5,
                    binwidth=0.05,
                )
            self.threshold_global.append(
                {
                    "channel": self.channels[x],
                    "back_value": np.exp(out[0][1] + 4 * out[0][2]),
                }
            )
        self.threshold_global = pd.DataFrame(self.threshold_global)

    def data_aggregation_with_threshold(self):
        """Collect number of bacteria above background threshold and per
        nucleus for each image and calculate the average value"""

        if self.threshold_global is None:
            self.global_threshold()
        # merge bacteria analysis and background analysis
        # merged = pd.merge(self.result, self.result_ran[['filename','mean_intensity','std_intensity','channel']],on = ['filename','channel'])
        merged = pd.merge(self.result, self.threshold_global, on="channel")

        # select cases where intensity is larger than background
        # merged['select'] = merged.apply(lambda row: row['mean_intensity_x']>row['mean_intensity_y']+10*row['std_intensity'],axis = 1)
        merged["select"] = merged.apply(
            lambda row: row["mean_intensity"] > row["back_value"], axis=1
        )
        selected = merged[merged.select]

        # group data by image and channel
        grouped = selected.groupby(["filename", "channel"])

        # count elements in each group i.e. ON bacteria for each image
        # in each channel
        aggregated_counts = grouped.agg(
            numbers=pd.NamedAgg(column="mean_intensity", aggfunc="count")
        ).reset_index()

        # combine those counts with the number of nuclei on each image
        complete = pd.merge(
            aggregated_counts,
            self.nuclei_count[["filename", "number_nuclei"]],
            on="filename",
        )

        # add hour and replicate infos
        complete = self.parse_hour_replicates(complete)

        # normalize bacteria counts by number of nuclei (on a per image basis)
        complete["normalized"] = complete["numbers"] / complete["number_nuclei"]

        # calculate average number of "bacteria/nuclei"
        averaged = complete.groupby(["channel", "hour"]).mean().reset_index()

        self.aggregated_th = averaged

        agg_renamed = self.aggregated.rename(columns=
                           {'number_bacteria':'Segmented bacteria',
                          'number_actin':'Segmented actin tails'})

        agg_renamed = agg_renamed.melt(
            id_vars='hour',
            value_vars=['Segmented bacteria','Segmented actin tails'],
            var_name='channel',
            value_name='normalized')

        complete = pd.concat([agg_renamed, self.aggregated_th])

        self.complete = complete
        self.sel_channel_time.options = complete.channel.unique()
        self.sel_channel_time2.options = complete.channel.unique()

    def data_aggregation(self):

        complete = pd.merge(
            self.nuclei_count[["filename", "number_nuclei"]],
            self.bact_count[["filename", "number_bacteria"]],
            on="filename"
        )
        complete = pd.merge(
            complete,
            self.actin_count[["filename", "number_actin"]],
            on="filename"
        )

        complete = self.parse_hour_replicates(complete)
        complete['number_bacteria'] = complete['number_bacteria'] / complete['number_nuclei']
        complete['number_actin'] = complete['number_actin'] / complete['number_nuclei']
        self.complete = complete
        self.complete = self.complete.fillna(0)

        # calculate average number of "bacteria/nuclei"
        averaged = complete.groupby(["hour"]).mean().reset_index()

        self.aggregated = averaged

    def create_color_list(self):
        """Create a color list that assigns green to GFP and blue to DAPI
        if those channels are present in the user-defined channels"""

        default_col = {"GFP": "g", "DAPI": "b"}
        other_colors = ["red", "purple"]
        colors = []
        count = 0
        for x in self.sel_channel_time.value:
            if default_col.get(x) is not None:
                colors.append(default_col.get(x))
            else:
                colors.append(other_colors[count])
                count += 1
        return colors

    def plot_time_curve_by_channel(self, b=None):
        """Callback to polot time curve of number of bacteria/nuclei for
        each selected channel. Called by plot_time_curve_button."""

        if self.aggregated is None:
            self.data_aggregation()

        if len(self.sel_channel_time.value) == 0:
            print("Select at least one channel")
        else:
            subset = self.aggregated[
                self.aggregated.channel.isin(self.sel_channel_time.value)
            ].copy(deep=True)
            subset.loc[:, "channel"] = subset.channel.astype(
                pd.CategoricalDtype(self.sel_channel_time.value, ordered=True)
            )

            colors = self.create_color_list()

            myfig = (
                ggplot(subset, aes("hour", "normalized", color="channel"))
                + geom_point()
                + geom_line()
                + labels.xlab("Time [hours]")
                + labels.ylab("Average number of bacteria/nuclei")
                + pn.scale_colour_manual(
                    values=colors, labels=list(self.sel_channel_time.value), name=""
                )
                + pn.labs(colour="")
                + pn.scale_x_continuous(
                    breaks=np.sort(self.result.hour.unique()),
                    labels=list(np.sort(self.result.hour.unique()).astype(str)),
                )
            )

            self.time_curve_fig = myfig

            self.out_plot2.clear_output()
            with self.out_plot2:
                display(myfig)

    def plot_time_curve_with_threshold(self):
        toplot = self.aggregated.melt(
            id_vars='hour', value_vars=['number_bacteria', 'number_actin'],
            value_name='counts', var_name='Object')

        colors = self.create_color_list()

        myfig = (
                ggplot(toplot, aes("hour", "counts", color="Object"))
                + geom_point()
                + geom_line()
                + labels.xlab("Time [hours]")
                + labels.ylab("Average number of objects/nuclei")
                + pn.scale_colour_manual(
                    values=colors, labels=list(self.sel_channel_time.value), 
                    name=""
                )
                + pn.labs(colour="")
                + pn.scale_x_continuous(
                    breaks=np.sort(self.result.hour.unique()),
                    labels=list(np.sort(self.result.hour.unique()).astype(str))
                )
            )

        self.time_curve_fig = myfig

        self.out_plot2.clear_output()
        with self.out_plot2:
            display(myfig)

    def plot_time_curve_segmented(self, b=None):
        '''Plot time curve'''

        self.out_plot2.clear_output()
        with self.out_plot2:
            fig, ax1 = plt.subplots()
            #color = 'tab:red'
            ax1.set_xlabel('Time (h)')
            ax1.set_ylabel('Number of bacteria')#, color=color)
            ax1.plot(
                self.aggregated.hour, self.aggregated.number_bacteria, '-o',
                color='black', label='Number of bacteria')
            ax1.tick_params(axis='y')#, labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            #color = 'tab:blue'
            ax2.set_ylabel('Number of actin tails')#, color=color)  # we already handled the x-label with ax1
            ax2.plot(
                self.aggregated.hour, self.aggregated.number_actin, '-s',
                color='black',label='Number of actin tails')
            ax2.tick_params(axis='y')#, labelcolor=color)

            fig.legend(loc=(0.15, 0.8))
            plt.xticks(ticks=np.unique(self.aggregated.hour))
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            self.time_curve_fig = fig

    def plot_time_curve(self, b=None):

        if self.aggregated_th is None:
            self.data_aggregation_with_threshold()

        subset = self.complete[
            self.complete.channel.isin(
                self.sel_channel_time.value)].copy(deep=True)
        subset2 = self.complete[
            self.complete.channel.isin(
                self.sel_channel_time2.value)].copy(deep=True)

        symbol_dict = {'Segmented bacteria':'o','Segmented actin tails': 's', 'DAPI': 'o',
                    'GFP': 's', 'Phal': 'v','Lamp1': 'd'}
        symbol_fill = {'Segmented bacteria':'full','Segmented actin tails': 'full', 'DAPI': 'none',
                    'GFP': 'none', 'Phal': 'none','Lamp1': 'none'}
        othersymbols = ['x','+']

        symbol_count = 0

        self.out_plot2.clear_output()
        with self.out_plot2:
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.set_xlabel('Time (h)')
            ax1.set_ylabel('Number of objects', color=color)
            for c in subset.channel.unique():
                if c in symbol_dict:
                    current_col = symbol_dict[c]
                    current_fill = symbol_fill[c]
                else:
                    current_col = othersymbols[symbol_count]
                    current_fill = 'full'
                    symbol_count += 1
                ax1.plot(
                    subset[subset.channel == c].hour, 
                    subset[subset.channel == c].normalized, marker=current_col,
                    fillstyle=current_fill, color=color, label=c)
            ax1.tick_params(axis='y', labelcolor=color)

            # Shrink current axis by 20%
            #box = ax1.get_position()
            #ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Number of objects', color=color)  # we already handled the x-label with ax1
            for c in subset2.channel.unique():
                if c in symbol_dict:
                    current_col = symbol_dict[c]
                    current_fill = symbol_fill[c]
                else:
                    current_col = othersymbols[symbol_count]
                    current_fill = 'full'
                    symbol_count += 1
                ax2.plot(
                    subset2[subset2.channel == c].hour,
                    subset2[subset2.channel == c].normalized, 
                    marker=current_col,
                    fillstyle=current_fill, color=color, label=c)
            ax2.tick_params(axis='y', labelcolor=color)



            # Put a legend to the right of the current axis
            fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xticks(ticks=np.unique(self.aggregated.hour))
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
        self.time_curve_fig = fig

    def plot_byhour_callback(self, b=None):
        """Callback function to plot bacteria intensities histograms.
        Called by button_plotbyhour and responds to selection changes 
        in channels and hour."""

        # calculate histograms if necessary
        self.create_result_with_th()

        self.bin_width.observe(self.plot_byhour_callback, names="value")
        self.hour_select.observe(self.plot_byhour_callback, names="value")
        self.sel_channel.observe(self.plot_byhour_callback, names="value")

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
        """Plot of bacteria intensities histograms"""

        if (len(channel) == 0) or (hour is None):
            print("select a channel and an hour")
        else:
            grouped = self.result.groupby("hour")
            sel_group = grouped.get_group(hour)
            channel_group = sel_group.groupby("channel")

            fig, ax = plt.subplots(figsize=(10, 7))
            for c in channel:

                hist_val, xdata = np.histogram(
                    channel_group.get_group(c).mean_intensity,
                    bins=np.arange(
                        0, channel_group.get_group(c).mean_intensity.max(), bin_width
                    ),
                    density=True,
                )
                xdata = np.array(
                    [0.5 * (xdata[x] + xdata[x + 1]) for x in range(len(xdata) - 1)]
                )

                ax.bar(
                    x=xdata,
                    height=hist_val,
                    width=xdata[1] - xdata[0],
                    color="gray",
                    label="Data",
                )

                if self.threshold_global is None:
                    self.global_threshold()
                limit = self.threshold_global[
                    self.threshold_global.channel == c
                ].back_value.values[0]
                ax.plot(
                    [limit, limit], [0, np.max(hist_val)], "green",
                )
            ax.legend()
            ax.set_title("Hour " + str(hour))
            plt.show()

    def save_time_curve_plot(self, b=None):
        """Callback to save time-curve plot.
        Called by save_time_curve_plot_button."""

        file_to_save = os.path.join(
            self.folder_name,
            self.saveto,
            os.path.split(self.folder_name)[-1] + "_timecurve.png",
        )
        '''self.time_curve_fig.save(
            file_to_save, width=6.4, height=4.8, units="in", verbose=False
        )'''
        self.time_curve_fig.savefig(file_to_save)

    def clean_mask(self, local_file):
        """Given global thresholds for all channels and bacteria detection masks,
        create cleaned masks with only bacteria above threshold.
        Serves as visual check."""

        filepath = os.path.join(self.folder_name, local_file)

        if self.threshold_global is None:
            self.global_threshold()

        if self.clean_masks[local_file] is None:
            self.clean_masks[local_file] = {x: None for x in self.channels}

            if self.current_file != local_file:
                self.import_file(filepath)
            for x, channel in enumerate(self.channels):
                if channel is not None:
                    image = self.current_image[:, :, x]
                    mask = self.bacteria_segmentation[local_file]
                    threshold = (
                        self.threshold_global.set_index("channel")
                        .loc[channel]
                        .back_value
                    )

                    # label and measure mask
                    mask_lab = skimage.morphology.label(mask)
                    mask_reg = pd.DataFrame(
                        skimage.measure.regionprops_table(
                            mask_lab, image, properties=("label", "mean_intensity")
                        )
                    )

                    # select bright regions
                    keep_labels = mask_reg[
                        mask_reg.mean_intensity > 2 * threshold
                    ].label.values

                    # create list of ON indices
                    indices = np.array(
                        [
                            i if i in keep_labels else 0
                            for i in np.arange(mask_reg.label.max() + 1)
                        ]
                    )

                    # create new maks keeping only ON labels
                    clean_labels = indices[mask_lab]

                    self.clean_masks[local_file][channel] = clean_labels

    def split(self, data):
        """Fit data using a gaussian mixture"""

        X = np.reshape(data, (-1, 1))
        GM = mixture.GaussianMixture(n_components=2)
        GM.fit(X)
        self.GM = GM

    """
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
    """

    """
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
    """

    def show_interactive_masks(self, b):
        """Call back to start interactive visualisation of cleaned masks.
        Called by show_analyzed_button."""
        with self.out:
            clear_output()
            self.show_clean_masks(self.all_files[0])

    def show_clean_masks(self, local_file):
        """napari visualization of all channels and their corresponding
        cleaned masks"""

        filepath = os.path.join(self.folder_name, local_file)
        self.import_file(filepath)
        self.clean_mask(local_file)

        viewer = napari.Viewer(ndisplay=2)
        for ind, c in enumerate(self.channels):
            if c is not None:
                image_name = self.current_file + "_" + c
                viewer.add_image(self.current_image[:, :, ind], name=image_name)
                viewer.add_labels(self.clean_masks[local_file][c], name="mask_" + c)

        self.viewer = viewer
        self.create_key_bindings_clean_mask()

    def create_key_bindings_clean_mask(self):
        """Add key bindings to go forward and backward accross images"""

        self.viewer.bind_key("w", self.cleanmask_forward_callback)
        self.viewer.bind_key("b", self.cleanmask_backward_callback)

    def cleanmask_forward_callback(self, viewer):
        """move to next image"""

        current_file_index = self.all_files.index(self.current_file)
        current_file_index = (current_file_index + 1) % len(self.all_files)
        self.load_new_cleanmask(current_file_index)

    def cleanmask_backward_callback(self, viewer):
        """move to previous image"""

        current_file_index = self.all_files.index(self.current_file)
        current_file_index = current_file_index - 1
        if current_file_index < 0:
            current_file_index = len(self.all_files)
        self.load_new_cleanmask(current_file_index)

    def load_new_cleanmask(self, current_file_index):
        """Replace current data in napari visualisation with data
        of new image. Called when moving forward/backward in files."""

        local_file = self.all_files[current_file_index]
        self.import_file(os.path.join(self.folder_name, local_file))
        self.clean_mask(local_file)

        for ind, c in enumerate(self.channels):
            if c is not None:
                layer_index = [
                    x.name.split(".")[1].split("_")[1] if "." in x.name else x.name
                    for x in self.viewer.layers
                ].index(c)
                self.viewer.layers[layer_index].data = self.current_image[:, :, ind]
                self.viewer.layers[layer_index].name = self.current_file + "_" + c

                self.viewer.layers["mask_" + c].data = self.clean_masks[local_file][c]
