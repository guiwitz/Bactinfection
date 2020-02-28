"""
Class implementing an interactive ipywidgets gui for segmentation
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3

import os
from pathlib import Path
import ipywidgets as ipw
import pickle
from IPython.display import clear_output

from .annotateml import Annotate
from .segmentation import Bact
from .folders import Folders


class Gui(Bact):
    def __init__(self):

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

        self.outcheck = ipw.Output()

        self.channel_field = ipw.Text(
            description="Channels",
            layout={"width": "700px"},
            value="DAPI, GFP, Phal, Unknow",
        )

        self.minsize_field = ipw.IntText(
            description="Minimal nucleus size",
            layout={"width": "200px"},
            style={"description_width": "initial"},
            value=0,
        )
        self.minsize_field.observe(self.update_minsize, names="value")

        self.fillholes_checks = ipw.Checkbox(description="Fill holes", value=False)
        self.fillholes_checks.observe(self.update_fillholes, names="value")

        self.use_ml_check = ipw.Checkbox(description="Use ML for nuclei")
        self.use_ml_check.observe(self.update_useml, names="value")

        self.nucl_channel_seg = ipw.Select(
            options=self.channel_field.value.replace(" ", "").split(","),
            layout={"width": "200px"},
            style={"description_width": "initial"},
            value=None,
        )
        self.bact_channel_seg = ipw.Select(
            options=self.channel_field.value.replace(" ", "").split(","),
            layout={"width": "200px"},
            style={"description_width": "initial"},
            value=None,
        )
        self.cell_channel_seg = ipw.Select(
            options=self.channel_field.value.replace(" ", "").split(","),
            layout={"width": "200px"},
            style={"description_width": "initial"},
            value=None,
        )

        self.nucl_channel_seg.observe(self.update_nucl_channel, names="value")
        self.bact_channel_seg.observe(self.update_bact_channel, names="value")
        self.cell_channel_seg.observe(self.update_cell_channel, names="value")

        self.channel_field.observe(self.update_values, names="value")
        self.out = ipw.Output()

        self.load_button = ipw.Button(
            description="Load ML and/or segmentation",
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.load_button.on_click(self.load_existing)

        self.load_otherML_button = ipw.Button(description="Load alternative ML")
        self.load_otherML_button.on_click(self.load_otherML)
        self.MLfolder = Folders()

        self.analyze_button = ipw.Button(
            description="Run segmentation",
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.analyze_button.on_click(self.run_interactive_analysis)

        self.save_button = ipw.Button(
            description="Save segmentation",
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.save_button.on_click(self.save_interactive_segmentation)

        self.show_button = ipw.Button(
            description="Show segmentation",
            layout={"width": "200px"},
            style={"description_width": "initial"},
        )
        self.show_button.on_click(self.show_interactive_segmentation)

        self.button_temp = ipw.Button()
        self.sel_temp = ipw.Select()

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

        self.get_filenames(None)
        self.initialize_output()
        self.update_values(None)

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

    def update_values(self, change):
        """update channel parameters"""

        self.channels = self.channel_field.value.replace(" ", "").split(",")
        self.nucl_channel_seg.options = self.channels
        self.bact_channel_seg.options = self.channels
        self.cell_channel_seg.options = self.channels

        self.nucl_channel_seg.value = None
        self.bact_channel_seg.value = None
        self.cell_channel_seg.value = None

    def update_nucl_channel(self, change):

        self.nucl_channel = change["new"]

    def update_bact_channel(self, change):

        self.bact_channel = change["new"]

    def update_cell_channel(self, change):

        self.cell_channel = change["new"]

    def update_minsize(self, change):

        self.minsize = change["new"]

    def update_fillholes(self, change):

        self.fillholes = change["new"]

    def update_useml(self, change):

        self.use_ml = change["new"]

    def load_existing(self, b):

        with self.out:
            clear_output()
            self.load_segmentation()

            temp_nucl = self.nucl_channel
            temp_bact = self.bact_channel
            temp_cell = self.cell_channel
            print(self.nucl_channel)

            self.channel_field.value = ", ".join(self.channels)
            self.nucl_channel_seg.value = temp_nucl
            self.bact_channel_seg.value = temp_bact
            self.cell_channel_seg.value = temp_cell
            self.minsize_field.value = self.minsize
            self.fillholes_checks.value = self.fillholes
            print(self.nucl_channel)

    def load_otherML(self, b):
        with self.out:
            clear_output()
            if len(self.MLfolder.file_list.value) == 0:
                print("Pick an ML file")
            else:
                file_to_load = (
                    self.MLfolder.cur_dir.as_posix()
                    + "/"
                    + self.MLfolder.file_list.value[0]
                )
                with open(file_to_load, "rb") as f:
                    self.ml = pickle.load(f)
                    with self.out:
                        clear_output()
                        print("ML model loaded")

    def run_interactive_analysis(self, b):
        with self.out:
            clear_output()
            totfiles = len(self.all_files)
            if totfiles == 0:
                print("Load folder first")
                self.analyze_button.description = "Run segmentation"
            else:
                self.import_file(self.folder_name + "/" + self.all_files[0])
                if len(self.channels) > self.current_image.shape[2]:
                    print("Too many channels defined")
                    self.analyze_button.description = "Run segmentation"
                elif (self.nucl_channel_seg.value is None) or (
                    self.bact_channel_seg.value is None
                ):
                    print("No channel selected")
                    self.analyze_button.description = "Run segmentation"

                else:
                    for ind, f in enumerate(self.all_files):
                        self.analyze_button.description = (
                            "Processing file " + str(ind) + "/" + str(totfiles) + " ..."
                        )
                        valid = self.run_analysis(
                            self.nucl_channel_seg.value,
                            self.bact_channel_seg.value,
                            self.folder_name + "/" + f,
                        )
                        if not valid:
                            break
            self.analyze_button.description = "Run segmentation"

    def show_interactive_segmentation(self, b):
        with self.out:
            clear_output()
            if len(self.folders.file_list.value) > 0:
                self.show_segmentation(self.folders.file_list.value[0])
            else:
                self.show_segmentation(self.all_files[0])

    def save_interactive_segmentation(self, b):

        with self.out:
            clear_output()
            self.save_segmentation()
            print("Segmentation saved")

