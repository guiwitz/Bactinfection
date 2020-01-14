"""
Class implementing an interactive ipywidgets gui for segmentation
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3

import ipywidgets as ipw
import pickle
from IPython.display import clear_output

from .annotateml import Annotate
from .segmentation import Bact
from .folders import Folders

class Gui:
    
    def __init__(self):
        
        self.channel_field = ipw.Text(description = 'Channels',layout = {'width': '700px'},
                                value = 'DAPI, GFP, Phal, Unknow')
        
        self.minsize_field = ipw.IntText(description = 'Minimal nucleus size',layout = {'width': '200px'},
                                style = {'description_width': 'initial'}, value = 0)
        self.minsize_field.observe(self.update_minsize, names = 'value')
        self.fillholes_checks = ipw.Checkbox(description = 'Fill holes', value = False)
        self.fillholes_checks.observe(self.update_fillholes, names = 'value')

        
        #self.load_existing = ipw.Checkbox(description = 'Load existing')
        self.use_ml = ipw.Checkbox(description = 'Use ML for nuclei')
        self.use_ml.observe(self.update_useml, names='value')
        
        self.nucl_channel_seg = ipw.Select(options = self.channel_field.value.replace(' ','').split(','),
                                     layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.bact_channel_seg = ipw.Select(options = self.channel_field.value.replace(' ','').split(','),
                                     layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.channel_field.observe(self.update_values, names='value')
        self.out = ipw.Output()
        
        self.load_button = ipw.Button(description = 'Load ML and/or segmentation',layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.load_button.on_click(self.load_existing)
        
        self.load_otherML_button = ipw.Button(description = 'Load alternative ML')
        self.load_otherML_button.on_click(self.load_otherML)
        self.MLfolder = Folders()
        
        
        self.analyze_button = ipw.Button(description = 'Run segmentation',layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.analyze_button.on_click(self.run_analysis)
        
        self.save_button = ipw.Button(description = 'Save segmentation',layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.save_button.on_click(self.save_segmentation)
        
        self.show_button = ipw.Button(description = 'Show segmentation',layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.show_button.on_click(self.show_segmentation)
        
        self.button_temp = ipw.Button()
        self.sel_temp = ipw.Select()
        
        self.bact = Bact(channels=self.nucl_channel_seg.options, use_ml=self.use_ml.value)
        
    def update_values(self, change):
        """call-back function for file upload. Uploads selected files
        and completes the files list"""
        
        channels = change['new'].replace(' ','').split(',')
        self.nucl_channel_seg.options = channels
        self.bact_channel_seg.options = channels
        
    def update_minsize(self, change):
        
        self.bact.minsize = change['new']
        
    def update_fillholes(self, change):
        
        self.bact.fillholes = change['new']
        
    def update_useml(self, change):
        
        self.bact.use_ml = change['new']
        
    def load_existing(self,b):
        
        with self.out:
            clear_output()
            self.bact.load_segmentation()
            
        self.channel_field.value = ', '.join(self.bact.channels)
        self.nucl_channel_seg.value = self.bact.nucl_channel
        self.bact_channel_seg.value = self.bact.bact_channel
        self.minsize_field.value = self.bact.minsize
        self.fillholes_checks.value = self.bact.fillholes
        
        
    def load_otherML(self, b):
        with self.out:
            clear_output()
            if len(self.MLfolder.file_list.value) == 0:
                print('Pick an ML file')
            else:
                file_to_load = self.MLfolder.cur_dir.as_posix()+'/'+self.MLfolder.file_list.value[0]
                with open(file_to_load, 'rb') as f:
                    self.bact.ml = pickle.load(f)
            
            
    def run_analysis(self,b):
        with self.out:
            clear_output()
            totfiles = len(self.bact.all_files)
            for ind, f in enumerate(self.bact.all_files):
                self.analyze_button.description = 'Processing file '+str(ind)+'/'+str(totfiles)+' ...'
                valid = self.bact.run_analysis(self.nucl_channel_seg.value,self.bact_channel_seg.value, self.bact.folder_name+'/'+f)
                if not valid:
                    break
            self.analyze_button.description = 'Run segmentation'
        
    def save_segmentation(self,b):
        self.bact.save_segmentation()
        
        
    def show_segmentation(self, b):
        with self.out:
            clear_output()
            if len(self.bact.folders.file_list.value)>0:
                self.bact.show_segmentation(self.bact.folders.file_list.value[0])
            else:
                self.bact.show_segmentation(self.bact.all_files[0])
                    
        
    def start(self,b):
        #None
        

        self.annotate = Annotate(self.bact, channel=self.nucl_channel_seg.value)
        
        
        self.button_temp.on_click(self.annotate.create_napariview)
        self.sel_temp = self.annotate.select_file