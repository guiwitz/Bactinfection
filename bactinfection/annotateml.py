"""
Class implementing an interactive annotation tool for ML segmentation
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3


import os
from pathlib import Path
import napari
import skimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as ipw
import pickle
from IPython.display import clear_output

from vispy.color.colormap import Colormap, ColorArray

import skimage.filters
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from . import utils
from .segmentation import Bact
from .folders import Folders

class Annotate(Bact):
    
    def __init__(self, bact = None, channel = 'DAPI', folder = None):

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
        
        #Recover and handle default settings
        self.notebook_path = os.getcwd()
        self.default_path_text = ipw.Text(layout = {'width': '600px'},style = {'description_width': 'initial'})
        self.default_path_button = ipw.Button(description = 'Update default path',layout = {'width': '200px'},
                                style = {'description_width': 'initial'})
        self.default_path_button.on_click(self.change_default)
        
        self.default_path_from_browser_button = ipw.Button(description = 'Update using browser',layout = {'width': '200px'},
                                style = {'description_width': 'initial'})
        self.default_path_from_browser_button.on_click(self.change_default_from_browser)

        self.folders = Folders()
        self.folders.file_list.observe(self.get_filenames, names='options')
        
        self.out = ipw.Output()
        
        self.channel_field = ipw.Text(description = 'Channels',layout = {'width': '700px'},
                                value = 'DAPI, GFP, Phal, Unknow')
        
        self.minsize_field = ipw.IntText(description = 'Minimal nucleus size',layout = {'width': '200px'},
                                style = {'description_width': 'initial'}, value = 0)
        self.minsize_field.observe(self.update_minsize, names = 'value')
        self.fillholes_checks = ipw.Checkbox(description = 'Fill holes', value = False)
        self.fillholes_checks.observe(self.update_fillholes, names = 'value')
        
        self.nucl_channel_seg = ipw.Select(options = self.channel_field.value.replace(' ','').split(','),
                                     layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.nucl_channel_seg.observe(self.update_nucl_channel, names='value')
        
        self.channel_field.observe(self.update_values, names='value')
        
        #recover default path
        if os.path.isfile('settings.txt'):
            with open('settings.txt','r') as f:
                default_path = f.readline()
                os.chdir(default_path)
                self.folders.cur_dir = Path('.').resolve()
                self.folders.refresh()
                self.default_path_text.value = default_path
        else:
            with open('settings.txt', 'w') as f:
                f.write(os.getcwd())
                        
        
        self.get_filenames(None)
        self.update_values(None)
                
        self.button_start = ipw.Button(description = 'Start annotating', layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.button_start.on_click(self.create_napariview)
        
        self.button_save = ipw.Button(description = 'Save annotations and ML',layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.button_save.on_click(self.save_state)
        
        self.button_load = ipw.Button(description = 'Load annotations and ML',layout = {'width': '200px'},style = {'description_width': 'initial'})
        self.button_load.on_click(self.load_state)
        
        self.names = ['Gauss $\sigma$=10', 'Gauss $\sigma$=20','Frangi','Prewitt','Meijering','Gauss+Laplace', 'Gradient',
                'Entropy','Roberts']
        
        self.annotations = {os.path.split(x)[1]: None for x in self.all_files}
            
        self.training_features = {os.path.split(x)[1]: None for x in self.all_files}
        self.training_targets = {os.path.split(x)[1]: None for x in self.all_files}
        
        

    def change_default(self, b):
        
        new_path = os.path.normpath(self.default_path_text.value)
        with self.out:
            if not os.path.isdir(new_path):
                print('Not a correct path')
        with open(os.path.join(self.notebook_path,'settings.txt'), 'w') as f:
            f.write(new_path)
            self.folders.cur_dir = Path('.').resolve()
            self.folders.refresh()
            
    def change_default_from_browser(self, b):
        
        new_path = self.folders.cur_dir.as_posix()
        self.default_path_text.value = new_path
        with open(os.path.join(self.notebook_path,'settings.txt'), 'w') as f:
            f.write(new_path)
            
            
    def get_filenames(self, change = None):
        """Initialize file list with lsm files present in folder"""
        
        #with self.outcheck:
        #    print('here')
        self.all_files = [
            os.path.split(x)[1] for x in self.folders.cur_dir.glob("*.oir")]
        if len(self.all_files)>0:
            self.current_file = os.path.split(self.all_files[0])[1]
            
        self.folder_name = self.folders.cur_dir.as_posix()
        self.initialize_output()
        
        
    def update_values(self, change):
        """call-back function for file upload. Uploads selected files
        and completes the files list"""
        
        self.channels = self.channel_field.value.replace(' ','').split(',')
        self.nucl_channel_seg.options = self.channels
        
    def update_minsize(self, change):
        
        self.minsize = change['new']
        
    def update_nucl_channel(self, change):
        
        self.nucl_channel = change['new']
    
    def update_fillholes(self, change):
        
        self.fillholes = change['new']
    
    def import_image(self, local_file = None):
        if local_file is None:
            if len(self.folders.file_list.value)>0:
                local_file = self.folders.file_list.value[0]
            else:
                local_file = self.all_files[0]
        
        
        filepath = self.folder_name+'/'+local_file
        
        self.import_file(filepath)
        ch = self.channels.index(self.nucl_channel_seg.value)
        self.image = skimage.transform.rescale(self.current_image[:,:,ch], 0.5)
    
    def create_napariview(self, b):
        """Initialize file list with lsm files present in folder"""
        
        self.get_training()
        self.import_image()
        #check number of channels
        if len(self.channels)>self.current_image.shape[2]:
            with self.out:
                print('Too many channels defined')
        else:
            self.calculate_filters(self.image)

            if self.annotations[self.current_file] is None:
                im_label = np.zeros(self.image.shape)
                im_label[0,0] = 1
                im_label[1,1] = 2
            else:
                im_label = self.annotations[self.current_file]

            self.view = napari.Viewer()
            #image layer
            self.image_layer = self.view.add_image(self.image)

            #labels layer. We create two labelled single-pixels for the two categories, so that 
            #sklearn stays happy after one draws the first label
            self.labels_layer = self.view.add_labels(data = im_label,name = 'manual label')
            self.labels_layer.colormap =('red_blue1', Colormap(colors = ColorArray(color = ((0,0,0,0),(1,0,0,1),(0,0,1,1))), controls=[0,0.1, 0.2, 1.], interpolation='zero'))

            #prediction layer
            prediction = self.segment_current_view()
            self.seg_layer = self.view.add_image(data = prediction,name = 'predict',
                               opacity = 0.2, contrast_limits = [0,2], colormap = 'hsv')
            self.seg_layer.colormap =('red_blue2', Colormap(colors = ColorArray(color = ((1,0,0,1),(0,0,1,1))), controls=[0, 0.8, 1.], interpolation='zero'))
            self.seg_layer.opacitiy = 0.5


            #bind mouse events
            self.add_callbacks()
        
    def change_image(self, change):
        
        self.update_viewer()
    
    def update_viewer(self, local_file = None):
        
        self.import_image(local_file)
        
        self.image_layer.data = self.image
        
        if self.annotations[self.current_file] is None:
            im_label = np.zeros(self.image_layer.data.shape)
            im_label[0,0] = 1
            im_label[1,1] = 2
        else:
            im_label = self.annotations[self.current_file]
        
        #calculate filters
        self.calculate_filters(self.image)
        #update training data using filters
        predicted = self.segment_current_view()
        
        self.labels_layer.data = im_label
        self.seg_layer.data = predicted.astype(np.uint8)
        
    def calculate_filters(self, image):

        self.all_filt, _ = utils.filter_sets(image)
    
    def get_training(self):
        
        for f in self.all_files:
            f = os.path.split(f)[1]
            current_labels = self.annotations[f]
            #capture new features
            if current_labels is not None:
                self.import_image(f)
                self.calculate_filters(self.image)
                
                features = pd.DataFrame(index=np.arange(np.sum(current_labels>0)))

                for ind, x in enumerate(self.names):
                    features[x] = np.ravel(self.all_filt[ind][current_labels>0])

                targets = pd.Series(np.ravel(current_labels[current_labels>0]).astype(int))

                self.training_features[f] = features
                self.training_targets[f] = targets
        
    def segment_current_view(self):
        
        #collect all features/targets
        collect_features=[]
        collect_targets=[]
        predicted_image = np.zeros(self.image_layer.data.shape)
        for x in self.training_features:
            if self.training_features[x] is not None:
                collect_features.append(self.training_features[x])
                collect_targets.append(self.training_targets[x])
                
        if len(collect_features)>0:
            features = pd.concat(collect_features).reset_index(drop = True)
            targets = pd.concat(collect_targets).reset_index(drop = True)


            #split train/test
            X, X_test, y, y_test = train_test_split(features, targets, 
                                                test_size = 0.2, 
                                                random_state = 42)

            #train a random forest classififer
            random_forrest = RandomForestClassifier(n_estimators=100)
            random_forrest.fit(X, y)
            self.ml = random_forrest

            #classify all pixels and update the segmentation layer
            all_pixels = pd.DataFrame(index=np.arange(len(np.ravel(self.image_layer.data))))
            for ind, x in enumerate(self.names):
                all_pixels[x] = np.ravel(self.all_filt[ind])
            predictions = random_forrest.predict(all_pixels)

            predicted_image = np.reshape(predictions, self.image_layer.data.shape)
            
            if self.minsize >0:
                predicted_image = utils.remove_small(predicted_image, self.bact.minsize)
            
        return predicted_image
    
        
    def mouse_callback(self, viewer, event):
    
        #clicking
        yield
        #dragging
        while event.type == 'mouse_move':
            yield
            
        #save labels
        self.annotations[self.current_file] = self.labels_layer.data
        
        #capture new features
        features = pd.DataFrame(index=np.arange(np.sum(self.labels_layer.data>0)))

        for ind, x in enumerate(self.names):
            features[x] = np.ravel(self.all_filt[ind][self.labels_layer.data>0])

        targets = pd.Series(np.ravel(self.labels_layer.data[self.labels_layer.data>0]).astype(int))
        
        self.training_features[self.current_file] = features
        self.training_targets[self.current_file] = targets
        
        predicted_image = self.segment_current_view()

        self.seg_layer.data = predicted_image.astype(np.uint8)
        
    def next_image(self, viewer):
        
        current_file_index = self.all_files.index(self.current_file)
        current_file_index = (current_file_index+1)%len(self.all_files)
        
        local_file = self.all_files[current_file_index]
        
        self.update_viewer(local_file)
        
    def previous_image(self, viewer):
        
        current_file_index = self.all_files.index(self.current_file)
        current_file_index = (current_file_index-1)
        
        if current_file_index<0:
            current_file_index = len(self.all_files)
        local_file = self.all_files[current_file_index]
        
        self.update_viewer(local_file)
        
    def save_state(self, b):
        self.save_segmentation()
        file_to_save = self.folder_name+'/Segmented/ml_model.pkl'
        with open(file_to_save, 'wb') as f:
            pickle.dump(self.ml, f)
            
    def load_state(self, b):
        
        with self.out:
            clear_output()
            self.load_segmentation()
            
        self.channel_field.value = ', '.join(self.channels)
        self.nucl_channel_seg.value = self.nucl_channel
        self.minsize_field.value = self.minsize
        self.fillholes_checks.value = self.fillholes
        
    def add_callbacks(self):
        
        self.view.mouse_drag_callbacks.append(self.mouse_callback)
        self.view.bind_key('w', self.next_image)
        self.view.bind_key('p', self.previous_image)
        
        
        
