"""
Class implementing segmentation tools.
"""

# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3

import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import napari
import pickle
import ipywidgets as ipw

from . import utils
from .folders import Folders


class Bact:
    
    def __init__(self,
                channels = None,
                corr_threshold = 0.5,
                load_existing = False,
                use_ml = False):

        """Standard __init__ method.
        
        Parameters
        ----------
        nulei : int
            index of DAPI
        GPF : int
            index of GFP
        cherry : int
            index of 600nm channel
        
        
        Attributes
        ----------
            
        all_files = list
            list of files to process
        
        """
        
        #self.folder_name = folder_name
        #self.current_file = None
        self.folders = Folders()
        self.folders.file_list.observe(self.get_filenames, names='options')
        
        self.channels = channels
        self.use_ml = use_ml
        
        self.current_image = None
        self.current_image_med = None
        self.hard_threshold = None
        self.maxproj = None
        self.bact_mask = None
        self.ml = None
        self.corr_threshold = corr_threshold
        self.result = None
        self.nucl_channel = None
        self.bact_channel = None
        self.minsize = 0
        self.fillholes = False

        self.get_filenames(None)
        self.initialize_output()        
        
            
    def get_filenames(self, change):
        """Initialize file list with lsm files present in folder"""
        
        
        self.all_files = [
            os.path.split(x)[1] for x in self.folders.cur_dir.glob("*.oir")]
        if len(self.all_files)>0:
            self.current_file = os.path.split(self.all_files[0])[1]
            
        self.folder_name = self.folders.cur_dir.as_posix()
        self.initialize_output()
      
    def initialize_output(self):
        
        self.nuclei_segmentation = {os.path.split(x)[1]: None for x in self.all_files}
        self.bacteria_segmentation = {os.path.split(x)[1]: None for x in self.all_files}
        self.annotations = {os.path.split(x)[1]: None for x in self.all_files}
        
        self.bacteria_channel_intensities = {os.path.split(x)[1]: None for x in self.all_files}
        

    def import_file(self, filepath):
        
        self.current_file = os.path.split(filepath)[1]
        filetype = os.path.splitext(self.current_file)[1]
        
        if filetype == '.oir':
            image = utils.oir_import(filepath)
        else:
            image = utils.oif_import(filepath,channels=False)
        
        self.current_image = image
    
    
    def calculate_median(self, channel):
        
        ch = self.channels.index(channel)
        self.current_image_med = skimage.filters.median(self.current_image[:,:,ch],
                                        skimage.morphology.disk(2))
        self.current_median_channel = channel
        
    def segment_nuclei(self, channel):
        
        ch = self.channels.index(channel)
        
        nucle_seg = utils.segment_nuclei(self.current_image[:,:,ch], radius=0)
        nucl_mask = nucle_seg>0
        
        self.current_nucl_mask = nucl_mask
        self.nuclei_segmentation[self.current_file] = nucl_mask
        
    def segment_nucleiML(self, channel):
        
        ch = self.channels.index(channel)
        image = self.current_image[:,:,ch]
        im_rescaled = skimage.transform.rescale(image, 0.5)
        filtered, filter_names = utils.filter_sets(im_rescaled)
        
        #classify all pixels and update the segmentation layer
        all_pixels = pd.DataFrame(index=np.arange(len(np.ravel(im_rescaled))))
        for ind, x in enumerate(filter_names):
            all_pixels[x] = np.ravel(filtered[ind])
        predictions = self.ml.predict(all_pixels)

        predicted_image = np.reshape(predictions, im_rescaled.shape)
        if self.minsize >0:
            predicted_image = utils.remove_small(predicted_image, self.minsize)
                
        predicted_image_upscaled = skimage.transform.resize(predicted_image, image.shape,
                                                           order = 0, preserve_range=True)
    
        self.nuclei_segmentation[self.current_file] = predicted_image_upscaled ==2
        
        #plt.imshow(image)
        #plt.show()
        #plt.imshow(self.nuclei_segmentation[self.current_file], cmap = 'gray')
        #plt.show()
        
    def segment_bacteria(self, channel):
        
        #median filter the image to segment and calculate a threshold on it
        self.calculate_median(channel)
        self.calculate_threshold()
        
        #if nuclei mask does not exist yet, calculate it
        nucl_mask = self.nuclei_segmentation[self.current_file]
        if nucl_mask is None:
            self.segment_nuclei(channel)
        
        #create a bacteria template
        rot_templ = utils.create_template()
        
        #im_test = -skimage.filters.farid(self.current_image_med)
        #maxproj = utils.detect_bacteria(im_test, rot_templ, mask= nucl_mask)
        maxproj = utils.detect_bacteria(self.current_image_med, rot_templ, mask= nucl_mask)
        
        self.maxproj = maxproj
        self.bact_mask = (maxproj>self.corr_threshold)*(self.current_image_med>self.hard_threshold)

        self.bacteria_segmentation[self.current_file] = self.bact_mask
        
    def calculate_threshold(self):
        
        binval, binpos = np.histogram(np.ravel(self.current_image_med),bins = np.arange(0,1000,10))
        hard_threshold = 1.5*binpos[np.argmax(binval)]
        self.hard_threshold = hard_threshold
        
        
    def bact_calc_intensity_channels(self):
        bact_labels = skimage.morphology.label(self.bact_mask)
        
        intensities = {self.channels[x]: skimage.measure.regionprops_table(bact_labels, 
                                                            self.current_image[:,:,x],
                                                           properties=('mean_intensity','label'))['mean_intensity']
                      for x in range(len(self.channels)) if self.channels[x] is not None}
        
        self.bacteria_channel_intensities[self.current_file] = intensities
        
      
    def run_analysis(self, nucl_channel, bact_channel, filepath):

        self.nucl_channel = nucl_channel
        self.bact_channel = bact_channel
        
        self.import_file(filepath)
             
        if self.use_ml:
            if self.ml is None:
                raise NameError('No ML training available')
            else:
                self.segment_nucleiML(nucl_channel)
        else:
            self.segment_nuclei(nucl_channel)
        self.segment_bacteria(bact_channel)
        self.bact_calc_intensity_channels()

        
    def save_segmentation(self):
        if not os.path.isdir(self.folder_name+'/Segmented/'):
            os.makedirs(self.folder_name+'/Segmented/',exist_ok=True)
        file_to_save = self.folder_name+'/Segmented/'+os.path.split(self.folder_name)[-1]+'.pkl'
        with open(file_to_save, 'wb') as f:
            to_export = {'bact_channel':self.bact_channel,
                         'nucl_channel':self.nucl_channel,
                         'minsize': self.minsize,
                         'fillholes': self.fillholes,
                         'nuclei_segmentation':self.nuclei_segmentation,
                         'bacteria_segmentation':self.bacteria_segmentation,
                         'annotations': self.annotations,
                         'bacteria_channel_intensities':self.bacteria_channel_intensities,
                         'channels':self.channels, 'all_files':self.all_files, 'ml': self.ml,
                        'result': self.result}
            pickle.dump(to_export, f)
            
    def load_segmentation(self, b = None):
        
        file_to_load = self.folder_name+'/Segmented/'+os.path.split(self.folder_name)[-1]+'.pkl'
        if not os.path.isfile(file_to_load):
            print('No analysis found')
        else:
            with open(file_to_load, 'rb') as f:
                temp = pickle.load(f)

            for k in temp.keys():
                setattr(self, k, temp[k])
            
            print('Loading Done')

            
    def show_segmentation(self, local_file):
        filepath = self.folder_name+'/'+local_file
        if self.bacteria_segmentation[local_file] is None:
            print('not yet segmented')
        
        else:
            #if local_file != self.current_file:
            self.import_file(filepath)

            viewer = napari.Viewer(ndisplay=2)
            for ind, c in enumerate(self.channels):
                if c is not None:
                    
                    viewer.add_image(self.current_image[:,:,ind],name = c)
            viewer.add_labels(skimage.morphology.label(self.bacteria_segmentation[local_file]),
                             name = 'bactseg')
            viewer.add_labels(skimage.morphology.label(self.nuclei_segmentation[local_file]),
                             name = 'nucleiseg')
            self.viewer = viewer
            self.create_key_bindings()
            
    def create_key_bindings(self):
        
        self.viewer.bind_key('w', self.move_forward_callback)
        
    def move_forward_callback(self, viewer):
        
        current_file_index = self.all_files.index(self.current_file)
        current_file_index = (current_file_index+1)%len(self.all_files)
        
        #local_file = os.path.split(self.all_files[current_file_index])[1]
        local_file = self.all_files[current_file_index]
        
        if self.bacteria_segmentation[local_file] is None:
            print('not yet segmented')
        
        else:
            self.import_file(self.folder_name+'/'+local_file)
            for ind, c in enumerate(self.channels):
                if c is not None:
                    layer_index = [x.name for x in self.viewer.layers].index(c)
                    self.viewer.layers[layer_index].data = self.current_image[:,:,ind]
                    
            self.viewer.layers[-2].data = skimage.morphology.label(self.bacteria_segmentation[local_file])
            self.viewer.layers[-1].data = skimage.morphology.label(self.nuclei_segmentation[local_file])
                          
    
    def interactivity(self):
        
        self.button_show = ipw.Button()
        self.button_show.on_click(self.on_click_show_seg)
        
    def on_click_show_seg(self,b):
        
        self.show_segmentation()
        
    
