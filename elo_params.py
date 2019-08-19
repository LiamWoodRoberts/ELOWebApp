import numpy as np 
class params:
    '''
    Class Containing all variables needed to execute project
    '''
    def __init__(self):
        '''
        variables:

        label_path - absolute path to labels (with label file name)
        image_path - absolute path to images (with image folder name)
        n_images - the number of images to train on
        batch_size - what batch size to use when training CNN
        epochs - How many passes through the data set when training model

        '''
        # Path to Folder

        # Local Path
        self.folder_path = "./static/data/"

        # Heroku Path
        # self.folder_path = "/app/static/data/"
        
        return
