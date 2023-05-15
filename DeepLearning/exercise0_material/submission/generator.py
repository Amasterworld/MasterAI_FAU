import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import io, transform

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.        

        self.path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        
        self.index = 0
        self.num_images = len(os.listdir(self.path))    
        
        self.epoch_cnt = 0
        self.full_labels = []

        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
        #check the size of data-set
        image_size_bytes = image_size[0] * image_size[1] * 3 #  RGB images
        dataset_size_mb = self.num_images * image_size_bytes / 1024**2 # 1
        if dataset_size_mb <= 500: # If dataset is smaller than 500MB
            self.images = np.zeros((self.num_images, image_size[0], image_size[1], 3), dtype=np.float32)
            for i in range (self.num_images):
                img = self.__load_image(str(i) + '.npy')
                img = transform.resize(img, self.image_size)               
                self.images[i] = img                
                self.full_labels.append(self.labels[str(i)])       
                
        
        
        
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        self.indices = list(range(self.num_images))
        if self.shuffle:
            random.shuffle(self.indices)
    #transform the image to image_size    
    def __resize_image(self, image):
        return transform.resize(image, self.image_size)
        
    def __load_image(self, filename):
        filepath = os.path.join(self.path, filename)
        return np.load(filepath)
    
    def current_epoch(self):
        return self.epoch_cnt
    
    def class_name(self, label):
        return self.class_dict[label]

    def next(self):
        #check whether the current index >= the number of images, if yes, reset index = 0 and re-check shuffle, 
        #increase the number of epoch        
        
        if self.index >= self.num_images:
            self.index = 0
            self.epoch_cnt += 1
            if self.shuffle:
                random.shuffle(self.indices)
       
        batch_indices = self.indices[self.index : self.index + self.batch_size]
        
        images = []
        labels = []
        
        for i in batch_indices:
            image = self.__load_image(str(i) + '.npy')
            #if rotatiion is True, then np.rot90() method rotate the image anti-clockwise 90 degree by k-times
            if self.rotation:
                k = random.choice([1, 2, 3])
                image = np.rot90(image, k=k)

            if self.mirroring:
                #if the mirror == True then randomly flip images
                flip = random.choice([True, False])
                if flip:
                    image = np.fliplr(image)
            image = self.__resize_image(image)
            
            images.append(image)
            labels.append(self.labels[str(i)])
            
        
        self.index += self.batch_size
        
        # Handle last batch with fewer images
        if len(images) < self.batch_size:
            
            num_missing = self.batch_size - len(images)
            missing_indices = self.indices[:num_missing]
            #update self.indices, remove missing_indices hat were used in the current batch and add them to the end of the list
            
            self.indices = self.indices[num_missing:] + missing_indices
            images.extend(self.images[0:num_missing]) 
            labels.extend(self.full_labels[:num_missing])
            
        return np.array(images), np.array(labels)
    
    def show(self, title = None):
        images, labels = self.next()
        num_image_in_row = 3
        rows = int(np.ceil(self.batch_size / num_image_in_row)) # for example if batch_size = 11, then we still want to have 4 rows, so ceil(), round up
        # from the example in pdf file: I want to create a grid of subplot with: rows and num_image_in_row per row
        #to do it: I use subplot
        fig, ax = plt.subplots(rows, num_image_in_row, figsize=(6, 6))
        for i in range(rows):
            for j in range(3):
                if i*num_image_in_row+j < self.batch_size:
                    ax[i, j].imshow(images[i*num_image_in_row+j])
                    ax[i, j].set_title(self.class_name(labels[i*num_image_in_row+j]))
                ax[i, j].axis("off")
            fig.subplots_adjust(hspace=0.5) #set the spacing between subplotsä
        if title is not None:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.show()
            










