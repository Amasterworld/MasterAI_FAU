
import numpy as np
import matplotlib.pyplot as plt
class Checker:

    #constructor, default size = 8 and two colors, they are constrast (like black and white)
   

    def __init__(self, resolution, tile_size):
        
        if resolution % (2* tile_size) != 0:
            raise ValueError ("Resolution must be evenly dividable by 2 * tile_size.")
        
        self.resolution = resolution
        self.tile_size = tile_size
        #Create an additional instance variable output that can store the pattern-> create 2D zero matrix with the input resolution
        self.output = np.zeros((resolution, resolution))
        

    #draw the board
    def draw(self):
        
           
        tile_shape = (self.tile_size, self.tile_size)
        even_tile = np.ones(tile_shape)
        odd_tile = np.zeros(tile_shape)
        tiles = np.concatenate([even_tile, odd_tile], axis=1)
        tiles = np.tile(tiles, (self.resolution // 2 * self.tile_size, 1))
        tiles = np.concatenate([tiles, np.flip(tiles, axis=1)], axis=1)
        self.output = np.tile(tiles, (self.resolution // 2 * self.tile_size, 1))
        return self.output.copy()
       
        
        
       

            
    def show(self):
        plt.imshow(self.checker)
        plt.imshow(self.output, cmap='gray')
        plt.show()