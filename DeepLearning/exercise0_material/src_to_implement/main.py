#import the classes
from pattern import Checker
from pattern import Circle
from pattern import Spectrum
from generator import ImageGenerator

import numpy as np
import matplotlib.pyplot as plt

#draw the chessboard
ch = Checker(300, 25)
ch.draw()
ch.show()

#draw the circle

circ = Circle(1024, 200, (256, 256))
circ.draw()
circ.show()

#draw the spectrum

spec = Spectrum(250)
spec.draw()
spec.show()

#visua images generator
label_path = './data/Labels.json'
file_path = './data/exercise_data/'

gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False,
                            shuffle=False)

gen.show("All parameters are False")
gen.next()
gen.show("Next medthod is used")

gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=False,
                            shuffle=True)
gen.show("shuffle is True")
gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=True, mirroring=False,
                            shuffle=True)
gen.show("rotation is True")

gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=False, mirroring=True,
                            shuffle=False)
gen.show("mirroring is True")