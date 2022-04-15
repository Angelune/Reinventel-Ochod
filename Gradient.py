import numpy as np 


class Gradient:
    "classe définissant les objets tenseurs"
    
    def __init__(self, n=3):
        self.n=n
        self.tens = np.zeros(n)
    
    def remonteGrad(self):
        self.tens[0] = 8
