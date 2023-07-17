import numpy as np

class BinomialFilter:
    '''Binomial filter for image'''
    def __init__(self, kernel_size):
        assert kernel_size % 2 == 1, "Kernel_size must be odd!"

        self.kernel_size = kernel_size
        # Generate kernel with pascal triangle
        self.kernel = []
        new_kernel = [0] * kernel_size
        for i in range(kernel_size):
            for j in range(i):
                new_kernel[j] = self.kernel[j] + self.kernel[j+1]
            self.kernel = new_kernel
    
    def filter(self, image):
        image = self.__horizontal_filter(image)
        return self.__vertical_filter(image)
    
    def __horizontal_filter(self, image):
        H,W = image.shape[:2]
        radius = self.kernel_size//2

        res = np.zeros(image.shape, dtype=float)

        for i in range(H):
            for j in range(W):
                s = 0
                for kx in range(-radius, radius + 1):
                    if j >= kx and j + kx < W: 
                        s += image[i, j+kx] * self.kernel[kx + radius]
                res[i,j] = s
        return res
    
    def __vertical_filter(self, image):
        H,W = image.shape[:2]
        radius = self.kernel_size//2

        res = np.zeros(image.shape, dtype=float)

        for i in range(H):
            for j in range(W):
                s = 0
                for ky in range(-radius, radius + 1):
                    if i >= ky and i + ky < H: 
                        s += image[i+ky, j] * self.kernel[ky + radius]
                res[i,j] = s
        return res