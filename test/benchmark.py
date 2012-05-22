import numpy
import Image
import time
from timeit import Timer
from gist import gist

def benchmark(img, orientations, num_times = 10):


    t_npy = Timer("numpy_gist(img, orientations)" , "from  __main__ import numpy_gist, img, orientations")
    print "%.2f usec/pass" % (10  * num_times * t_npy.timeit(number= num_times) / num_times)

def numpy_gist(img, orientations):

    return  gist(img, orientations, boundary_extension = 0)



if __name__ == '__main__':
    img = '../matlab/demo1.jpg'
    orientations = (8,8,8,8)
    benchmark(img, orientations)
