import Image
import numpy
import scipy.io


from gist import gist


demo_mat = scipy.io.loadmat('demo1.mat')['gist1'].reshape(512)

demo_gist = gist('demo1.jpg', image_size = 256)

assert(numpy.allclose(demo_gist, demo_mat))


