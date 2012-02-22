#!/usr/bin/env python

"""
This module is the re-implementation of GIST for scene recognition
http://people.csail.mit.edu/torralba/code/spatialenvelope/

The module has been re-implemented from the Matlab sources available
from authors, and has been tested on the same dataset from authors.
"""

import os.path
import Image

import scipy
from numpy import *
import pylab
from shogun.Kernel import GaussianKernel, WeightedDegreeStringKernel
from shogun.Distance import EuclidianDistance
from shogun.Features import *
from shogun.Classifier import *


def create_gabor(ori, img_size):
    ''' create_gabor(numberOfOrientationsPerScale, img_siz);
    
    Precomputes filter transfer functions. All computations are done on the
    Fourier domain. 
    
    If you call this function without output arguments it will show the
    tiling of the Fourier domain.
    
    Input
    numberOfOrientationsPerScale = vector that contains the number of
                            orientations at each scale (from HF to BF)
        img_size = imagesize (square images)
    
    output
    gabor = transfer functions for a jet of gabor filters '''

    n_scales = len(ori)
    n_filters = sum(ori)
    param = empty([1, 4])    

    for i_ind in range(n_scales):
        for j_ind in range(ori[i_ind]):
            param = append(param, [[0.35, 0.3 / (power(1.85, i_ind)), 
            16. * power(ori[i_ind], 2) / power(32, 2), 
            (pi / ori[i_ind]) * j_ind]], axis=0)
            
    param = delete(param, 0, 0)    #remove the first empty row
    
    #Frequencies:
    rang = range(-img_size/2, img_size/2)
    f_x, f_y = meshgrid(rang, rang)
    f_r = fft.fftshift(sqrt(power(f_x, 2) + power(f_y, 2)))
    f_t = fft.fftshift(angle(f_x + complex(0, 1) * f_y))

    #Transfer functions:
    gabor = zeros([img_size, img_size, n_filters])
    for i_ind in range(n_filters):
        trans = f_t + param[i_ind, 3]
        trans = trans + 2 * pi * (trans < -pi) - 2 * pi * (trans > pi)
        
        gabor[:, :, i_ind] = exp(-10 * param[i_ind, 0] * power((f_r / img_size 
                             / param[i_ind, 1] - 1), 2) - 2 * param[i_ind, 2] 
                             * pi * power(trans, 2))
    return gabor


def prefilt(in_img, fc_p):
    ''' ima = prefilt(in_img, fc_p);
    
    Input images are double in the range [0, 255];
    
    For color images, normalization is done by dividing by the local
    luminance variance. '''

    win = 5
    s_1 = fc_p / sqrt(log(2))
    
    # Pad images to reduce boundary artifacts
    in_img = log(in_img + 1)
    in_img = symmetric_pad(in_img, [win, win], 'both')       

    sim = shape(in_img)
    num = max(sim[0], sim[1])
    num = num + mod(num, 2)
    in_img = symmetric_pad(in_img, [num-sim[0], num-sim[1]], 'post')
    
    # Filter
    rang = range(-num/2, num/2)
    f_x, f_y = meshgrid(rang, rang)
    g_f = fft.fftshift(exp( - (power(f_x, 2) + power(f_y, 2)) /
         (power(s_1, 2))))
    g_f = tile(g_f, (1, 1))
    
    # Whitening
    out_img = in_img - (fft.ifft2(fft.fft2(in_img) * g_f)).real
    
    # Local contrast normalization
    localstd = tile(sqrt(abs(fft.ifft2(fft.fft2(power(out_img, 2)) * g_f))), 
                   (1, 1)) 

    out_img = out_img / (0.2 + localstd)

    # Crop output to have same size than the input
    out_img = out_img[win:(sim[0] - win), win:(sim[1] - win)]
    return out_img


def gist_gabor(in_img, win, gist):
    ''' Input:
    in_img = input image
    win_num = number of windows (win*win)
    gis = precomputed transfer functions
    
    Output:
    g_feat: are the global features = [Nfeatures Nimages], 
                   Nfeatures = win*win*n_filters*c '''

    
    n_filters = shape(gist)[2]
    win_num = win*win
    g_feat = zeros([win_num*n_filters, 1])
    
    in_img = fft.fft2(in_img)
    
    k_index = 0
    
    for num in range(1):
        i_g = abs(fft.ifft2(in_img * tile(gist[:, :, num], (1, 1)))) 
        avg = down_n(i_g, win)
        g_feat[k_index:k_index + win_num, :] = reshape(avg.T, [win_num, 1])
        k_index = k_index + win_num
    return g_feat


def down_n(in_img, num):
    ''' averaging over non-overlapping square image blocks
    
    Input
    input = [nrows ncols nchanels]
    Output
    out = [num num nchanels] '''

    n_x = fix(linspace(0, shape(in_img)[0], num + 1))
    n_y = fix(linspace(0, shape(in_img)[1], num + 1))
    out = zeros([num, num, 1])
    for x_x in range(num):
        for y_y in range(num):
            avg = mean(mean(in_img[n_x[x_x]:n_x[x_x + 1], n_y[y_y]:n_y[y_y + 1]]
                  , 0))
            out[x_x, y_y, :] = avg
    return out


def symmetric_pad(arr, pad_size, direction):
    ''' Pads array 'arr' using symmetric method
    Implemented from Matlab padarray function '''

    num_dims = len(pad_size)

    # Form index vectors to subsasgn input array into output array.
    # Also compute the size of the output array.
    idx = []
    if len(shape(arr)) == 1:
        size_arr = (1, len(arr))
    else:
        size_arr = shape(arr)

    for k_indx in range(num_dims):
        tot = size_arr[k_indx]
        dim_nums = array(range(1, tot + 1))
        dim_nums = append(dim_nums, range(tot, 0, -1))
        pad = pad_size[k_indx]
        
        if direction == 'pre':
            idx.append([dim_nums[mod(range(-pad, tot), 2 * tot)]])
        elif direction == 'post':
            idx.append([dim_nums[mod(range(tot + pad), 2 * tot)]])           
        elif direction == 'both':
            idx.append([dim_nums[mod(range(-pad, tot + pad), 2 * tot)]])

    first = idx[0][0]-1
    second = idx[1][0]-1
    return arr[ix_(first, second)]

        
def lsit_of_images(path):
    ''' Loads list of images in path '''

    images = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.jpg'):
            images.append(path + filename)
    return images


def find_category(image_name, categories):
    ''' Find the category index from image names '''

    for item in categories:
        if image_name.find(item) > 0:
            return int(categories.index(item))


if __name__ == '__main__':
    #Main procedure for scene recognition task

    # Parameteres
    HOMEIMAGES = 'dataset/spatial_envelope_256x256_static_8outdoorcategories/'
    CATEGORIES = ['tallbuilding', 'insidecity', 'street', 'highway', 'coast',
                  'opencountry', 'mountain', 'forest']
    IMG_SIZE = 256
    ORIENTATION_PER_SCALE = [8, 8, 8, 8]
    NUMBERBLOCKS = 4
    FC_PREFILT = 4
    N_TRAINING_PER_CLASS = 100

    N_CLASSES = len(CATEGORIES)

    #
    # Compute global features
    #
    SCENES = lsit_of_images(HOMEIMAGES)
    N_SCENES = len(SCENES)
    GABOR = create_gabor(ORIENTATION_PER_SCALE, IMG_SIZE)
    N_FEATURES = GABOR.shape[2] * power(NUMBERBLOCKS, 2)

    # Loop: Compute global features for all scenes
    FEATURES = zeros([N_SCENES, N_FEATURES])
    CLASSES = zeros([N_SCENES, 1])

    for n in range(N_SCENES):
        print n, N_SCENES
        img = Image.open(SCENES[n])
        img = asarray(img)
        img = mean(img, 2)

        sh = shape(img)
        if sh[0] != IMG_SIZE:
            img = scipy.misc.imresize(img, [IMG_SIZE, IMG_SIZE], 
                  interp = 'bilinear')

        output = prefilt(img, FC_PREFILT)
        GIST = gist_gabor(output, NUMBERBLOCKS, GABOR)
        FEATURES[n] = transpose(GIST)
        CLASSES[n] = find_category(SCENES[n], CATEGORIES)
        

    # 
    # Split training/test (train = index training samples, test = index test)
    #
    TRAIN = array((1), int)
    for c in range(N_CLASSES):
        j = where(CLASSES == c)
        t = random.permutation(range(len(j[0])))
        TRAIN = append(TRAIN, j[0][t[0:N_TRAINING_PER_CLASS]])

    TEST = transpose(setdiff1d(range(N_SCENES), TRAIN))

    #
    # Train and test classifier
    #
    SCORES = zeros((N_CLASSES, len(TEST)))
    WIDTH = 0.003
    PARAM_C = 20.0
    #epsilon = 1e-5
    #num_threads = 2
    for c in range(N_CLASSES):
        feats_train = RealFeatures(FEATURES[TRAIN, :].conj().T)
        feats_test = RealFeatures(FEATURES[TEST, :].conj().T)
        labels = Labels(array((2*(CLASSES[TRAIN] == c) - 1)[:,
                 0]).astype(float))
        kernel = GaussianKernel(feats_train, feats_train, WIDTH)
        svm = LibSVM(PARAM_C, kernel, labels)
        #svm.set_epsilon(epsilon)
        #svm.parallel.set_num_threads(num_threads)
        svm.train()
        kernel.init(feats_train, feats_test)
        SCORES[c, :] = svm.classify().get_labels()

    CTEST_HAT = zeros([len(TEST), 1], int)
    for k in range(len(TEST)):
        CTEST_HAT[k] = argmax(SCORES[:, k])

    #
    # Plot performance and Confusion matrix
    #
    C_MAT = zeros((N_CLASSES, N_CLASSES))
    for j in range(N_CLASSES):
        for i in range(N_CLASSES):
            # row i, col j is the percentage of images from class i that
            # were missclassified as class j.
            C_MAT[i, j] = 100 * sum((CLASSES[TEST] == i) * 
                         (CTEST_HAT == j)) / sum(CLASSES[TEST] == i)


    pylab.subplot(121, aspect='equal')
    pylab.pcolor(C_MAT)
    GCA = pylab.gca()
    GCA.set_ylim([8, 0])
    pylab.colorbar()
    pylab.subplot(122)
    pylab.bar(range(8), diag(C_MAT), 0.35)
    pylab.title(mean(diag(C_MAT)))
    pylab.show()
(N_CLASSES):
            # row i, col j is the percentage of images from class i that
            # were missclassified as class j.
            C_MAT[i, j] = 100 * sum((CLASSES[TEST] == i) * 
                         (CTEST_HAT == j)) / sum(CLASSES[TEST] == i)


    pylab.subplot(121, aspect='equal')
    pylab.pcolor(C_MAT)
    GCA = pylab.gca()
    GCA.set_ylim([8, 0])
    pylab.colorbar()
    pylab.subplot(122)
 