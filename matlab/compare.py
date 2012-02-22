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



def compute_feature(orientations_per_scale, number_blocks, fc_prefilt):
    ''' Computes the feaure of a list of images based on the input
    parametres '''

    # Parameteres
    home_images = 'dataset/tmp/'
    image_size = 256

    #
    # Compute global features
    #
    scenes = lsit_of_images(home_images)
    n_scenes = len(scenes)
    gabor = create_gabor(orientations_per_scale, image_size)
    n_features = gabor.shape[2] * power(number_blocks, 2)

    # Loop: Compute global fetures for all scenes
    feat_list = zeros([n_scenes, n_features])

    for sc_n in range(n_scenes):
        img = Image.open(scenes[sc_n])
        img = asarray(img)
        img = mean(img, 2)
        if shape(img)[0] != image_size:
            img = scipy.misc.imresize(img, [image_size, image_size], 
                  interp = 'bilinear')

        output = prefilt(img, fc_prefilt)
        gist = gist_gabor(output, number_blocks, gabor)
        feat_list[sc_n] = transpose(gist)

    return feat_list
      
if __name__ == '__main__':    
    # Load parametes:
    F_PARAM = open('data/param.txt')
    PARAM = F_PARAM.readlines()
    F_PARAM.close()
    TEST_N = int(float(PARAM[0].rstrip('\n')))

    # For each set of paramaetes
    for test_idx in range(TEST_N):
        number_blocks_list = int(float(PARAM[1].rstrip('\n').split('   ')\
                        [test_idx + 1]))
        fc_prefilt_list = int(float(PARAM[2].rstrip('\n').split('   ')\
                     [test_idx + 1]))
        orientations_per_scale_list = []
        for indx in range(TEST_N):
            orientations_per_scale_list.append(int(float(
                  PARAM[3 + test_idx].rstrip('\n').split('   ')[indx + 1])))

        # compute the feature with current parameteres
        feature = compute_feature(orientations_per_scale_list, 
                  number_blocks_list, fc_prefilt_list)

        #Compare it with matlab result
        matlab_feature = loadtxt('data/feat' + str(test_idx + 1) + '.txt')

        print 'Test ' + str(test_idx) + ':'
        for img_idx in range(len(feature)):
            print 'Image ' + str(img_idx + 1) + ' :' + \
                  str(linalg.norm(matlab_feature - feature))


        

))

        # compute the feature with current parameteres
        feature = compute_feature(orientations_per_scale_list, 
                  number_blocks_list, fc_prefilt_list)

        #Compare it with matlab result
        matlab_feature = loadtxt('data/feat' + str(test_idx +