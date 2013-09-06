#!/usr/bin/env python

"""
This module is the re-implementation of GIST for scene recognition
http://people.csail.mit.edu/torralba/code/spatialenvelope/

The module has been re-implemented from the Matlab sources available
from authors, and has been tested on the same dataset from authors.
"""

import os
import numpy, scipy
from PIL import Image, ImageChops

def create_gabor(ori, img_size):
    ''' create_gabor(numberOfOrientationsPerScale, img_siz);

    Precomputes filter transfer functions. All computations are done on the
    Fourier domain.

    If you call this function without output arguments it will show the
    tiling of the Fourier domain.

    Input
    numberOfOrientationsPerScale = vector that contains the number of
                            orientations at each scale (from HF to BF)

    output
    gabor = transfer functions for a jet of gabor filters '''

    n_scales = len(ori)
    n_filters = sum(ori)
    param = numpy.empty([1, 4])

    for i_ind in range(n_scales):
        for j_ind in range(ori[i_ind]):
            param = numpy.append(param, [[0.35, 0.3 / (numpy.power(1.85, i_ind)),
            16. * numpy.power(ori[i_ind], 2) / numpy.power(32, 2),
            (numpy.pi / ori[i_ind]) * j_ind]], axis=0)

    param = numpy.delete(param, 0, 0)    #remove the first empty row

    #Frequencies
    rng_x = range(-img_size[1]/2, img_size[1]/2)
    rng_y = range(-img_size[0]/2, img_size[0]/2)
    f_x, f_y = numpy.meshgrid(rng_x, rng_y)
    f_r = numpy.fft.fftshift(numpy.sqrt(numpy.power(f_x, 2) + numpy.power(f_y, 2)))
    f_t = numpy.fft.fftshift(numpy.angle(f_x + complex(0, 1) * f_y))

    #Transfer functions:
    gabor = numpy.zeros([img_size[0], img_size[1], n_filters])
    for i_ind in range(n_filters):
        trans = f_t + param[i_ind, 3]
        trans = trans + 2 * numpy.pi * (trans < -numpy.pi) - 2 * numpy.pi * (trans > numpy.pi)


        a= -10 * param[i_ind, 0]
        b = numpy.power((f_r / img_size[1] / param[i_ind, 1] -1), 2)
        c = - 2 * param[i_ind, 2]
        d = numpy.pi * numpy.power(trans, 2)
        e =c * b
        f = c * d
        #import ipdb
        #ipdb.set_trace()
        gabor[:, :, i_ind] = numpy.exp(-10 * param[i_ind, 0] * numpy.power((f_r / img_size[1]
                             / param[i_ind, 1] - 1), 2) - 2 * param[i_ind, 2]
                             * numpy.pi * numpy.power(trans, 2))
    return gabor


def prefilt(img, fc = 4):
    ''' ima = prefilt(in_img, fc_p);

    Input images are double in the range [0, 255];

    For color images, normalization is done by dividing by the local
    luminance variance. '''

    win = 5
    s1 = fc / numpy.sqrt(numpy.log(2.))

    # Pad images to reduce boundary artifacts
    img = numpy.log(img + 1.)
    img = _symmetric_pad(img, [win, win])

    sim = numpy.shape(img)
    num = max(sim[0], sim[1])
    num = num + numpy.mod(num, 2)
    img = _symmetric_pad(img, [num-sim[0], num-sim[1]], 'post')


    # Filter
    rng = numpy.arange(-num/2., num/2.)
    fx, fy = numpy.meshgrid(rng, rng)
    gf = numpy.fft.fftshift(numpy.exp(-(fx ** 2. + fy ** 2.) /(s1 ** 2.)))
    # for RGB image
    if img.ndim  == 3:
        gf = numpy.tile(gf.reshape((sim[0], sim[1], 1)), (1, 1, sim[2]))
    # Whitening
    out = img - (numpy.fft.ifft2(numpy.fft.fft2(img) * gf)).real

    # Local contrast normalization
    localstd = numpy.sqrt(numpy.abs(numpy.fft.ifft2(numpy.fft.fft2((out ** 2.)) * gf)))

    out = out / (.2 + localstd)

    # Crop output to have same size than the input
    out = out[win:(sim[0] - win), win:(sim[1] - win)]

    return out


def gist_gabor(img, gist, win, be):
    ''' Input:
    in_img = input image
    win_num = number of windows (   win*win)
    gis = precomputed transfer functions

    Output:
    g_feat: are the global features = [Nfeatures Nimages],
                   Nfeatures = win*win*n_filters*c '''

    if img.ndim == 2:
        c_ = 1
        N_ = 1
        img = img.reshape(img.shape[0], img.shape[1])
    elif img.ndim == 3:
        c_ = img.shape[2]
        N_ = c_
    elif img.ndim == 4:
        nrow, ncol, c_, N_ = img.shape
        img = img.reshape((nrow, ncol, c_ * N_))
        N_ = c_ * N_

    nx, ny, n_filters = gist.shape
    win_num = win*win
    g_feat = numpy.zeros((win_num*n_filters, N_))

    img = _symmetric_pad(img, (be, be))
    img = numpy.fft.fft2(img)

    k_index = 0
    for n in range(n_filters):
        if N_ == 1:
            gist_ = gist[:, :, n]
        else:
            gist_ = numpy.tile(gist[:, :, n].reshape(gist.shape[0], gist.shape[0], 1), (1, 1, N_))

        ig = numpy.abs(numpy.fft.ifft2(img * gist_))
        ig = ig[be : ny - be, be: nx -be]
        v = down_n(ig, win)

        g_feat[k_index:k_index + win_num, :] = numpy.reshape(v.T, [win_num, N_])
        k_index = k_index + win_num


    return g_feat


def down_n(x, num):
    ''' averaging over non-overlapping square image blocks

    Input
    input = [nrows ncols nchanels]
    Output
    out = [num num nchanels] '''

    nx = numpy.fix(numpy.linspace(0, x.shape[0], num + 1))
    ny = numpy.fix(numpy.linspace(0, x.shape[1], num + 1))
    if x.ndim == 2:
        out = numpy.zeros([num, num])
    else:
        out = numpy.zeros([num, num, x.shape[2]])

    for xx in range(num):
        for yy in range(num):
            if x.ndim == 2:
                avg = numpy.mean(numpy.mean(x[nx[xx]:nx[xx + 1], ny[yy]:ny[yy + 1]]), 0)
                out[xx, yy] = avg.flatten()
            else:
                avg = numpy.mean(numpy.mean(x[nx[xx]:nx[xx + 1], ny[yy]:ny[yy + 1],:], 0), 0)
                out[xx, yy, :] = avg.flatten()
    return out


def _symmetric_pad(arr, pad_size, direction = 'both'):
    ''' Pads array 'arr' using symmetric method
    Implemented from Matlab padarray function '''

    num_dims = len(pad_size)

    # Form index vectors to subsasgn input array into output array.
    # Also compute the size of the output array.
    idx = []
    if len(numpy.shape(arr)) == 1:
        size_arr = (1, len(arr))
    else:
        size_arr = numpy.shape(arr)

    for k_indx in range(num_dims):
        tot = size_arr[k_indx]
        dim_nums = numpy.array(range(1, tot + 1))
        dim_nums = numpy.append(dim_nums, range(tot, 0, -1))
        pad = pad_size[k_indx]

        if direction == 'pre':
            idx.append([dim_nums[numpy.mod(range(-pad, tot), 2 * tot)]])
        elif direction == 'post':
            idx.append([dim_nums[numpy.mod(range(tot + pad), 2 * tot)]])
        elif direction == 'both':
            idx.append([dim_nums[numpy.mod(range(-pad, tot + pad), 2 * tot)]])

    first = idx[0][0]-1
    second = idx[1][0]-1
    return arr[numpy.ix_(first, second)]


def _im_resize_crop(img, size, method = 'bilinear'):
    """
    resize and crop an image
    """

    scaling = max(float(size[0]) / img.shape[0], float(size[1]) / img.shape[1])

    new_size = numpy.round((img.shape[0] * scaling, img.shape[1] * scaling)).astype(int)
    # TODO imresize just work with integers and we loose some perscion here
    img = scipy.misc.imresize(img, new_size, method, mode = 'F')
    sr = numpy.floor((img.shape[0] - size[0]) / 2.)
    sc = numpy.floor((img.shape[1] - size[1]) / 2.)

    img = img[sr : sr + size[0], sc : sc + size[1]]
    return img


def gist(image_path, orientations = (8,8,8,8), num_blocks = 4, fc_prefilt = 4,
            boundary_extension = 32, image_size = None ):
    """
    Compute gist representation of image.
    Both RGB and gray scale images are accepted.
    """

    img = Image.open(image_path)
    img = numpy.asarray(img, dtype = float)
    img = img.mean(axis = 2)

    if image_size == None:
        image_size = numpy.asarray(img.shape)

    if numpy.ndim(image_size) == 0:
        image_size = numpy.asarray((image_size, image_size))

    # prepare image
    img = _im_resize_crop(img, image_size, 'bilinear')
    img = img - img.min()
    img =   255. * img / img.max()

    gabor = create_gabor(orientations, image_size + 2 * boundary_extension)
    output = prefilt(img, fc_prefilt)
    gist = gist_gabor(output, gabor, num_blocks, boundary_extension)

    return gist.flatten()



def trim(im, border):
    """
    https://gist.github.com/mattjmorrison/932345
    """
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def test(img_path):

    image_size = (256, 256)
    number_blocks = 4
    orientations = (8,8,8, 8)
    fc_prefilt = 4
    boundary_extension = 32

    img = Image.open(img_path)
    img = trim(img, 256)
    img.save('_temp.jpg')

    print gist('_temp.jpg')


if __name__ == '__main__':

    test("sea.jpg")
