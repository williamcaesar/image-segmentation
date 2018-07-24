#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from cv2 import cv2
from dither import basic as basic_dither


def segment(image, threshold):
    """Divide pixels by a threshold."""
    pixels_less = []
    pixels_greater = []
    if len(image.shape) == 2:
        for row in range(1, image.shape[0]):
            for column in range(1, image.shape[1]):
                if image[row, column] <= threshold:
                    pixels_less.append(image[row, column])
                else:
                    pixels_greater.append(image[row, column])
    return (pixels_less, pixels_greater)


def mean(elements):
    """Calculate the mean of N elements."""
    if type(elements) != list or len(elements) == 0:
        return 0

    result = 0
    for item in elements:
        result += item
    return result/len(elements)


def iterative_sgmnt(image, limits=[0, 255], variation=0.1):
    """Build a segmented image."""
    threshold = numpy.average(image)
    print('*'*40)
    print('threshold: ', threshold, '\n')
    print('*'*40)
    while (True):
        (pixels_less, pixels_greater) = segment(image, threshold)
        med = (mean(pixels_less), mean(pixels_greater))
        print('mean minor pixels: ', med[0])
        print('mean greater pixels: ', med[1])
        if abs(threshold - (med[0]+med[1])/2) <= variation:
            break
        threshold = (med[0] + med[1]) / 2
        print('\tthreshold:', threshold, '\n')
    print('\tthreshold:', threshold, '\n')

    return basic_dither(image, limits, threshold)


def show(image):
    """Show a image untill ENTER key is pressed."""
    cv2.imshow('press ENTER to close', image)
    cv2.waitKey(0)


if (__name__ == '__main__'):
    image = cv2.imread('images/world.png', cv2.IMREAD_GRAYSCALE)
    show(iterative_sgmnt(image))
