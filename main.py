import numpy as np
import cv2 as cv
import os
from statistics import mean

left_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_rightear.xml')


def detectionAccuracy(maskDetected, maskCorrect):
    blended = cv.addWeighted(maskDetected, 0.5, maskCorrect, 0.5, 0.0)
    blendedGrey = cv.cvtColor(blended, cv.COLOR_BGR2GRAY)

    intersection = np.sum(blendedGrey == 203)
    union = np.sum(blendedGrey == 128) + np.sum(blendedGrey == 75) + intersection
    accuracy = round(intersection/union, 3)

    if accuracy > 0:
        cv.imshow('blended', blendedGrey)
        cv.waitKey(0)
        print(accuracy)

    return accuracy


def maskCoordinates(image):
    mask = np.array(np.where(np.array(image) == 255))
    firstWhitePixel = mask[:, 0]
    lastWhitePixel = mask[:, -1]
    return firstWhitePixel[0], firstWhitePixel[1], lastWhitePixel[0], lastWhitePixel[1]


def detectEars(testImage, scale, neighbours):
    height, width, channels = testImage.shape
    testImageGray = cv.cvtColor(testImage, cv.COLOR_BGR2GRAY)

    left_ears = left_ear_cascade.detectMultiScale(testImageGray, scale, neighbours)
    right_ears = right_ear_cascade.detectMultiScale(testImageGray, scale, neighbours)

    maskDetected = np.zeros(shape=[height, width, channels], dtype=np.uint8)

    for (x, y, w, h) in left_ears:
        maskDetected = cv.rectangle(maskDetected, (x, y), (x + w, y + h), (0, 255, 0), -1)

    for (x, y, w, h) in right_ears:
        maskDetected = cv.rectangle(maskDetected, (x, y), (x + w, y + h), (0, 255, 0), -1)

    return maskDetected


def runDetector(scaleFactor, nOfNeighbours):
    testImageDir = 'AWEForSegmentation/test/'
    maskImageDir = 'AWEForSegmentation/testannot_rect/'

    testImageFilenames = sorted(os.listdir(testImageDir))
    accuracies = []

    for filename in testImageFilenames:
        testImage = cv.imread(testImageDir + filename)

        try:
            maskDetected = detectEars(testImage, scaleFactor, nOfNeighbours)
            maskCorrect = cv.imread(maskImageDir + filename)
            accuracy = detectionAccuracy(maskDetected, maskCorrect)
            accuracies.append(accuracy)
        except:
            print(f'There was an exception during detecting ears in file {filename}')

    return mean(accuracies)