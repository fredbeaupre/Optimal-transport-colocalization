# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import skimage
from specpy import File
from skimage import io
import os
import glob
import numpy
import tifffile


def load_msr(msrPATH):
    """Loads the msr data from the msr file. The data is in a numpy array.

    :param msrPATH: The path to the msr file

    :returns: A dict with the name of the stack and the corresponding data
    """
    imageObj = File(msrPATH, File.Read)
    outputDict = {}
    for stack in range(imageObj.number_of_stacks()):
        stackObj = imageObj.read(stack)  # stack object
        outputDict[stackObj.name()] = stackObj.data(
        ).squeeze()  # removes shape of 1
    return outputDict


path = os.path.expanduser(
    r"E:\ImageJ analysis\2021-09-02 (12.4) (virus infection times)\aSyn\*.msr")
images = glob.glob(path)
print('There are ', len(images), 'Images in this folder')
# example 'Conf_561':['Conf_561 {15}'],"STED_561":['STED_561 {15}'],,'CompositeSTED':['STED_561 {15}','STED_640 {15}'
mapcomp = {'CompositeSTED': [
    'STED_561 {15}', 'STED_640 {15}', "Conf_488 {15}"]}


for imagei in images:
    imagemsr = load_msr(imagei)
    for key in mapcomp:
        channels = mapcomp[key]
        if len(channels) == 1:
            image1 = imagemsr[channels[0]]
            io.imsave(imagei.split(".msr")[0] + "_{}.tiff".format(key), image1)
        elif len(channels) > 1:

            image1 = [imagemsr[channel] for channel in channels]
            imagecomp = numpy.dstack(image1)
            imagecomp = numpy.moveaxis(imagecomp, 2, 0)
            with tifffile.TiffWriter(imagei.split(".msr")[0] + "_{}.tiff".format(key), bigtiff=False, imagej=True) as tif:
                tif.save(imagecomp, metadata={'mode': 'composite'})
