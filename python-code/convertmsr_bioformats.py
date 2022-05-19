import os
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import numpy as np
import cv2 as cv
try:
    import javabridge
    import bioformats
except ImportError:
    print("Bioformats does not seem to be installed on your machine...")
    print("Try running `pip install python-bioformats`")
    exit()


PATH = './jmdata_ot/bassoon_psd/318'
CONDITION = '318'


def crop_260(image, new_h, new_w):
    w, h = image.size
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    im = image.crop((left, top, right, bottom))
    return im


class MSRReader:
    """
    Creates a `MSRReader`. It will take some time to create the object
    :param logging_level: A `str` of the logging level to use {WARN, ERROR, OFF}
    :usage :
        with MSRReader() as msrreader:
            data = msrreader.read(file)
            image = data["STED_640"]
    """

    def __init__(self, logging_level="OFF"):

        # Starts the java virtual machine
        javabridge.start_vm(class_path=bioformats.JARS)

        rootLoggerName = javabridge.get_static_field(
            "org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
        rootLogger = javabridge.static_call(
            "org/slf4j/LoggerFactory", "getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
        logLevel = javabridge.get_static_field(
            "ch/qos/logback/classic/Level", logging_level, "Lch/qos/logback/classic/Level;")
        javabridge.call(rootLogger, "setLevel",
                        "(Lch/qos/logback/classic/Level;)V", logLevel)

    def read(self, msrfile):
        """
        Method that implements a `read` of the given `msrfile`
        :param msrfile: A file path to a `.msr` file
        :returns : A `dict` where each keys coresponds to a specific image in the
                   measurement file
        """
        data = {}
        with bioformats.ImageReader(path=os.path.join(PATH, msrfile)) as reader:
            metadata = bioformats.OMEXML(
                bioformats.get_omexml_metadata(path=reader.path))
            rdr = reader.rdr
            count = rdr.getSeriesCount()
            for c in range(count):
                image = reader.read(series=c, rescale=False)
                image_metadata = metadata.image(c)
                data[image_metadata.get_Name()] = image
        return data

    def close(self):
        javabridge.kill_vm()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()


def main():
    mapcomp = [
        'STED_561 {15}', 'STED_640 {15}'
    ]
    msrfiles = os.listdir(PATH)
    with MSRReader() as msrreader:
        for msrfile in msrfiles:
            tiff_image = []
            print(os.path.join(PATH, msrfile))
            data = msrreader.read(msrfile)
            for key in data.keys():
                if key in mapcomp:
                    img = data[key].astype(np.uint16)
                    tiff_image.append(img)
            tiff_image = np.dstack(tiff_image)
            tiff_image = np.moveaxis(tiff_image, 2, 0)
            with tifffile.TiffWriter('./jmdata_ot/composite/bassoon_psd/{}/{}.tif'.format(CONDITION, msrfile), bigtiff=False, imagej=True) as tif:
                tif.save(tiff_image, metadata={'mode': 'composite'})


if __name__ == "__main__":
    main()
    exit()
    PATH = "."
    msrfiles = os.listdir(IN_DIR)
    with MSRReader() as msrreader:
        for msrfile in msrfiles:
            data = msrreader.read(os.path.join(IN_DIR, msrfile))
            print(data.keys())
            exit()
            # save conf561
            if "Conf_561 {15}" in data:
                conf = data["Conf_561 {15}"].astype(np.uint16)
                norm_conf = cv.normalize(
                    conf, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_16U)
                np.save(
                    './ConfSTED/conf/conf_{}.npy'.format(conf_index), norm_conf)
                conf_index += 1
