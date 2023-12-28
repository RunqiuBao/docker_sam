import csv
import json
import numpy
import os
import glob
import shutil
from PIL import Image
import cv2

import datasets

_DESCRIPTION = "This dataset is for object instance segmentation task. Generated with Mujin metal workpieces detection resources."
_HOMEPAGE = ""
_LICENSE = ""
_CITATION = ""


def LoadDataIndices(filepath):
    with open(filepath, 'r') as file:
        sDataIndices = file.readlines()
    dataIndices = []
    for datumIndex in sDataIndices:
        if datumIndex[-1] != '\n':
            dataIndices.append(int(datumIndex))
        else:
            dataIndices.append(int(datumIndex[:-1]))
    return dataIndices

    
def WriteDatumInfo(dataCount, filePath):
    if dataCount == 0:
        with open(filePath, 'w') as f:
            f.write(str(dataCount))
    else:
        with open(filePath, 'a') as f:
            f.write('\n' + str(dataCount))


def SplitDataset(dataroot):
    # create train and valid folders
    dirsToMake = [
        "train/images",
        "train/masks",
        "valid/images",
        "valid/masks"
    ]
    for dirToMake in dirsToMake:
        os.makedirs(os.path.join(dataroot, dirToMake), exist_ok=False)

    dataList = glob.glob(os.path.join(dataroot, 'all', 'images', '*.png'))
    dataList.sort()
#    from IPython import embed; embed()
    countTrainSet = 0
    countValidSet = 0
    trainSetInfoPath = os.path.join(dataroot, 'train', 'dataList.txt')
    validSetInfoPath = os.path.join(dataroot, 'valid', 'dataList.txt')
    for datumIndex, datumName in enumerate(dataList):
        datumName = os.path.basename(datumName)
        if datumIndex % 2 == 0:
            shutil.copyfile(
                os.path.join(dataroot, 'all', 'images', datumName),
                os.path.join(dataroot, 'train', 'images', str(countTrainSet).zfill(6) + '.png')
            )
            shutil.copyfile(
                os.path.join(dataroot, 'all', 'masks', datumName),
                os.path.join(dataroot, 'train', 'masks', str(countTrainSet).zfill(6) + '.png')
            )
            WriteDatumInfo(countTrainSet, trainSetInfoPath)
            countTrainSet += 1
        elif datumIndex % 2 != 0:
            shutil.copyfile(
                os.path.join(dataroot, 'all', 'images', datumName),
                os.path.join(dataroot, 'valid', 'images', str(countValidSet).zfill(6) + '.png')
            )
            shutil.copyfile(
                os.path.join(dataroot, 'all', 'masks', datumName),
                os.path.join(dataroot, 'valid', 'masks', str(countValidSet).zfill(6) + '.png')
            )
            WriteDatumInfo(countValidSet, validSetInfoPath)
            countValidSet += 1

    print("Split dataset into train ({}), valid ({}).".format(countTrainSet, countValidSet))


class FtSam(datasets.GeneratorBasedBuilder):
    """fine tuning Segment Anything dataset"""
    _dataroot = './'

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.Image(),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE
        )

    def _split_generators(self, dl_manager):
        if not os.path.exists(os.path.join(self._dataroot, "train")):
            SplitDataset(self._dataroot)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(self._dataroot, "train"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(self._dataroot, "valid"),
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        dataIndices = LoadDataIndices(os.path.join(filepath, 'dataList.txt'))
        imageList = []
        maskList = []
        for datumIndex in dataIndices:
            imageList.append(Image.open(os.path.join(filepath, "images", str(datumIndex).zfill(6) + ".png")))
            maskList.append(Image.open(os.path.join(filepath, "masks", str(datumIndex).zfill(6) + ".png")))
        yield "image", imageList
        yield "label", maskList



class FtSamDataset(object):
    """fine tuning Segment Anything dataset"""
    _data = None
    _num_rows = None
    _imageRows = 300
    _imageCols = 300
    
    def __init__(self, dataroot, split):
        if not os.path.exists(os.path.join(dataroot, "train")):
            SplitDataset(dataroot)
        if split == 'train':
            filepath = os.path.join(dataroot, "train")
        elif split == 'valid':
            filepath = os.path.join(dataroot, "valid")
        dataIndices = LoadDataIndices(os.path.join(filepath, 'dataList.txt'))
        imageList = []
        maskList = []
        for datumIndex in dataIndices:
            # resizing and padding images to 300 x 300 fixed size.
            oneImage = Image.open(os.path.join(filepath, "images", str(datumIndex).zfill(6) + ".png"))
            oneMask = Image.open(os.path.join(filepath, "masks", str(datumIndex).zfill(6) + ".png"))
            oneMask = Image.fromarray(numpy.array(oneMask) / 255)
#            from IPython import embed; embed()
            imageList.append(self._ResizeImage(oneImage))
            maskList.append(self._ResizeImage(oneMask))
        self._data = {
            "image": imageList,
            "label": maskList
        }
        self._num_rows = len(imageList)

    def _ResizeImage(self, pilImage):
        img = numpy.array(pilImage)
        height, width = img.shape[:2]
        if height > width:
            scale = self._imageRows / height
            newHeight, newWidth = self._imageRows, int(round(width * scale))
        else:
            scale = self._imageCols / width
            newHeight, newWidth = int(round(height * scale)), self._imageCols
        newImg = cv2.resize(img, (newWidth, newHeight))
        # padding short dim
        if height > width:
            newImg = numpy.pad(
                newImg,
                (
                    (0, 0),
                    (0, self._imageCols - newWidth),
                ),
            )
        else:
            newImg = numpy.pad(
                newImg,
                (
                    (0, self._imageRows - newHeight),
                    (0, 0),
                ),
            )
        return Image.fromarray(newImg)

    def __getitem__(self, key):
        if isinstance(key, int):
            return {
                "image": self._data["image"][key],
                "label": self._data["label"][key]
            }
        elif isinstance(key, str):
            return self._data[key]

    @property
    def num_rows(self):
        return self._num_rows

    def __len__(self):
        return self._num_rows
