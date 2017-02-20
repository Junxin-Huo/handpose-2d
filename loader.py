import numpy as np
import os
import random
import cv2

IMAGE_SIZE = 32
# 0: gray
# 1: probability
DATA_TYPE = 0

def loadDataLabel(dir_name, various=False, shuffle=False):
    assert os.path.isdir(dir_name), "dir_name is not dir"
    dir = os.listdir(dir_name)
    len_dir = len(dir)
    datas = []
    labels = []
    for i in range(len_dir):
        sub_dir = dir[i].split('_')[0]
        label = -1
        if sub_dir == '0':
            label = 0
        elif sub_dir == '1':
            label = 1
        elif sub_dir == '4':
            label = 2
        elif sub_dir == '5':
            label = 3
        elif sub_dir == '7':
            label = 4
        if label == -1:
            continue

        names = os.listdir(dir_name + '/' + dir[i])
        names.sort()
        names_size = len(names)
        for j in range(names_size):
            filename = names[j]
            data = getDataFromPicFile(dir_name + '/' + dir[i] + '/' + filename)

            datas.append(data)
            labels.append(label)
            if various:
                img2_m = middle(data)
                datas.append(img2_m)
                labels.append(label)

    if shuffle:
        print("Shuffling...")
        index = range(len(labels))
        random.shuffle(index)
        xx = []
        yy = []
        for i in range(len(labels)):
            xx.append(datas[index[i]])
            yy.append(labels[index[i]])
        datas = xx
        labels = yy
    return np.asarray(datas, np.float32), np.asarray(labels, np.float32)

def getDataFromPicFile(filename, zoon=IMAGE_SIZE):
    img = cv2.imread(filename)
    ret = getDataFromPic(img, zoon)
    return ret

def getDataFromPic(img, zoon=IMAGE_SIZE):

    assert len(img.shape) == 3
    imgReshape = reshapeCube(img, zoon)
    if DATA_TYPE == 0:
        img2 = cv2.cvtColor(imgReshape, cv2.COLOR_BGR2GRAY)
        imgdata = np.asarray(img2, np.float32) / 255
    if DATA_TYPE == 1:
        imgdata = handProb(imgReshape)
    
    return grayNormalize(imgdata)

def rotate90(image):
    [M, N] = image.shape
    assert M == N
    ret = np.zeros((M, N), np.float32)
    for i in range(M):
        for j in range(M):
            ret[i][j] = image[M-j-1][i]
    return ret

def middle(image):
    [M, N] = image.shape
    assert M == N
    ret = np.zeros((M, N), np.float32)
    for i in range(M):
        for j in range(M):
            ret[i][j] = image[i][M-j-1]
    return ret

def handProb(image):
    Mean = np.matrix([117.4316, 148.5599], np.float32).T
    C = np.matrix([[97.0946, 24.4700], [24.4700, 141.9966]], np.float32)

    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    Y = img2[..., 0]
    Cr = img2[..., 1]
    Cb = img2[..., 2]
    M, N = np.shape(Cr)
    FaceProbImg = np.zeros((M, N), np.float32)
    CbCr = np.matrix([0, 0], np.float32).T
    for i in range(M):
        for j in range(N):
            CbCr[0, 0] = Cb[i][j]
            CbCr[1, 0] = Cr[i][j]
            temp = (-0.5 * (CbCr - Mean).T * C.I * (CbCr - Mean))
            FaceProbImg[i, j] = np.e ** temp[0, 0]
    return FaceProbImg

def grayNormalize(image):
    img_copy = np.asarray(image, np.float32)
    imgMax = max(img_copy[img_copy != 0])
    imgMin = min(img_copy[img_copy != 0])
    img_copy[img_copy == 0] = imgMin
    img_copy -= imgMin
    img_copy /= imgMax - imgMin
    return img_copy

def reshapeCube(image, zoon=IMAGE_SIZE):
    if len(np.shape(image)) == 2:
        M, N = np.shape(image)
        if M > N:
            row = zoon
            col = N * zoon / M
        else:
            col = zoon
            row = M * zoon / N
        rz = cv2.resize(image, (col, row), interpolation=cv2.INTER_NEAREST)
        M2, N2 = np.shape(rz)

        ret = np.zeros((zoon, zoon), np.float32)
        xstart = int(np.floor(zoon) / 2 - M2 / 2)
        xend = int(xstart + M2)
        ystart = int(np.floor(zoon) / 2 - N2 / 2)
        yend = int(ystart + N2)
        ret[xstart:xend, ystart:yend] = rz

    if len(np.shape(image)) == 3:
        M, N, C = np.shape(image)
        if M > N:
            row = zoon
            col = N * zoon / M
        else:
            col = zoon
            row = M * zoon / N
        rz = cv2.resize(image, (col, row), interpolation=cv2.INTER_NEAREST)
        M2, N2, C = np.shape(rz)

        ret = np.zeros((zoon, zoon, 3), np.float32)
        xstart = int(np.floor(zoon) / 2 - M2 / 2)
        xend = int(xstart + M2)
        ystart = int(np.floor(zoon) / 2 - N2 / 2)
        yend = int(ystart + N2)
        ret[xstart:xend, ystart:yend, ...] = rz

    return np.asarray(ret, np.uint8)

