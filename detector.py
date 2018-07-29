# import the necessary packages
import imutils,cv2, time, os, sys
import numpy as np
import argparse
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

(imgW, imgH) = (128, 128)


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
         # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def pyramid(image, scale=1.5, minSize=(25, 25)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

###############################################################
######################### MAIN ################################
###############################################################

def main(test_img):
    print("[INFO] Loadin Model...")
    model = load_model('pepsi_model')
    sliding_win_a, sliding_win_b = (29, 29)
    stride = 4
    print("[INFO] evaluating image pyramid...")
    testimage = cv2.imread(test_img)
    testimage = testimage.astype("float") / 255.0
    boundbox = []
    for (a, resized) in enumerate(pyramid(testimage)):
        s = (1.5)**a
        clone = resized.copy()
        for (x, y, window) in sliding_window(resized, stride, windowSize=(sliding_win_a, sliding_win_b)):
            if window.shape[0] != sliding_win_a or window.shape[1] != sliding_win_b:
                continue
            batch = window
            batch = img_to_array(batch)
            batch = cv2.resize(batch, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            batch = np.expand_dims(batch, axis=0)
            (nologo, logo) = model.predict(batch)[0]
            label = "logo" if logo > nologo else "Not logo"
            proba = logo if logo > nologo else nologo
            if logo>nologo and proba>0.88 and proba<1:
                cv2.rectangle(clone, (x, y), (x + sliding_win_a, y + sliding_win_b), (0, 255, 0), 2)
                bbox = (x*s, y*s , x*s + sliding_win_a, y*s + sliding_win_b)
                bbox = list(map(int, bbox))
                boundbox.append(bbox)
    pick = non_max_suppression_fast(np.array(boundbox), 0.05)
    "[x] after applying non-maximum, %d bounding boxes" % (len(pick))
    print(pick)
    # loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(testimage, (startX, startY), (endX, endY), (0, 255, 0), 2)

    df = pd.DataFrame(pick)
    df.columns = ['x1', 'y1', 'x2', 'y2']
    df.to_csv(os.path.split(test_img)[1][0:-4]+"_logo_cord.csv")
    cv2.imshow("Image", testimage)
    print('press any key to exit...')
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--testimage", required=True, type=str)
    args = parser.parse_args()
    testimage = os.path.abspath(args.testimage)
#    if not (testimage.endswith('jpg') or testimage.endswith('png') or testimage.endswith('tif')):
#        raise ('Please Input valid image file')

    main(testimage)

