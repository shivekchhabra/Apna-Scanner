import cv2
import numpy as np
import argparse
import imutils
from skimage.filters import threshold_local


# argument parser
def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="Path to the image to be scanned")
    args = vars(ap.parse_args())
    return args


# transforming image  (considering it as a rectangle)
def rectangle_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    sum_pts = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum_pts)]
    rect[2] = pts[np.argmax(sum_pts)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect  # top left, top right, bottom right, bottom left (keep the order)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    temp = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, temp, (maxWidth, maxHeight))
    return warped


# detecing edges in the image after processing
def edge_detection(img):
    image = cv2.imread(img)
    ratio = image.shape[0] / 900.0
    orig = image.copy()
    image = imutils.resize(image, height=900)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 75, 200)
    return orig, ratio, canny, image


# identifying image corners.
def find_contours(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # first 4 contours after sorting to get only outer box
    for c in cnts:
        perimeter = cv2.arcLength(c, True)  # true for closed shape
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    return screenCnt


# perspective transform to get a bird eye view
def perspective_transform(img, ratio, cnt):
    warped = rectangle_transform(img, cnt.reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255
    cv2.imshow("Scanned", imutils.resize(warped, height=850))
    cv2.waitKey(0)


if __name__ == '__main__':
    args = argument_parser()
    orig, ratio, edged, resized = edge_detection(args['image'])
    cnts = find_contours(edged)
    perspective_transform(orig, ratio, cnts)
