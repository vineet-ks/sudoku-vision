import cv2
import numpy as np


def maxContour(contours):
    """Returns the contour with maximum area, if any"""
    max_area = 0
    max_cnt = None
    for cnt_elt in contours:
        area = cv2.contourArea(cnt_elt)
        if area > 100000:
            if area > max_area:
                max_area = area
                max_cnt = cnt_elt
    return max_cnt

def getCorners(max_cnt):
    """Returns the edges of the contour if the approx polygon is a quadrilateral"""
    perimeter = cv2.arcLength(max_cnt,True)
    epsilon = 0.1 * perimeter
    approx = cv2.approxPolyDP(max_cnt, epsilon, True)
    print(approx)
    print(len(approx))
    print("-----")
    img_contours = None
    if len(approx) == 4:
        x2, y2, x1, y1, x3, y3, x4, y4 = approx.item(0), approx.item(1), approx.item(2), approx.item(3), approx.item(4), approx.item(5), approx.item(6), approx.item(7)
        boundary_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        width, height = 900, 900
        new_boundary = np.float32([[0,0], [width, 0], [0, height], [width, height]])

        persp_mat = cv2.getPerspectiveTransform(boundary_pts, new_boundary)
        img_persp = cv2.warpPerspective(frame, persp_mat, (width, height))
        img_contours = cv2.cvtColor(img_persp, cv2.COLOR_BGR2GRAY)
    return img_contours


vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    canny = cv2.Canny(frame, 50, 350)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((4, 4), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    #blacks = np.zeros(frame.shape, dtype = np.uint8)
    #_, thresh = cv2.threshold(canny, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_contour = maxContour(contours)
    image_contours = None
    if max_contour is not None:
        cv2.drawContours(frame, max_contour, -1, (0, 0, 255), 2)
        image_contours = getCorners(max_contour)

    #cv2.imshow('canny', canny)
    #cv2.imshow('dilated', dilated)
    cv2.imshow('eroded', eroded)
    cv2.imshow('frame', frame)
    if image_contours is not None:
        cv2.imshow('image_contours', image_contours)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
