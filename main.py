import cv2
import numpy as np
from keras.models import load_model

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
    #print(approx)
    #print(len(approx))
    #print("-----")
    img_persp = None
    if len(approx) == 4:
        x2, y2, x1, y1, x3, y3, x4, y4 = approx.item(0), approx.item(1), approx.item(2), approx.item(3), approx.item(4), approx.item(5), approx.item(6), approx.item(7)
        boundary_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        width, height = 306, 306
        new_boundary = np.float32([[0,0], [width, 0], [0, height], [width, height]])

        persp_mat = cv2.getPerspectiveTransform(boundary_pts, new_boundary)
        img_persp = cv2.warpPerspective(img_processed, persp_mat, (width, height)) #Using eroded
        #img_contours = cv2.cvtColor(img_persp, cv2.COLOR_BGR2GRAY)
    return img_persp

def predict_all_grid(cell_grid):
    "Returns a list of prediction for each cell"
    prediction_array = []
    for row in cell_grid:
        for cell in row:
            prediction = predict_digit(cell)
            prediction_array.append(prediction)
    return prediction_array

def predict_digit(cell):
    "Returns prediction of a digit for the given cell"
    #print(model.summary())
    if np.sum(cell) < 6000:
        return np.nan
    #gray = cv2.cvtColor(im_4, cv2.COLOR_BGR2GRAY)
    cell = np.reshape(cell, (1, 28, 28, 1))
    return model.predict_classes(cell)

vid = cv2.VideoCapture(0)
model = load_model('model_mnist')



while True:
    ret, frame = vid.read()

    canny = cv2.Canny(frame, 50, 350)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(canny, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(blur, kernel, iterations=1)
    kernel = np.ones((4, 4), np.uint8)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    # blacks = np.zeros(frame.shape, dtype = np.uint8)
    _, thresh = cv2.threshold(eroded, 0, 255, cv2.THRESH_OTSU)
    img_processed = thresh
    contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_contour = maxContour(contours)
    image_contours = None
    if max_contour is not None:
        cv2.drawContours(frame, max_contour, -1, (0, 0, 255), 2)
        image_contours = getCorners(max_contour)

    #cv2.imshow('canny', canny)
    #cv2.imshow('dilated', dilated)
    cv2.imshow('eroded', img_processed)
    cv2.imshow('frame', frame)
    if image_contours is not None:
        cv2.imshow('image_contours', image_contours)

        cell_grid = []
        for i in range(9):
            row = []
            for j in range(9):
                row.append(image_contours[3+i*34:31+i*34, 3+j*34:31+j*34])
            cell_grid.append(row)

        #print(len(cell_grid),len(cell_grid[0]),len(cell_grid[0][0]),len(cell_grid[0][0][0]),)
        cv2.imshow('cell1', cell_grid[0][1])
        #temp = cell_grid[0][1]
        #print(np.sum(temp))
        #print(predict_digit(cell_grid[0][1]))
        prediction_array = predict_all_grid(cell_grid)
        print(prediction_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
