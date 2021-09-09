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
    perimeter = cv2.arcLength(max_cnt, True)
    epsilon = 0.1 * perimeter
    approx = cv2.approxPolyDP(max_cnt, epsilon, True)
    #print(approx)
    #print(len(approx))
    #print("-----")
    img_persp = None
    persp_mat = None
    if len(approx) == 4:
        x2, y2, x1, y1, x3, y3, x4, y4 = approx.item(0), approx.item(1), approx.item(2), approx.item(3), approx.item(4), approx.item(5), approx.item(6), approx.item(7)
        boundary_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        width, height = 342, 342
        new_boundary = np.float32([[0,0], [width, 0], [0, height], [width, height]])

        persp_mat = cv2.getPerspectiveTransform(boundary_pts, new_boundary)
        img_persp = cv2.warpPerspective(im, persp_mat, (width, height)) #Using eroded
        #img_contours = cv2.cvtColor(img_persp, cv2.COLOR_BGR2GRAY)
    return img_persp, persp_mat


def grid_to_batch(cell_grid):
    """Takes an array of each image cell and return a single batch for ML prediction. Also returns estimated empty cells"""
    cell_batch = np.empty([0, 28, 28, 1])
    empty_filter = np.ones(81)
    for i, row in enumerate(cell_grid):
        for j, cell in enumerate(row):
            cell_batch = np.append(cell_batch, cell.reshape(1, 28, 28, 1), axis=0)
            if np.sum(cell) < 15000:
                empty_filter[i*9 + j] = 0
    return cell_batch, empty_filter


def predict_batch(cell_batch, empty_filter):
    """Predicts the digits for the given mask and returns it after replacing empty cells with zeroes"""
    predictions_raw = model.predict_classes(cell_batch)
    predictions = predictions_raw * empty_filter
    return predictions

def write_over(image):
    print(type(image))
    l, w, _ = image.shape
    #print(l, w)
    for i in range(9):
        for j in range(9):
            digit = str(int(predictions[i][j]))
            if digit == '0':
                continue
            io = int(l * i / 9 + 27)
            wo = int(w * j / 9 + 32)
            image = cv2.putText(image, digit, (io, wo), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

vid = cv2.VideoCapture(0)
model = load_model('model_temp')



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

    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    im = cv2.bitwise_not(im)

    max_contour = maxContour(contours)
    image_contours = None
    if max_contour is not None:
        cv2.drawContours(frame, max_contour, -1, (0, 0, 255), 2)
        image_contours, persp_mat = getCorners(max_contour)

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
                row.append(image_contours[5+i*38:33+i*38, 5+j*38:33+j*38])
            cell_grid.append(row)

        cv2.imshow('cell1', cell_grid[7][3])

        cell_batch, empty_filter = grid_to_batch(cell_grid)
        predictions = predict_batch(cell_batch, empty_filter)
        predictions = predictions.reshape(9, 9)
        print(predictions)

        frame_persp = cv2.warpPerspective(frame, persp_mat, (342, 342))
        image_write_over = write_over(frame_persp)
        cv2.imshow("write_over", image_write_over)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
