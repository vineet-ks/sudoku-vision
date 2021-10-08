import cv2
import numpy as np
from keras.models import load_model
from solver import solve


def image_processing_boundary(image):
    """Returns input image after processing for sudoku boundary detection"""
    image = cv2.Canny(image, 50, 100)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((4, 4), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return image


def image_processing_digits(image):
    """Returns input image after processing for digit recognition"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    image = cv2.bitwise_not(image)
    return image


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
    # print(approx)
    # print(len(approx))
    # print("-----")
    img_persp = None
    persp_mat = None
    if len(approx) == 4:
        x2, y2, x1, y1, x3, y3, x4, y4 = approx.item(0), approx.item(1), approx.item(2), approx.item(3), approx.item(
            4), approx.item(5), approx.item(6), approx.item(7)
        boundary_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

        width, height = 342, 342
        new_boundary = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        persp_mat = cv2.getPerspectiveTransform(boundary_pts, new_boundary)
        img_persp = cv2.warpPerspective(img_processed_digits, persp_mat, (width, height))  # Using eroded
        # img_contours = cv2.cvtColor(img_persp, cv2.COLOR_BGR2GRAY)
    return img_persp, persp_mat


def grid_to_batch(cell_grid):
    """Takes an array of each image cell and return a single batch for ML prediction. Also returns estimated empty cells"""
    cell_batch = np.empty([0, 28, 28, 1])
    empty_filter = np.ones(81)
    for i, row in enumerate(cell_grid):
        for j, cell in enumerate(row):
            cell_batch = np.append(cell_batch, cell.reshape(1, 28, 28, 1), axis=0)
            if np.sum(cell) < 15000:
                empty_filter[i * 9 + j] = 0
    return cell_batch, empty_filter


def predict_batch(cell_batch, empty_filter):
    """Predicts the digits for the given mask and returns it after replacing empty cells with zeroes"""
    predictions_vectorized = model.predict(cell_batch)
    predictions_raw = np.argmax(predictions_vectorized, axis=1)
    predictions = predictions_raw * empty_filter
    return predictions


def write_over(image):
    print(type(image))
    l, w, _ = image.shape
    # print(l, w)
    for i in range(9):
        for j in range(9):
            digit = str(int(pred_reshaped[i][j]))
            if digit == '0':
                continue
            io = int(l * i / 9 + 32)
            jo = int(w * j / 9 + 27)
            image = cv2.putText(image, digit, (jo, io), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image


vid = cv2.VideoCapture(0)
model = load_model('model')

while True:
    ret, frame = vid.read()

    img_processed_boundary = image_processing_boundary(frame)
    contours, _ = cv2.findContours(img_processed_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img_processed_digits = image_processing_digits(frame)

    max_contour = maxContour(contours)
    image_contours = None
    if max_contour is not None:
        cv2.drawContours(frame, max_contour, -1, (0, 0, 255), 2)
        image_contours, persp_mat = getCorners(max_contour)

    cv2.imshow('img_processed_boundary', img_processed_boundary)
    cv2.imshow('frame', frame)
    if image_contours is not None:
        cv2.imshow('image_contours', image_contours)

        cell_grid = []
        for i in range(9):
            row = []
            for j in range(9):
                row.append(image_contours[5 + i * 38:33 + i * 38, 5 + j * 38:33 + j * 38])
            cell_grid.append(row)

        cv2.imshow('cell1', cell_grid[7][3])

        cell_batch, empty_filter = grid_to_batch(cell_grid)
        predictions = predict_batch(cell_batch, empty_filter)
        pred_reshaped = predictions.reshape(9, 9)
        print(pred_reshaped)


        frame_persp = cv2.warpPerspective(frame, persp_mat, (342, 342))
        image_write_over = write_over(frame_persp)
        cv2.imshow("write_over", image_write_over)

        pred_formatted = np.array2string(predictions.astype('int'), separator='')
        pred_formatted = pred_formatted[1:-1]
        solution = solve(pred_formatted)


        if solution:
            print(solution.values())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
