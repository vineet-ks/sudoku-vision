import cv2
import numpy as np
from keras.models import load_model
import solver


def image_processing_boundary(image):
    """Returns input image after processing for sudoku boundary detection"""
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.Canny(image, 50, 100)
    """image = cv2.GaussianBlur(image, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((4, 4), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)"""
    return image


def image_processing_digits(image):
    """Returns input image after processing for digit recognition"""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    #image = cv2.bitwise_not(image)
    return image


def maxContour(contours):
    """Returns the contour with maximum area, if any"""
    areas = [cv2.contourArea(x) for x in contours]
    max_index = np.argmax(areas)
    max_cnt = contours[max_index]

    if areas[max_index] > 50000:
        return max_cnt
    else:
        return None


def getCorners(max_cnt):
    """Returns the edges of the contour if the approx polygon is a quadrilateral"""
    perimeter = cv2.arcLength(max_cnt, True)
    epsilon = 0.1 * perimeter
    approx = cv2.approxPolyDP(max_cnt, epsilon, True)
    boundary_pts = None

    if len(approx) == 4:
        x2, y2, x1, y1, x3, y3, x4, y4 = approx.item(0), approx.item(1), approx.item(2), approx.item(3), approx.item(
            4), approx.item(5), approx.item(6), approx.item(7)
        boundary_pts = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return boundary_pts


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
    #image = np.zeros((image.shape), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            digit = str(int(pred_reshaped[i][j]))
            if digit == '0':
                continue
            io = int(l * i / 9 + 32)
            jo = int(w * j / 9 + 27)
            image = cv2.putText(image, digit, (jo, io), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

def draw_over_predictions(image):
    #black = np.zeros((342, 342), dtype=np.uint8)
    l, w, _ = image.shape
    for i in range(9):
        for j in range(9):
            sol_digit = sol_reshaped[i][j]
            pred_digit = str(int(pred_reshaped[i][j]))
            if pred_digit == '0':
                io = int(l * i / 9 + 28)
                jo = int(w * j / 9 + 12)
                image = cv2.putText(image, sol_digit, (jo, io), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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
        boundary_pts = getCorners(max_contour)
        if boundary_pts is not None:
            width, height = 342, 342
            new_boundary = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            persp_mat = cv2.getPerspectiveTransform(boundary_pts, new_boundary)
            image_contours = cv2.warpPerspective(img_processed_digits, persp_mat, (width, height))

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

        cv2.imshow('cell1', cell_grid[0][2])

        cell_batch, empty_filter = grid_to_batch(cell_grid)
        predictions = predict_batch(cell_batch, empty_filter)
        pred_reshaped = predictions.reshape(9, 9)
        print(pred_reshaped)


        frame_persp = cv2.warpPerspective(frame, persp_mat, (342, 342))
        image_write_over = write_over(frame_persp)
        cv2.imshow("write_over", image_write_over)

        pred_formatted = np.array2string(predictions.astype('int'), separator='')
        pred_formatted = pred_formatted[1:-1]
        solution = solver.solve(pred_formatted)


        if solution:
            print(solution.values())
            sol_reshaped = np.asarray(list(solution.values())).reshape(9, 9)
            sol_write_over = draw_over_predictions(image_write_over)
            cv2.imshow("sol_write_over", sol_write_over)
            #print(np.fromiter(solution.values(), dtype=str))

            persp_mat = cv2.getPerspectiveTransform(new_boundary, boundary_pts)
            sol_warped = cv2.warpPerspective(sol_write_over, persp_mat, (frame.shape[1], frame.shape[0]))
            sol_warp_overlayed = cv2.max(sol_warped, frame)
            cv2.imshow("sol_warp_overlayed", sol_warp_overlayed)
            #print(frame.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
