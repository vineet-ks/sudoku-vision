import cv2
import numpy as np
import os
import shutil


image = cv2.imread("sudoku_digits_grid.png")
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
canny = cv2.Canny(image, 50, 350)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(canny, kernel, iterations=1)
#kernel = np.ones((3, 3), np.uint8)
##eroded = cv2.erode(dilated, kernel, iterations=1)
#img_processed = canny
#cv2.imshow('dilated', dilated)
#cv2.waitKey(0)

shutil.rmtree('data')
os.mkdir('data')
for i in range(10):
    os.mkdir('data/{}'.format(i))

contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

i = 0
j = -1
for cnt_elt in contours:
    area = cv2.contourArea(cnt_elt)
    if 1000 < area < 10000:
        j += 1
        last_digit = i % 10
        remaining_digit = i // 10
        #second_last_digit = temp % 10
        if j % 11 == 0:
            continue

        x, y, w, h = cv2.boundingRect(cnt_elt)
        cell = image[y:y + h, x:x + w]
        cell = cv2.resize(cell, (28, 28))
        cell = cv2.bitwise_not(cell)
        cv2.imwrite('data/{}/{}_{}.jpg'.format(9 - last_digit, 9 - last_digit, remaining_digit), cell)
        i += 1


