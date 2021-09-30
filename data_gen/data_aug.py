from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import os


augmentation = ImageDataGenerator(rotation_range=5, width_shift_range=0.15,
                                  height_shift_range=0.15,
                                  zoom_range=0.3, brightness_range=[0.4, 1])

data_path = 'data'

for subdir in os.listdir(data_path):
    sub_path = os.path.join(data_path, subdir)
    list_images =  os.listdir(sub_path)
    for img_name in list_images:
        img_path = os.path.join(sub_path, img_name)
        image = cv2.imread(img_path)
        image = np.expand_dims(image, axis=0)
        transformed_image = augmentation.flow(image)
        for i in range(10):
            x = transformed_image.next()
            aug_image = x[0].astype('uint8')
            aug_name = ''.join([img_name[:-4], '__{}.jpg'.format(i)])
            cv2.imwrite('data/{}/{}'.format(subdir, aug_name), aug_image)
"""
image = cv2.imread('/home/enix/PycharmProjects/sudoku_vision/3_33.jpg')
image = np.expand_dims(image, axis=0)
transformed_image = augmentation.flow(image)

for i in range(10):
    x = transformed_image.next()
    aug = x[0].astype('uint8')
    plt.imshow(aug)
plt.show()
"""