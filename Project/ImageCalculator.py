import os

import cv2

import tensorflow.keras as keras
import numpy as np
import pandas as pd

class DigitCalculator:

    def __init__(self):
        self.model = keras.models.load_model('./models/Digit_Recognizer')
        self.train_set = {'+': 0, '-': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10,
                             '9': 11}

    def CalculateImage(self, image):
        features = self.__extract_labels(image)
        #return features
        classification = []
        #return features

        for image in features:
            classification.append(self.__prediction(image))

        classification = np.rot90(classification)
        detected_labels = classification[1]
        calcu = ""
        for symbol in detected_labels:
            calcu += symbol


        return eval(calcu)



    def __extract_labels(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 100,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        data = []
        x_data = []

        output = cv2.connectedComponentsWithStats(
            thresh, 10, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        returnarray = []
        for i in range(1, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format(i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            output = image.copy()
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            crop = image.copy()
            crop = crop[y :y + h , x :x + h ]

            data.append(crop)
            x_data.append(x)
           # print(x)

        returnarray = [data, x_data]
        df = pd.DataFrame(returnarray)
        df = df.T
#        df = df.iloc[1:, :]
        df[1] = df[1].apply(pd.to_numeric)
        df = df.sort_values(by=[1])

        return np.array(df[0])

    def __prediction(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(img, cmap = 'gray')
        img = cv2.resize(img, (40, 40))
        norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = img / 255
        norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
        case = np.asarray([norm_image])
        pred = np.argmax(self.model.predict([case]), axis=-1)
        return ([i for i in self.train_set if self.train_set[i] == (pred[0])][0], pred)

