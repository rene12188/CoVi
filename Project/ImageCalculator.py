import os

import cv2
import tensorflow.keras as keras



class DigitCalculator:

    def __init__(self):
        self.model = keras.models.load_model('./models/Digit_Recognizer')

    def CalculateImage(self, image):
        features = self.__extract_labels(image)
        print(len(features))
        return features.copy()

    def __extract_labels(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 100,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        returnarray = []

        output = cv2.connectedComponentsWithStats(
            thresh, 10, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output

        print(numLabels)

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
            crop = crop[y - 10:y + h + 10, x - 10:x + h + 10]
            returnarray.append(crop)
        return returnarray
