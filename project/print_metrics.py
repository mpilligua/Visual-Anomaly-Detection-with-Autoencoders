from sklearn.metrics import classification_report
import numpy as np
import cv2
import matplotlib.pyplot as plt
from metrics import MSE_Red_channel
"""
This script contains the different functions of metrics.py
modified to visualize the results
"""
def count_red_pixels_HSV(image_bgr):
    #convert to HSV
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV
    lower_red1 = np.array([0,50,50])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])

    # Threshold the HSV image to count only red pixels
    mask1 = cv2.inRange(image, lower_red1, upper_red1)
    mask2 = cv2.inRange(image, lower_red2, upper_red2)

    mask1 = cv2.add(mask1, mask2)
    return mask1

def count_red_pixels(image_bgr):
    """
    Count the number of red pixels in an image
    """
    
    red_channel, green_channel, blue_channel = image_bgr[:, :, 2], image_bgr[:, :, 1], image_bgr[:, :, 0]
    reddnes = red_channel - (green_channel + blue_channel) / 2
    
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    threshold = 140  # Adjust this threshold as needed

    # Create a mask to filter the red pixels
    red_mask = (reddnes >= threshold)
    img_gray = np.bitwise_and(red_mask, img_gray)
    return img_gray


def MSE(original, treshold=155):
    red_channel, green_channel, blue_channel = original[:, :, 2], original[:, :, 1], original[:, :, 0]
    redness = red_channel - (green_channel + blue_channel) / 2
    redness[redness < treshold] = 0
    return redness

def vectordifference(loader):
    """
    Obtain the X, Y vectors for the tree
    """
    count_function = {"HSV": count_red_pixels_HSV, "RGB": count_red_pixels,
                      "MSE": MSE}
    differences = []
    y = []
    fig, axis = plt.subplots(3, 4, figsize = (10, 10))
    axis = axis.flatten()
    idx = 0

    for original, pred, label, name in loader:
        if idx == 12:
            plt.savefig("/fhome/gia07/project/metric_differences.png")
            exit(0)
        if label.numpy()[0] == 2:
            original = original.squeeze(0).numpy()
            pred = pred.squeeze(0).numpy()
            axis[idx].imshow(original[:, :, ::-1]), axis[idx].set_title("Original")
            axis[idx].axis("off")
            idx += 1
            for function_name in count_function.keys():
                orginal_out = count_function[function_name](original)
                pred_out = count_function[function_name](pred)
                diff = (orginal_out - pred_out)
                if function_name == "MSE":
                    diff = diff ** 2
                axis[idx].imshow(diff, cmap = 'gray'), axis[idx].set_title(f"Difference using {function_name}")
                axis[idx].axis("off")
                idx += 1
            

    cv2.imwrite("/fhome/gia07/project/ouput_autoncoder/test.png", pred.squeeze(0).numpy())

    return np.array(differences), np.array(y)

def vectordifference2D(loader):
    """
    Obtain the X, Y vectors for the tree
    """
    # Counting red pixels in an image
    def difeerence_red(original, autoencoder):
        return [count_red_pixels(original), count_red_pixels(autoencoder)]

    differences = []
    y = []
    for orginial, pred, label, name in loader:
        difference = difeerence_red(orginial.squeeze(0).numpy(), pred.squeeze(0).numpy())
        differences.append(difference)
        y.append(label.numpy()[0])

    cv2.imwrite("/fhome/gia07/project/ouput_autoncoder/test.png", pred.squeeze(0).numpy())

        
    return np.array(differences), np.array(y)
