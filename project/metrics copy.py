from sklearn.metrics import classification_report
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Counting red pixels in an image
def count_red_pixels_HSV(image_bgr):
    #convert to HSV
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV
    lower_red1 = np.array([0,100,20])
    upper_red1 = np.array([8,255,255])

    lower_red2 = np.array([175,100,20])
    upper_red2 = np.array([179,255,255])
    # Threshold the HSV image to count only red pixels
    mask1 = cv2.inRange(image, lower_red1, upper_red1)
    mask2 = cv2.inRange(image, lower_red2, upper_red2)
    mask = cv2.add(mask1 + mask2)
    return np.sum(mask)


def count_red_pixels(image_bgr):
    """
    Count the number of red pixels in an image
    """
    
    red_channel, green_channel, blue_channel = image_bgr[:, :, 2], image_bgr[:, :, 1], image_bgr[:, :, 0]
    reddnes = red_channel - (green_channel + blue_channel) / 2
    bluness = blue_channel - (green_channel + red_channel) / 2
    greenss = green_channel - (blue_channel + red_channel) / 2
    # If reddnes is negative, set it to 0
    
    threshold = 140  # Adjust this threshold as needed

    # Create a mask to filter the red pixels
    red_mask = (reddnes >= threshold)
    blue_mask = (bluness >= threshold)
    green_mask = (greenss >= threshold)

    return [np.sum(red_mask), np.sum(blue_mask), np.sum(green_mask)]


#Calculating histogram of the image
def calculate_red_histogram(image_path):
    
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    red_channel, green_channel, blue_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    reddnes = red_channel - (green_channel + blue_channel) / 2
    # If reddnes is negative, set it to 0
    reddnes[reddnes < 0] = 0

    hist = np.histogram(reddnes, bins=256, range=(0, 256))

    return hist

def vectordifference(loader):
    """
    Obtain the X, Y vectors for the tree
    """
    # Counting red pixels in an image
    def difeerence_red(original, autoencoder):
        n1 = count_red_pixels(original) 
        n2 = count_red_pixels(autoencoder)
        return np.array(n1)-np.array(n2)

    differences = []
    y = []
    for orginial, pred, label, name in loader:
        difference = difeerence_red(orginial.squeeze(0).numpy(), pred.squeeze(0).numpy())
        differences.append(difference)
        y.append(label.numpy()[0])

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
