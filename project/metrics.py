from sklearn.metrics import classification_report
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    # Open cv 
    mask_value = np.sum(mask1) + np.sum(mask2)
    return mask_value

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

def difeerence_red(original, autoencoder):
    n1 = count_red_pixels_HSV(original) 
    n2 = count_red_pixels_HSV(autoencoder)
    return np.array(n1) - np.array(n2)

def vectordifference(loader):
    differences = []
    y = []
    for orginial, pred, label, name in loader:
        difference = difeerence_red(orginial.squeeze(0).numpy(), pred.squeeze(0).numpy())
        differences.append(difference)
        y.append(label.numpy()[0])

    return np.array(differences), np.array(y)

def vectordifference2D(loader):
    """
    Obtain the X, Y vectors for the tree
    """
    # Counting red pixels in an image
    def difeerence_red(original, autoencoder):
        return np.array(count_red_pixels(original)) - np.array(count_red_pixels(autoencoder))

    differences = []
    y = []
    for orginial, pred, label, name in loader:
        difference = difeerence_red(orginial.squeeze(0).numpy(), pred.squeeze(0).numpy())
        differences.append(difference)
        y.append(label.numpy()[0])

    # cv2.imwrite("/fhome/gia07/project/ouput_autoncoder/test.png", pred.squeeze(0).numpy())

        
    return np.array(differences), np.array(y)



def vectordifference_Cropped(loader):
    """
    Obtain the X, Y vectors for the tree
    """
    # Counting red pixels in an image
    def difeerence_red(original, autoencoder):
        return np.array(count_red_pixels(original)) - np.array(count_red_pixels(autoencoder))

    differences = []
    for orginial, pred, name in tqdm(loader):
        difference = difeerence_red(orginial.squeeze(0).numpy(), pred.squeeze(0).numpy())
        differences.append(difference)
        
    return np.array(differences)

def MSE_Red_channel(loader, treshold=155):
    """
    Obtain the X, Y vectors for the tree
    """
    # Counting red pixels in an image
    def MSE(original, autoencoder, treshold=155):
        red_channel, green_channel, blue_channel = original[:, :, 2], original[:, :, 1], original[:, :, 0]
        red_channel_pred, green_channel_pred, blue_channel_pred = autoencoder[:, :, 2], autoencoder[:, :, 1], autoencoder[:, :, 0]

        redness = red_channel - (green_channel + blue_channel) / 2
        redness[redness < treshold] = 0
        redness_pred = red_channel_pred - (green_channel_pred + blue_channel_pred) / 2
        redness_pred[redness_pred < treshold] = 0
        return np.mean((redness - redness_pred)**2)

    differences = []
    y = []
    for orginial, pred, label, name in tqdm(loader):
        difference = MSE(orginial.squeeze(0).numpy(), pred.squeeze(0).numpy(), treshold=treshold)
        differences.append(difference)
        label_np = label.numpy()[0]
        y.append(label_np)
        
    return np.array(differences).reshape(-1, 1), np.array(y)

def MSE(original, autoencoder, treshold=155):
    # Resize original to 256x256 if needed
    if original.shape != (256, 256, 3):
        original = cv2.resize(original, (256, 256))
    red_channel, green_channel, blue_channel = original[:, :, 2], original[:, :, 1], original[:, :, 0]
    red_channel_pred, green_channel_pred, blue_channel_pred = autoencoder[:, :, 2], autoencoder[:, :, 1], autoencoder[:, :, 0]

    redness = red_channel - (green_channel + blue_channel) / 2
    redness[redness < treshold] = 0
    redness_pred = red_channel_pred - (green_channel_pred + blue_channel_pred) / 2
    redness_pred[redness_pred < treshold] = 0
    return np.mean((redness - redness_pred)**2)

def MSE_Red_channel_cropped(loader, treshold=155):
    """
    Obtain the X, Y vectors for the tree
    """
    differences = []
    list_names = []
    for orginial, pred, name in tqdm(loader):
        difference = MSE(orginial.squeeze(0).numpy(), pred.squeeze(0).numpy(), treshold=treshold)
        differences.append(difference)
        list_names.append(name[0])
        
    return np.array(differences).reshape(-1, 1), list_names