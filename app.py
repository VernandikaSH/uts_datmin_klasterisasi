import streamlit as st
import cv2
import numpy as np
from PIL import Image

import joblib

def load_clustering_model():
    try:
        # Load the pre-trained model
        model = joblib.load('model/model.pkl')
        st.write("Model loaded successfully!")  # Log success
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is saved as 'model.pkl'.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the scaler
def load_scaler():
    try:
        scaler = joblib.load('model/scaler.pkl')
        return scaler
    except Exception as e:
        st.error(f"An error occurred while loading the scaler: {e}")
        return None

# Load PCA
def load_pca():
    try:
        pca = joblib.load('model/pca.pkl')
        return pca
    except Exception as e:
        st.error(f"An error occurred while loading PCA: {e}")
        return None

# Function to preprocess an uploaded image
def preprocess_image(pil_image):
    # Convert PIL image to a NumPy array
    image = np.array(pil_image)

    # Check if the image is in RGB format and convert to BGR
    if image.shape[-1] == 3:  # If the image has 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the image
    image = cv2.resize(image, (128, 128))

    # Normalize pixel values (optional, depends on your next model)
    normalized_image = image.astype(np.float32) / 255.0

    return normalized_image  # Return the original resized color image


def extract_color_histogram(image):
    # Convert the image to uint8 if it's not already
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Ensure the image has 3 channels before calculating the histogram
    if len(image.shape) < 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

    # Extract color histogram with 32 bins per channel
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Normalize histogram

    return hist

from skimage.feature import graycomatrix, graycoprops
def extract_glcm_features(image):
    # Check if the image is already grayscale. If not, convert it.
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check for 3-channel color image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:  # Assume image is already grayscale or has a single channel
        gray_image = image

    gray_image = np.uint8(gray_image)  # Ensure image is in uint8 format
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [contrast, homogeneity, energy, correlation]

def extract_sift_features(image, fixed_length=128):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    if descriptors is not None:
        descriptors = descriptors.flatten()
        # Truncate or pad to fixed length
        if len(descriptors) > fixed_length:
            descriptors = descriptors[:fixed_length]
        elif len(descriptors) < fixed_length:
            descriptors = np.pad(descriptors, (0, fixed_length - len(descriptors)))
        return descriptors
    else:
        return np.zeros(fixed_length)  # Return zeros if no descriptors
    
def extract_combined_features(image):
    color_features = extract_color_histogram(image)
    glcm_features = extract_glcm_features(image)
    sift_features = extract_sift_features(image)

    # Gabungkan semua fitur
    combined_features = np.hstack([color_features, glcm_features, sift_features])
    return combined_features

def predict_cluster(image, model, scaler, pca):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Extract combined features
    features = extract_combined_features(processed_image)

    # Convert features to a numpy array
    features = np.array([0 if val is None else val for val in features]).reshape(1, -1)  # Ensure it's 2D

    # Normalize the features using the loaded scaler
    normalized_features = scaler.transform(features)  # Use transform instead of fit_transform

    # Apply PCA
    features_reduced = pca.transform(normalized_features)

    # Predict cluster using the features
    cluster = model.predict(features_reduced)

    return cluster[0], processed_image

# Streamlit App
st.title("Image Clustering App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

# Load the clustering model
model = load_clustering_model()
scaler = load_scaler()
pca = load_pca()

if uploaded_file is not None and model is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform clustering
    cluster, preprocessed_image = predict_cluster(image, model, scaler, pca)
    
    # Display the preprocessed image
    preprocessed_image_display = (preprocessed_image * 255).astype(np.uint8)  # Scale back to 0-255
    st.image(preprocessed_image_display, caption="Preprocessed Image", use_column_width=True)
    
    # Display the cluster result
    st.write(f"The uploaded image belongs to Cluster: {cluster}")
else:
    if model is None:
        st.error("The model could not be loaded.")