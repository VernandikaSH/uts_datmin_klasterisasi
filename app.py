# import library library yang digunakan
import streamlit as st
import cv2
import numpy as np
from PIL import Image

import joblib

# Load model clustering
def load_clustering_model():
    try:
        model = joblib.load('model/model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is saved as 'model.pkl'.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load scaler model
def load_scaler():
    try:
        scaler = joblib.load('model/scaler.pkl')
        return scaler
    except Exception as e:
        st.error(f"An error occurred while loading the scaler: {e}")
        return None

# Load PCA model
def load_pca():
    try:
        pca = joblib.load('model/pca.pkl')
        return pca
    except Exception as e:
        st.error(f"An error occurred while loading PCA: {e}")
        return None

# fungsi preprocess gambar
def preprocess_image(pil_image):

    image = np.array(pil_image)

    if image.shape[-1] == 3:  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (128, 128))

    normalized_image = image.astype(np.float32) / 255.0

    return normalized_image  

# ekstraksi fitur warna
def extract_color_histogram(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) < 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  

    return hist

# ekstraksi fitur tekstur
from skimage.feature import graycomatrix, graycoprops
def extract_glcm_features(image):
    if len(image.shape) == 3 and image.shape[2] == 3:  
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: 
        gray_image = image

    gray_image = np.uint8(gray_image) 
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [contrast, homogeneity, energy, correlation]

# ekstrak fitur lokal
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
        if len(descriptors) > fixed_length:
            descriptors = descriptors[:fixed_length]
        elif len(descriptors) < fixed_length:
            descriptors = np.pad(descriptors, (0, fixed_length - len(descriptors)))
        return descriptors
    else:
        return np.zeros(fixed_length)  

# gabungkan ketiga fitur
def extract_combined_features(image):
    color_features = extract_color_histogram(image)
    glcm_features = extract_glcm_features(image)
    sift_features = extract_sift_features(image)

    combined_features = np.hstack([color_features, glcm_features, sift_features])
    return combined_features

# memprediksi gambar yang diinput masuk ke cluster mana
def predict_cluster(image, model, scaler, pca):
    processed_image = preprocess_image(image)
    
    features = extract_combined_features(processed_image)

    features = np.array([0 if val is None else val for val in features]).reshape(1, -1)  # Ensure it's 2D

    normalized_features = scaler.transform(features)  

    features_reduced = pca.transform(normalized_features)

    cluster = model.predict(features_reduced)

    return cluster[0], processed_image

# Website Streamlit
st.title("Image Clustering App")

# Upload gambar
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

# Load model
model = load_clustering_model()
scaler = load_scaler()
pca = load_pca()

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # jalankan fungsi clustering
    cluster, preprocessed_image = predict_cluster(image, model, scaler, pca)
    
    # tampilkan preprocessed image
    preprocessed_image_display = (preprocessed_image * 255).astype(np.uint8)  # Scale back to 0-255
    st.image(preprocessed_image_display, caption="Preprocessed Image", use_column_width=True)
    
    # tampilkan hasil cluster
    st.write(f"The uploaded image belongs to Cluster: {cluster}")
else:
    if model is None:
        st.error("The model could not be loaded.")