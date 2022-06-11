# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 16:26:36 2022

@author: mosta
"""

import numpy as np
import cv2
import streamlit as st
import keras.models import load_model

# loading the Model
model = load_model('C:\Users\mosta\Desktop\plant_model_weights.h5')

# Name of classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

# Uploading the image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

# On predict button click
if submit:


    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
