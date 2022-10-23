import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import numpy as np
from tensorflow.keras.models import load_model

class_labels = ['street', 'mountain', 'sea', 'forest', 'glacier', 'buildings']

st.title("Intel Multi-Class Image Classification")
st.sidebar.markdown("## Day 1- AV Workshop")
st.sidebar.markdown("### Hands-on Deep Learning Workshop")
st.sidebar.markdown("#### #LearnByDoing")
st.sidebar.markdown("[Visit our Website](https://animevyuh.org/)")

image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if image_file is not None:
    st.image(image_file)
    model = load_model("Intel_model.h5")
    
    img = load_img(image_file,target_size = (224,224,3))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)

    target = model.predict(x)
    target = np.argmax(target)
    
    st.markdown(f"#### The predicted image is {class_labels[target]}")