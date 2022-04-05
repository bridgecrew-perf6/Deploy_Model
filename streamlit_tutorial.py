import streamlit as st
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import img_to_array
import image_net
import json
import io
from PIL import Image

def preprocess_image(img,shape):
    img_rz = img.resize(shape)
    img_rz = img_to_array(img_rz)
    img_rz = np.expand_dims(img_rz, axis=0)
    return img_rz

# class NumpyEncoder(json.JSONEncoder):
#     '''
#     Encoding numpy into json
#     '''
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, np.int32):
#             return int(obj)
#         if isinstance(obj, np.int64):
#             return int(obj)
#         if isinstance(obj, np.float32):
#             return float(obj)
#         if isinstance(obj, np.float64):
#             return float(obj)
#         return json.JSONEncoder.default(self, obj)

def load_image(image_file):
	img = Image.open(image_file)
	return img

if __name__ == '__main__':
    flower_model = tf.keras.models.load_model("models/model.h5")
    st.title("Deploy model!!!")
    st.header("flower image classification")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    image = st.file_uploader("get Image",type=['png', 'jpg', 'jpeg'])
    if st.button('Predict'):
        if image:
            st.header("Result!!!")
            st.info("JPEG, JPG, PNG")
            data = {"success": False}
            image = load_image(image)
            # image = Image.open(io.BytesIO(image))
            # resize ảnh
            image_rz = preprocess_image(image, (224, 224))
            # Dự báo phân phối xác suất
            dist_probs = flower_model.predict(image_rz)
            # argmax 5
            argmax_k = np.argsort(dist_probs[0])[::-1][:5]
            # classes 5
            classes = [image_net.classes[idx] for idx in list(argmax_k)]
            # probability of classes
            classes_prob = [dist_probs[0, idx] for idx in list(argmax_k)]
            data["probability"] = dict(zip(classes, classes_prob))
            data["success"] = True
            st.image(image, width=250)
            st.write(data)
        else:
            st.header("Fail!!!")

