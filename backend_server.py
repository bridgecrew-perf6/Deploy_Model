from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
import tensorflow as tf
from PIL import Image
import numpy as np
import keras
from keras.preprocessing.image import img_to_array
import numpy as np
import json
import io
import image_net

app = Flask(__name__)

global flower_model
flower_model = None

# Encoding numpy to json
class NumpyEncoder(json.JSONEncoder):
    '''
    Encoding numpy into json
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def preprocess_image(img,shape):
    img_rz = img.resize(shape)
    img_rz = img_to_array(img_rz)
    img_rz = np.expand_dims(img_rz, axis=0)
    return img_rz

@app.route('/backend',methods = ['POST'])
def classification_process():
    data = {"success": False}
    if request.files.get("image"):
        # Lấy file ảnh người dùng upload lên
        image = request.files["image"].read()
        # Convert sang dạng array image
        image = Image.open(io.BytesIO(image))
        # resize ảnh
        image_rz = preprocess_image(image,(224,224))
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
    return json.dumps(data, ensure_ascii=False, cls= NumpyEncoder)

if __name__ == "__main__":
    print("App run!")
    flower_model = tf.keras.models.load_model("models/model.h5")
    app.run(debug=False, threaded=False)