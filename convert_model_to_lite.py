import tensorflow as tf

model = tf.keras.models.load_model('models/model.h5')

#create module converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#converter
tflite_model = converter.convert()

#write
open("models/model.tflite","wb").write(tflite_model)