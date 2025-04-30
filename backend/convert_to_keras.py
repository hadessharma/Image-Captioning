# convert_to_keras.py
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import TFSMLayer

# 1) Load the legacy SavedModel
saved = tf.saved_model.load("image_caption_model")

# 2) Grab its serving signature
sig = saved.signatures.get("serving_default")
if sig is None:
    sig = list(saved.signatures.values())[0]

# 3) Build a tiny Keras Model wrapping that signature
#    - assume your model takes [1,299,299,3] float32 input
img_in = Input(shape=(299, 299, 3), dtype=tf.float32, name="image")
# TFSMLayer will handle batching: wrap the loaded SavedModel
inf_layer = TFSMLayer(saved, call_endpoint=sig.name)
# call it
out = inf_layer(img_in)

model = Model(inputs=img_in, outputs=out)
model.trainable = False

# 4) Save in Keras v3 format
model.save("model.keras", save_format="keras")
print("✅ Converted to model.keras")
