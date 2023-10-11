import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import IPython as ipt
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from IPython.display import Image
from tensorflow.keras.preprocessing import image


model_loaded = tf.keras.models.load_model('./nnet/my_model.h5')

st.title("Ants & Bees")

st.Header("Ants & Bees")
uploaded_file = st.file_uploader("Load an image!", [png, jpg, jpeg])

if uploaded_file is not null:
  Image(uploaded_file, width=150, height=150)
  img = image.load_img(uploaded_file, target_size=(150, 150), grayscale=False)
  x = image.img_to_array(img)
  x = 255 - x
  x /= 255
  x = np.expand_dims(x, axis=0)
  prediction = model.predict(x)
  score = float(prediction[0])
  fig = plt.figure()
  plt.imshow(img)
  plt.title(f'ant:{100*(1-score):.2f}%; bee:{100 * score:.2f}%')
  plt.show
  st.pyplot(fig)
