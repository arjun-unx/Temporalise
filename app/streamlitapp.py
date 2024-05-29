import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
from moviepy.editor import *
import cv2

st.set_page_config(layout='wide', page_title='Temporalize', initial_sidebar_state='collapsed')

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title('Temporalize')
col11, col22 = st.columns(2)
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, 'data', 's1')
options = os.listdir(data_dir)
selected_video = st.selectbox("Chose Video", options)
col1, col2 = st.columns(2)

if options:
    with col1:
        st.info(
            "The video display's the speech which is ready to be converted to text.")
        file_path = os.path.join(base_dir, 'data', 's1', selected_video)
        output_path = os.path.join(base_dir, 'test_video.mp4')
        # os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        video_convert = VideoFileClip(file_path)
        video_convert.write_videofile(output_path, codec='libx264')

        # display the video
        video = open(output_path, 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info("Input for Lip Reading Model")
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        # Normalize pixel values to [0, 8] and convert to uint8 data type
        image_data = tf.clip_by_value(video, 0.0, 9.0)
        # Scale to uint8 range (0-255)
        image_data = tf.cast(image_data * 50, tf.uint8)
        # increase brightness of the image
        image_data = tf.image.adjust_brightness(image_data, 0.3)
        # increase contrast of the image
        image_data = tf.image.adjust_contrast(image_data, 1.5)
        # Convert the tensor data to a list of NumPy arrays for each frame
        frames = [frame.numpy()[:, :, 0] for frame in tf.unstack(image_data)]
        animation_file = os.path.join(base_dir, 'animation.gif')
        imageio.mimsave('animation.gif', frames)
        # imageio.mimsave('animation.gif', video, duration=20)
        st.image('animation.gif', width=400)

        st.info("Resultant vector in tokens")
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(
            yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info("Sentence Level Prediction")
        converted_prediction = num_to_char(decoder)
        st.text(tf.strings.reduce_join(
            converted_prediction).numpy().decode('utf-8'))

