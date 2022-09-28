
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import streamlit as st 
import functools
import os
import urllib.request
from PIL import Image
st.set_page_config(layout="wide")
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


def load_image(img_path):
    #img = tf.keras.utils.get_file(os.path.basename(img_path)[-128:], img_path)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img
#---------------------------------------------------------------------------
style_names = ['1',
               '2',
               '3',
               '4',
               '5',
               '6',
               '7',
               '8',
               '9',
               "10",
               "11",
               "12"]

style_path = ['models/style_image/10123252_lion-print-file.jpg',
              'models/style_image/Amadeo_de_Souza-Cardoso,_1915_-_Landscape_with_black_figure.jpg',
              'models/style_image/Edvard_Munch,_1893,_The_Scream,_oil,_tempera_and_pastel_on_cardboard,_91_x_73_cm,_National_Gallery_of_Norway.jpg',
              'models/style_image/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg',
              "models/style_image/Les_Demoiselles_d'Avignon.jpg",
              "models/style_image/Pablo_Picasso,_1911-12,_Violon_(Violin),_oil_on_canvas,_Kröller-Müller_Museum,_Otterlo,_Netherlands.jpg",
              "models/style_image/Pablo_Picasso,_1911,_Still_Life_with_a_Bottle_of_Rum,_oil_on_canvas,_61.3_x_50.5_cm,_Metropolitan_Museum_of_Art,_New_York.jpg",
              'models/style_image/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',
              'models/style_image/The_Great_Wave_off_Kanagawa.jpg',
              "models/style_image/Untitled_(Still_life)_(1913)_-_Amadeo_Souza-Cardoso_(1887-1918)_(17385824283).jpg",
              'models/style_image/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
              'models/style_image/Vassily_Kandinsky,_1913_-_Composition_7.jpg']
style_image_temp = load_image(style_path[4])

content_temp = load_image('VAN_CAT.png')
stylized_image = model(tf.constant(content_temp), tf.constant(style_image_temp))[0]

content_url= ""
col_url1 , col_style = st.columns([6,3])
with col_url1:
    content_url = st.text_input(label = 'Content image URL')
    
with col_style:
    option = st.selectbox(
            'You can select style', style_names)        
        
if st.button('Apply',key=11):
        option = int(option)
        urllib.request.urlretrieve(content_url, '{content_url}.jpg')
        content = load_image('{content_url}.jpg')
        style_image = load_image(style_path[option])
        stylized_image = model(tf.constant(content), tf.constant(style_image))[0]
    
#----------------------------------------header-----------------------------
col31, col41  = st.columns([6,3])
with col31 :
    st.subheader('Stylized Image', anchor=None)
with col41 :
    st.subheader('Styles', anchor=None)

#-----------------style---------------------------
img_size =(200,200)
col1, col2,col3, col4  = st.columns([6, 1,1,1])
with col1:
            
            st.image(np.squeeze(stylized_image))            
with col2:
            
            img1 = Image.open(style_path[0])
            resized1 = img1.resize(img_size)
            st.text(style_names[0])
            st.image(resized1)
            
            img2 = Image.open(style_path[1])
            resized2 = img2.resize(img_size)
            st.text(style_names[1])
            st.image(resized2)
            
            img3 = Image.open(style_path[2])
            resized3 = img3.resize(img_size)
            st.text(style_names[2])
            st.image(resized3)
            
            img4 = Image.open(style_path[3])
            resized4 = img4.resize(img_size)
            st.text(style_names[3])
            st.image(resized4)
            
with col3:
            img5 = Image.open(style_path[4])
            resized5 = img5.resize(img_size)
            st.text(style_names[4])
            st.image(resized5)
            
            img6 = Image.open(style_path[5])
            resized6 = img6.resize(img_size)
            st.text(style_names[5])
            st.image(resized6)
            
            img7 = Image.open(style_path[6])
            resized7 = img7.resize(img_size)
            st.text(style_names[6])
            st.image(resized7)
            
            img8 = Image.open(style_path[7])
            resized8 = img8.resize(img_size)
            st.text(style_names[7])
            st.image(resized8)
            
     
with col4 :
            
            img9 = Image.open(style_path[8])
            resized9 = img9.resize(img_size)
            st.text(style_names[8])
            st.image(resized9)
            
            img10 = Image.open(style_path[9])
            resized10 = img10.resize(img_size)
            st.text(style_names[9])
            st.image(resized10)
            
            img11 = Image.open(style_path[10])
            resized11 = img11.resize(img_size)
            st.text(style_names[10])
            st.image(resized11)
            
            img12 = Image.open(style_path[11])
            resized12 = img12.resize(img_size)
            st.text(style_names[11])
            st.image(resized12)
    


