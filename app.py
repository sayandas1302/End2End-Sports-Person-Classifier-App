import streamlit as st
import cv2 as cv
import tensorflow as tf
import numpy as np

face_classifier = cv.CascadeClassifier('opencv/haarcascade_face.xml')
model = tf.keras.models.load_model('./artifacts/model.h5')
celeb_dict = {
    0:{'Name':'Lionel Messi', 'Country':'Argentina', 'Field':'Football', 'image':'./Dataset/images_for_ui/messi.png'},
    1:{'Name':'Maria Sharapova', 'Country':'Russia', 'Field':'Tennis', 'image':'./Dataset/images_for_ui/sharapova.png' },
    2:{'Name':'Roger Federer', 'Country':'Switzerland', 'Field':'Tennis', 'image':'./Dataset/images_for_ui/federer.png'},
    3:{'Name':'Serena Williams', 'Country':'US', 'Field':'Tennis', 'image':'./Dataset/images_for_ui/serena.png'},
    4:{'Name':'Virat Kohli', 'Country':'India', 'Field':'Criket', 'image':'./Dataset/images_for_ui/virat.png'}
}

def detectFace(img):
    faces=[]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_rect = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,h,k) in face_rect:
        faces.append(img[y:y+k, x:x+h])
    return faces

# app background
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://c1.wallpaperflare.com/preview/914/23/709/person-ball-soccer-football.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# title
st.markdown('''
## Sports Person Classifier App

This app classifies images of **Lionel Messi**, **Maria Sharapova**, **Roger Federer**, **Serena Williams**, **Virat Kohli**. Just put any image of this 5 and see whosw image it is.
''')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.image('./Dataset/images_for_ui/messi.png', caption='Lionel Messi')

with col2:
    st.image('./Dataset/images_for_ui/sharapova.png', caption='Maria Sharapova')

with col3:
    st.image('./Dataset/images_for_ui/federer.png', caption='Roger Federer')

with col4:
    st.image('./Dataset/images_for_ui/serena.png', caption='Serana Williams')

with col5:
    st.image('./Dataset/images_for_ui/virat.png', caption='Virat Kohli')

# image input
st.markdown('### Upload your image here: ')
uploaded_image = st.file_uploader('', type=['jpg', 'png', 'jpeg', 'webp'], 
                                  accept_multiple_files=False)

button = st.button('Recognize')

if button:
    
    # saving the uploaded file 
    with open('./Dataset/temp/uploadedImage.jpg', 'wb') as file:
        file.write(uploaded_image.getbuffer())

    # getting the uploaded image as array
    img = cv.imread('./Dataset/temp/uploadedImage.jpg')
    
    # detecting and preprocessing faces 
    faces = detectFace(img)
    if len(faces)==0:
        st.markdown(f'### No faces found')
    else:
        preprocessed_faces = np.array([cv.resize(face, (224,224))/255 for face in faces])

        # recognizing faces
        result = model.predict(preprocessed_faces)
        st.markdown(f'### {len(result)} face(s) detected!') # printing the no of detected faces

        for counter in range(len(result)):
            predict_prob = np.max(result[counter]) # checking the matching probability
            
            col1, col2 = st.columns([1,4])
            
            if predict_prob>0.80: # showing player name with details if matching probability is more that 80%
                prediction = np.argmax(result[counter])
                info_dict = celeb_dict[prediction]

                with col1: 
                    st.image(info_dict['image'])
                with col2:
                    st.markdown(f'''
                    ## {info_dict['Name']} ({round(100*predict_prob,1)}% match) 
                    ##### Country: {info_dict['Country']}
                    ##### Event: {info_dict['Field']}''')
            else:  # showing 'face not detected message' if matching probability is more that 80%
                with col1: 
                    st.image('./Dataset/images_for_ui/no_dp.png')
                with col2:
                    st.markdown(f'### Sorry! Face can\'t be recognized ☹')