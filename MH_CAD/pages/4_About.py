import streamlit as st
for k, v in st.session_state.items():
    st.session_state[k] = v

from PIL import Image
import os
path = os.path.dirname(__file__)
my_file = path+'/images/mechub_logo.png'
img = Image.open(my_file)

st.set_page_config(
    page_title='About - MH CAD',
    layout="wide",
    page_icon=img
                   )

st.sidebar.image(img)
st.sidebar.markdown("[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Mechub?sub_confirmation=1) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GitMechub)")

hide_menu = '''
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        '''
st.markdown(hide_menu, unsafe_allow_html=True)

if 'image_ok' in st.session_state:
    try:
        st.session_state['extrude_button_img'] = False
    except:
        pass

st.header("MecHub CAD v2.0.0", divider="gray", anchor=False)

st.markdown('''
## About MecHub CAD:

MecHub CAD is a simple application designed to help engineers, designers, and hobbyists transform coordinate data, images, or AI chatbot messages into 3D models through extrusion.

## How it Works:

MecHub CAD offers three primary functions:

1. **Coordinate-Based 3D Extrusion:**  
   Upload a file containing a series of (x, y) coordinates representing a contour. MecHub CAD will extrude this contour into a fully defined 3D shape, ready for CAD modeling and further processing.

2. **Image-Based 3D Extrusion:**  
   Upload a .jpg, .jpeg, or .png image containing an external contour. MecHub CAD will process the image and extract the contour, generating a corresponding 3D model through extrusion.

3. **Prompt-Based 3D Extrusion via AI Chatbot:** **(NEW):**   
   Enter a prompt into the AI chatbot powered by Google Gemini. MecHub CAD will interpret the message and automatically generate a 3D model based on the described object contour.

## Key Features:

- **Coordinate File Support:** Compatible with .xlsx and .csv formats for contour generation.
- **Image Contour Extraction:** Supports common image formats (.jpg, .jpeg, .png) for smooth contour identification.
- **(NEW)****Prompt-to-Model AI Integration:** Create 3D models directly from descriptive text prompts using Google Gemini.  
- **Precise Extrusion:** Easily generate complex 3D models from contours.
- **CAD Integration:** Export clean, ready-to-use models for CAD software.
- **User-Friendly Interface:** Simplifies the transformation from 2D to 3D with minimal input requirements.

MecHub CAD makes it simple to turn 2D representations, whether from coordinate files, image contours, or AI chatbot prompts, into 3D models through intuitive extrusion tools.

  ---
  '''
            )



path2 = os.path.dirname(__file__)
my_file2 = path2+'/images/Thumbs Mechub_MHCAD.png'
img2 = Image.open(my_file2)

st.image(img2)
