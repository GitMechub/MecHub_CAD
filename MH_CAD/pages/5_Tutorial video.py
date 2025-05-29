import streamlit as st
from streamlit_player import st_player
for k, v in st.session_state.items():
    st.session_state[k] = v

from PIL import Image
import os
path = os.path.dirname(__file__)
my_file = path+'/images/mechub_logo.png'
img = Image.open(my_file)

st.set_page_config(
    page_title='Tutorial - MH CAD',
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

st.header("Tutorial: Geometry From Coordinates v1.0.0", divider="gray", anchor=False)
st_player("https://youtu.be/SMm36mVPtyY")

st.header("Tutorial: Geometry From Image v1.0.0", divider="gray", anchor=False)
st_player("https://youtu.be/oMfN20RiWQQ")


