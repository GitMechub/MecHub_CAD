import streamlit as st
st.session_state.update(st.session_state)
for k, v in st.session_state.items():
    st.session_state[k] = v

from PIL import Image
import os
path = os.path.dirname(__file__)
my_file = path+'/images/mechub_logo.png'
img = Image.open(my_file)

st.set_page_config(
    page_title='MecHub CAD',
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


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cadquery as cq
from cadquery import exporters

import pyvista as pv
import plotly.graph_objects as go

import os  # Required to check if the file exists
from os import listdir
from os.path import isfile, join

import cv2  #OpenCV

import streamlit_stl
from streamlit_stl import stl_from_file

import io

#import openai
#from openai import OpenAI

from pydantic import BaseModel
from typing import List, Tuple

from google import genai

from google.genai import types
from PIL import Image
from io import BytesIO


################# FUNCTIONS #################

# * Processing "EXTRUDE GEOMETRY" Input:

def proc_EG_input(coordinates, extrusion_type, length, revolve_angle):

  revolve_axis = (0, 1, 0)  # * v1.0.0

  if isinstance(revolve_axis, tuple):
    revolve_axis = revolve_axis
  else:
    if revolve_axis.lower() == "x":
      revolve_axis = (1, 0, 0)
    elif revolve_axis.lower() == "y":
      revolve_axis = (0, 1, 0)
    elif revolve_axis.lower() == "z":
      revolve_axis = (0, 0, 1)
    else:
      revolve_axis = (0, 1, 0)

  if extrusion_type.lower() == "basic":
    extrusion_type = "basic"
  elif extrusion_type.lower() == "revolve":
    extrusion_type = "revolve"
  else:
    extrusion_type = "basic"

  if revolve_angle > 360:
    revolve_angle = 360
  elif revolve_angle < 0:
    revolve_angle = 0


  # Rounding a list of tuples
  ## Convert the list of tuples into a NumPy array
  coordinates = np.array(coordinates)
  ## Round all values in the array to 4 decimal places
  coordinates = np.round(coordinates, 4)
  ## Convert the rounded array back to a list of tuples directly using NumPy
  coordinates = list(map(tuple, coordinates))


  # Removing consecutive duplicate tuples
  ## Convert the list of tuples into a NumPy array
  coordinates_array = np.array(coordinates)
  ## Find the indices where consecutive rows are different
  unique_indices = np.any(coordinates_array[1:] != coordinates_array[:-1], axis=1)
  ## Include the first coordinate by prepending True
  unique_indices = np.insert(unique_indices, 0, True)
  ## Select only the unique rows
  unique_coordinates = coordinates_array[unique_indices]
  ## Convert back to a list of tuples if needed
  coordinates = list(map(tuple, unique_coordinates))

  return coordinates, extrusion_type, length, revolve_angle, revolve_axis


# * Function to check and reorder points to a consistent clockwise order

def reorder_clockwise(points):
    points = np.array(points)
    center = points.mean(axis=0)  # Calculate centroid
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    coordinates = points[sorted_indices].tolist()
    coordinates = list(map(tuple, coordinates))
    return coordinates


def scale_contour_df(df, scale=1):
    df = df.copy().astype(float)
    #center = df[['x', 'y']].mean(axis=0)
    #scaled_df = (df[['x', 'y']] - center) * scale + center
    scaled_df = (df[['x', 'y']]) * scale
    df[['x', 'y']] = scaled_df
    return df


def centralizar_conjunto_de_contornos(contornos):
    if not contornos:
        return []

    todos_os_pontos = [p for contorno in contornos for p in contorno]
    x_coords, y_coords = zip(*todos_os_pontos)
    x_centro = (max(x_coords) + min(x_coords)) / 2
    y_centro = (max(y_coords) + min(y_coords)) / 2

    contornos_centralizados = [
        [(x - x_centro, y - y_centro) for (x, y) in contorno]
        for contorno in contornos
    ]
    return contornos_centralizados


def revolve_centralize_y_multiple(contours):
    contornos_processados = []

    for contour in contours:
        novo_contorno = []

        for i in range(len(contour) - 1):
            p1 = contour[i]
            p2 = contour[i + 1]

            if p1[0] >= 0:
                novo_contorno.append(p1)

            if (p1[0] < 0 and p2[0] > 0) or (p1[0] > 0 and p2[0] < 0):
                x1, y1 = p1
                x2, y2 = p2
                t = -x1 / (x2 - x1)
                y_intersec = y1 + t * (y2 - y1)
                novo_contorno.append((0, y_intersec))

        # VERIFICA INTERSEÇÃO DO ÚLTIMO COM O PRIMEIRO
        p1 = contour[-1]
        p2 = contour[0]

        if p1[0] >= 0:
            novo_contorno.append(p1)

        if (p1[0] < 0 and p2[0] > 0) or (p1[0] > 0 and p2[0] < 0):
            x1, y1 = p1
            x2, y2 = p2
            t = -x1 / (x2 - x1)
            y_intersec = y1 + t * (y2 - y1)
            novo_contorno.append((0, y_intersec))

        contornos_processados.append(novo_contorno)

    return contornos_processados


##CONTOUR FROM IMAGE

def plot_contour(contour_id, contours):
  if contour_id < 0 or contour_id >= len(contours):
    print(f"Contour {contour_id} not found. Valid range is 0 to {len(contours) - 1}.")
    return

  # Get the specific contour coordinates
  contour = contours[contour_id]

  # Extract x and y coordinates separately
  x_coords = [point[0] for point in contour]
  y_coords = [point[1] for point in contour]

  # Create the plot
  fig = plt.figure(figsize=(10, 10))

  # Choose a color for the contour
  colors = ['r', 'g', 'b', 'c', 'm', 'orange', 'k']
  color = colors[contour_id % len(colors)]

  # Plot the contour
  plt.plot(x_coords, y_coords, color=color, marker='o', linestyle='-')

  plt.title(f'Contour {contour_id}, Points={len(contour)}')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.axis('equal')

  # Show the plot
  plt.gca().invert_yaxis()
  st.pyplot(fig)


def plot_all_contours(contours):
  fig = plt.figure(figsize=(10, 10))

  colors = ['r', 'g', 'b', 'c', 'm', 'orange', 'k']  # Define a palette of colors

  for contour_id, contour in enumerate(contours):
    try:
      # Extract x and y coordinates separately
      x_coords = [point[0] for point in contour]
      y_coords = [point[1] for point in contour]

      # Choose a color for the contour
      color = colors[contour_id % len(colors)]

      # Plot the contour on the same graph
      plt.plot(x_coords, y_coords, color=color, marker='o', linestyle='-',
               label=f'Contour {contour_id}, Points={len(contour)}')
    except:
      pass

  # Adding labels, grid, and legend
  plt.title('All Contours')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.axis('equal')
  plt.legend(loc='upper right')  # Add a legend to identify each contour

  # Show the plot
  plt.gca().invert_yaxis()
  st.pyplot(fig)


def plot_all_contours_closed(contours):
  fig = plt.figure(figsize=(10, 10))

  colors = ['r', 'g', 'b', 'c', 'm', 'orange', 'k']  # Define a palette of colors

  for contour_id, contour in enumerate(contours):
    try:
      # Extract x and y coordinates separately
      x_coords = [point[0] for point in contour]
      y_coords = [point[1] for point in contour]

      # Add the first point to the end to close the contour
      x_coords.append(x_coords[0])
      y_coords.append(y_coords[0])

      # Choose a color for the contour
      color = colors[contour_id % len(colors)]

      # Plot the contour on the same graph
      #plt.gca().invert_yaxis()
      plt.plot(x_coords, y_coords, color=color, marker='o', linestyle='-',
               label=f'Contour {contour_id}, Points={len(contour)}')
    except:
      pass

  # Adding labels, grid, and legend
  plt.title('All Contours')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.axis('equal')
  plt.legend(loc='upper right')  # Add a legend to identify each contour

  # Show the plot
  plt.gca().invert_yaxis()
  st.pyplot(fig)


def remove_contours(contour_coordinates, contours_to_remove):
  # Converte contours_to_remove para um conjunto para verificações mais rápidas
  indices_a_remover = set(contours_to_remove)

  # Substitui os índices a remover por None
  lista_resultante = [item if i not in indices_a_remover else None for i, item in enumerate(contour_coordinates)]
  return lista_resultante


def manter_extrude_button_ativo_prompt():
    st.session_state.extrude_button_prompt = True
    st.session_state.generate_trigger = True


def img_contour(img_name, contours_to_remove):

  #img_color = cv2.imread(img_name, cv2.IMREAD_COLOR)
  #img_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

  img_color = cv2.imdecode(img_name, cv2.IMREAD_COLOR)
  img_gray = cv2.imdecode(img_name, cv2.IMREAD_GRAYSCALE)

  corner_pixel = img_gray[0, 0]  # Coordenada (0, 0) para o canto superior esquerdo
  #print(f"Intensidade do pixel no canto superior esquerdo: {corner_pixel}")
  if corner_pixel > 50:
    img_gray = cv2.bitwise_not(img_gray)


  # Converting the grayscale image to binary (black and white only image).
  _, threshold = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)

  # Detecting external contours in the binary image.
  contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


  contour_coordinates = []

  for i, contour in enumerate(contours):
      cv2.drawContours(img_gray, [contour], -1, (0, 255, 0), 2)  # Green color for the contour
      #cv2.drawContours(img_color, [contour], -1, (0, 255, 0), 2)  # Green color for the contour
      contour_points = []
      for point in contour:
          x, y = point[0]  # Extracting x and y coordinates
          contour_points.append((int(x), int(y)))  # Convert to Python int
      contour_coordinates.append(contour_points)  # Add points of this contour to the list

  # Countour ID List:

  contour_coordinates_ = remove_contours(contour_coordinates, contours_to_remove) # Remove os contornos da lista

  indices_listas = [i for i, elemento in enumerate(contour_coordinates_) if isinstance(elemento, list)]

  try:
      if not st.session_state['st_contours_ids']:
          st.session_state['st_contours_ids'] = indices_listas
  except:
      pass

  return contour_coordinates_


def img_contour_smoother(img_name, contours_to_remove):

  #img_color = cv2.imread(img_name, cv2.IMREAD_COLOR)
  #img_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

  img_color = cv2.imdecode(img_name, cv2.IMREAD_COLOR)
  img_gray = cv2.imdecode(img_name, cv2.IMREAD_GRAYSCALE)

  corner_pixel = img_gray[0, 0]  # Coordenada (0, 0) para o canto superior esquerdo
  #print(f"Intensidade do pixel no canto superior esquerdo: {corner_pixel}")
  if corner_pixel > 50:
    img_gray = cv2.bitwise_not(img_gray)

  # Converting the grayscale image to binary (black and white only image).
  _, threshold = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)

  # Detecting external contours in the binary image.
  contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


  # Lista para armazenar as coordenadas suavizadas
  contour_coordinates = []

  # Processar os contornos
  for i, contour in enumerate(contours):
      # Suavizar o contorno usando approxPolyDP
      peri = cv2.arcLength(contour, True)
      smoothed_contour = cv2.approxPolyDP(contour, 0.001 * peri, True)

      # Adicionar as coordenadas suavizadas à lista
      contour_points = []
      for point in smoothed_contour:
          x, y = point[0]  # Extrair as coordenadas x e y
          contour_points.append((int(x), int(y)))  # Armazenar como inteiros
      contour_coordinates.append(contour_points)

  contour_coordinates_ = remove_contours(contour_coordinates, contours_to_remove) # Remove os contornos da lista

  indices_listas = [i for i, elemento in enumerate(contour_coordinates_) if isinstance(elemento, list)]

  try:
    if not st.session_state['st_contours_ids']:
        st.session_state['st_contours_ids'] = indices_listas
  except:
    pass

  return contour_coordinates_


############################################################


st.title("MecHub CAD v2.0.0", anchor=False)

st.subheader("Geometry From Coordinates", divider="gray", anchor=False,
               help='Set the geometry coordinates and extrude it by revolving through an axis or by basic extrusion. '
                    'You can do it manually or by uploading an .xlsx sheet with both "x" (Column A) and "y" (Column B) columns')


#SETUP

if 'image_ok' in st.session_state:
    try:
        st.session_state['extrude_button_img'] = False
    except:
        pass

if 'active_page_3' not in st.session_state:
    st.session_state.active_page_3 = '3_Geometry_From_AI_Prompt'

    st.session_state.st_img_bytes_prompt = ""
    st.session_state.st_contours_ids_prompt = []

    st.session_state.st_try_smoother_prompt = True

    st.session_state.st_contour_coordinates_prompt = []
    st.session_state.st_extrusion_type_prompt = "basic"
    st.session_state.st_length_prompt = 50.
    st.session_state.st_revolve_angle_prompt = 360.
    st.session_state.st_centralize_prompt = True
    st.session_state.st_twist_length_prompt = 500.
    st.session_state.st_twist_angle_prompt = 30.

    st.session_state.st_set_scale_prompt = False
    st.session_state.st_scale_prompt = 2.

    st.session_state.st_sketch_prompt = None
    st.session_state.st_solid_prompt = None

    st.session_state.st_message_id_list = []
    st.session_state.st_message_list = []

    st.session_state.st_api_key = ""


if "generate_trigger" not in st.session_state:
    st.session_state.generate_trigger = False

def set_generate_flag():
    st.session_state.generate_trigger = True


st.subheader('Extrude Geometry',help='Points must be listed in sequential order, where each point connects to the previous one.'
               ,anchor=False)


user_msg = st.chat_input("Say something")

with st.expander("Options", expanded=True):

    st.link_button("GOOGLE API KEYS", 'https://aistudio.google.com/apikey')

    api_key = st.text_input("API Key", key='st_api_key',type="password",on_change=set_generate_flag)

    extrusion_type = st.radio(
        "Extrusion Type",
        ["basic", "revolve","twist"],
        key='st_extrusion_type_prompt', help="Revolve axis = 'y'",on_change=set_generate_flag
    )

    if extrusion_type == "basic":
        length = st.number_input("Extrusion Length",format='%f',step=1.,key='st_length_prompt',on_change=set_generate_flag)

        revolve_angle = st.session_state['st_revolve_angle_prompt']
        centralize = st.session_state['st_centralize_prompt']
        twist_length = st.session_state['st_twist_length_prompt']
        twist_angle = st.session_state['st_twist_angle_prompt']

    elif extrusion_type == "revolve":
        revolve_angle = st.number_input("Revolve Angle °",format='%f',step=1.,key='st_revolve_angle_prompt',on_change=set_generate_flag)
        centralize = st.checkbox("Centralize", help = "Rotate around the center of the object",key='st_centralize_prompt',on_change=set_generate_flag)

        length = st.session_state['st_length_prompt']
        twist_length = st.session_state['st_twist_length_prompt']
        twist_angle = st.session_state['st_twist_angle_prompt']

    else:
        twist_length = st.number_input("Twist Extrusion Length", format='%f', step=1., key='st_twist_length_prompt',
                                       on_change=set_generate_flag)
        twist_angle = st.number_input("Twist Angle °",format='%f',step=1.,key='st_twist_angle_prompt',on_change=set_generate_flag)

        length = st.session_state['st_length_prompt']
        revolve_angle = st.session_state['st_revolve_angle_prompt']
        centralize = st.session_state['st_centralize_prompt']


    ## Set Scale Input
    set_scale = st.toggle("Optional: Set Scale", help='This option allows you to adjust the scale of the drawing contour that has been set.'
                                              , value=False,on_change=set_generate_flag)

    if not set_scale:
        set_scale = st.session_state['st_set_scale_prompt']
        scale = st.session_state['st_scale_prompt']
    else:
        scale = st.number_input("Scale",format='%f',step=0.5,min_value=0.01,key='st_scale_prompt',on_change=set_generate_flag)
    ##

    ## Try Smoother?

    try_smoother = st.toggle("Try Smoother", key='st_try_smoother',on_change=set_generate_flag)

    ##


# Control

if st.session_state['st_message_id_list'] != []:
    for i in range(len(st.session_state['st_message_id_list'])):
        if st.session_state['st_message_list'][i]:
            message_user = st.chat_message(st.session_state['st_message_id_list'][i]) if \
            st.session_state['st_message_id_list'][i] == 'assistant' else st.chat_message(
                st.session_state['st_message_id_list'][i])

            message_user.write(st.session_state['st_message_list'][i])

#

if api_key:
    client = genai.Client(api_key=api_key)

if user_msg or (
    st.session_state.get("generate_trigger") and
    st.session_state.get("st_message_list")
      ):

  # Saving the inputs
  try:
      st.session_state.st_api_key = str(api_key)
  except:
      pass

  message_user = st.chat_message("user")
  message_ai = st.chat_message("assistant")

  if user_msg:
    message_user.write(user_msg)

  with st.spinner('Wait for it...'):

        try:

            if user_msg:
                role = "Give the contour coordinates in python needed to CadQuery draw the object"
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=f'A 2D silhouette of {user_msg} on a white background, perspective view, no shadows',
                    config=types.GenerateContentConfig(
                      response_modalities=['TEXT', 'IMAGE']
                    )
                )

                message_ai.write("Result:")


                for part in response.candidates[0].content.parts:
                    if part.text is not None:
                        print(part.text)
                    elif part.inline_data is not None:
                        #image = Image.open(BytesIO((part.inline_data.data)))
                        #image.save('user_msg.png')

                        #st.session_state['st_img_bytes_prompt'] = image

                        # A imagem já vem como bytes — você só precisa armazenar isso
                        img_bytes = part.inline_data.data  # Isso já é do tipo `bytes`

                        # Para uso com OpenCV
                        file_bytes_opencv = np.asarray(bytearray(img_bytes), dtype=np.uint8)

                        # Armazena diretamente os bytes na sessão
                        st.session_state['st_img_bytes_prompt'] = file_bytes_opencv

            try:
                img_bytes = st.session_state['st_img_bytes_prompt']
                contour_coordinates_ = img_contour("user_msg.png", []) \
                    if try_smoother is False else img_contour_smoother("user_msg.png",
                                                                       [])
            except:
                contour_coordinates_ = img_contour(img_bytes, []) \
                    if try_smoother is False else img_contour_smoother(img_bytes,
                                                                       [])

            if extrusion_type == "revolve" and centralize:
                contour_coordinates_ = centralizar_conjunto_de_contornos(contour_coordinates_)
                contour_coordinates_ = revolve_centralize_y_multiple(contour_coordinates_)
            elif extrusion_type == "twist":
                contour_coordinates_ = centralizar_conjunto_de_contornos(contour_coordinates_)


            #start_axis, end_axis = calcular_eixo_y_para_conjunto_de_contornos(contour_coordinates_)
            with st.expander('Sketch'):
                plot_all_contours_closed(contour_coordinates_)
                #st.image("user_msg.png",width=256)
                st.image(BytesIO(img_bytes), width=256)


            st.session_state.st_message_id_list = st.session_state['st_message_id_list'] + ["user"]
            st.session_state.st_message_list = st.session_state['st_message_list'] + [user_msg]


            ################# RUNNING #################

            ############################### GEOMETRY

            result = None
            modelo_combinado = None

            for contour in contour_coordinates_:

                if contour is None:
                    continue

                ## Criar o esboço inicial
                sketch1 = cq.Sketch()

                try:
                    for i in range(len(contour) - 1):
                        sketch1 = sketch1.segment(contour[i], contour[i + 1])

                    sketch1 = sketch1.close().assemble(tag="face").reset()

                    try:
                        result = result + sketch1
                    except:
                        result = sketch1
                except:
                    # st.error("Error generating contour sketch")
                    continue

            ## Exportar o modelo como STL
            exporters.export(result, 'ai_sketch_MHCAD.stl')


            ## 3D
            #st.write(contour_coordinates_)
            modelo_combinado = cq.Workplane("XY").placeSketch(result).revolve(angleDegrees=revolve_angle, axisStart=(0,0,0),
                                   axisEnd=(0,1,0)) if extrusion_type == "revolve" else cq.Workplane("XY").placeSketch(result).twistExtrude(twist_length, twist_angle) if extrusion_type == "twist" else cq.Workplane("XY").placeSketch(result).extrude(length)


            ### Exportar como STL
            exporters.export(modelo_combinado, 'ai_solid_MHCAD.stl')

            # Display
            try:
                stl_from_file(
                    file_path='ai_solid_MHCAD.stl',
                    material='material',
                    auto_rotate=False,
                    opacity=1,
                    cam_h_angle=90,
                    height=610,
                    max_view_distance=100000,
                    color='#4169E1'
                )
                st.success("Sketch creation completed. Solid generated successfully.")
                st.session_state.st_sketch_prompt = result
                st.session_state.st_solid_prompt = modelo_combinado

            except:
                pass

            st.session_state.generate_trigger = False

            ############################### DOWNLOAD STL OR STEP

            st.divider()
            st.subheader("⬇️ Download", divider='gray', anchor=False)

            col21, col22 = st.columns([1, 1])

            stl_file_sketch = str(path)[:-6] + "/" + 'ai_sketch_MHCAD.stl'
            stl_file_solid = str(path)[:-6] + "/" + 'ai_solid_MHCAD.stl'

            # Create a download button for STL
            ## Sketch
            col21.download_button(
                label="Sketch Surface .stl",
                # data=open(stl_file_sketch, "rb").read(),
                data=open('ai_sketch_MHCAD.stl', "rb").read(),
                file_name='ai_sketch_MHCAD.stl',
                mime="application/stl",
                on_click=manter_extrude_button_ativo_prompt,
                use_container_width=True
            )
            ## Solid
            col22.download_button(
                label="Solid .stl",
                # data=open(stl_file_solid, "rb").read(),
                data=open('ai_solid_MHCAD.stl', "rb").read(),
                file_name='ai_solid_MHCAD.stl',
                mime="application/stl",
                on_click=manter_extrude_button_ativo_prompt,
                use_container_width=True
            )

            with st.spinner('Creating .step File...'):
                exporters.export(result, 'ai_sketch_MHCAD.step')
                exporters.export(modelo_combinado, 'ai_solid_MHCAD.step')

                step_file_sketch = str(path)[:-6] + "/ai_sketch_MHCAD.step"
                step_file_solid = str(path)[:-6] + "/ai_solid_MHCAD.step"

                # Create a download button for STEP
                ## Sketch
                col21.download_button(
                    label="Sketch Surface .step",
                    # data=open(step_file_sketch, "rb").read(),
                    data=open('ai_sketch_MHCAD.step', "rb").read(),
                    file_name='ai_sketch_MHCAD.step',
                    mime="application/step",
                    on_click=manter_extrude_button_ativo_prompt,
                    use_container_width=True
                )
                ## Solid
                col22.download_button(
                    label="Solid .step",
                    # data=open(step_file_solid, "rb").read(),
                    data=open('ai_solid_MHCAD.step', "rb").read(),
                    file_name='ai_solid_MHCAD.step',
                    mime="application/step",
                    on_click=manter_extrude_button_ativo_prompt,
                    use_container_width=True
                )

            ############################### XLSX FILE

            try:

                file_name_ai_sheet = 'ai_contour_coordinates.xlsx'
                ai_coord_dict = {
                    i: pd.DataFrame(coords, columns=['x', 'y'])
                    for i, coords in enumerate(contour_coordinates_)
                }

                buffer1 = io.BytesIO()
                with pd.ExcelWriter(buffer1, engine="xlsxwriter") as writer:

                    for contour in range(len(contour_coordinates_)):

                        try:
                            ai_coord_ = pd.DataFrame(ai_coord_dict[contour])
                            ai_coord_.to_excel(writer, sheet_name='Contour ' + str(contour), index=False)
                            # ai_coord_.to_excel(excel_writer, sheet_name='Contour ' + str(contour), index=False)
                        except Exception as e:
                            st.error(f"Error: Saving contour coordinates - {e}")
                            pass

                    writer.close()

                    st.download_button(
                        label="Download Contour Coordinates",
                        data=buffer1,
                        file_name=file_name_ai_sheet,
                        use_container_width=True,
                        on_click=manter_extrude_button_ativo_prompt,
                    )

            except Exception as e:
                st.error("Error: Generating the .xlsx coordinates file.")


        except Exception as e:
            st.error(f"Error in generating the geometry: {e}")  # Keep

