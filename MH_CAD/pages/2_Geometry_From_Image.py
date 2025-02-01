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


def process_coord_files(file, file_name):
    df, pontos = None, None

    if file_name.endswith('.xlsx'):
        try:
            excel_data = pd.read_excel(file, sheet_name=None)  # Load all sheets
            _sheet = list(excel_data.keys())[0]
            df = excel_data[_sheet]

            # Process columns
            if isinstance(list(df.columns)[0], str):
                df = df.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
            elif isinstance(list(df.columns)[0], (int, float)):
                df_part1 = pd.DataFrame({'x': [list(df.columns)[0]], 'y': [list(df.columns)[1]]})
                df_part2 = df.iloc[0:]
                df_part2 = df_part2.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
                df = pd.concat([pd.DataFrame(df_part1), df_part2], ignore_index=True)

            df = df.dropna().reset_index(drop=True)
            pontos = [(round(df['x'][i], 6), round(df['y'][i], 6)) for i in range(len(df['x']))]

        except Exception as e:
            st.error(f"Failed to read the Excel file: {str(e)}")

    elif file_name.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            column_data = list(df.columns)[0]

            try:
                column_data = float(column_data)
            except ValueError:
                pass

            # Process columns
            if isinstance(column_data, str):
                df = df.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
            elif isinstance(column_data, (int, float)):
                df_part1 = pd.DataFrame({'x': [list(df.columns)[0]], 'y': [list(df.columns)[1]]})
                df_part2 = df.iloc[0:]
                df_part2 = df_part2.rename(columns={list(df.columns)[0]: 'x', list(df.columns)[1]: 'y'})
                df = pd.concat([pd.DataFrame(df_part1), df_part2], ignore_index=True)

            df = df.dropna().reset_index(drop=True)
            pontos = [(round(float(df['x'][i]), 6), round(float(df['y'][i]), 6)) for i in range(len(df['x']))]

        except Exception as e:
            st.error(f"Failed to read the CSV file: {str(e)}")

    else:
        st.warning("Please upload a valid file in .xlsx or .csv format.")

    return df

@st.dialog("Uploading an Image File")
def show_uploading_recommendations():
    st.markdown("""
    - This app reads the external contour of images to create an extrusion.
    - The image file must be in **.jpg**, **.jpeg** or **.png** format.
    - The background must be **BLACK** or **WHITE**, contrasting with the image contour in the center.  
    - Higher image quality leads to better contour quality. 
    - Fewer colors and shadows in the image produce better results.  
    - **:red[See the example image below:]**
    """)
    st.image(path+'/images/example_mhcad_img.png')


def scale_contour_df(df, scale=1):
    df = df.copy().astype(float)
    #center = df[['x', 'y']].mean(axis=0)
    #scaled_df = (df[['x', 'y']] - center) * scale + center
    scaled_df = (df[['x', 'y']]) * scale
    df[['x', 'y']] = scaled_df
    return df


@st.dialog("Sketch Preview")
def show_sketch_preview_2(df, df2, scale=1):
    df = df.copy().astype(float).round(6)
    df2 = df2.copy().astype(float).round(6)

    # Garantir que o primeiro ponto do df seja igual ao último
    if not (df.iloc[0]['x'] == df.iloc[-1]['x'] and df.iloc[0]['y'] == df.iloc[-1]['y']):
        df = pd.concat([df, pd.DataFrame({'x': [df.iloc[0]['x']], 'y': [df.iloc[0]['y']]})], ignore_index=True)

    # Garantir que o primeiro ponto do df2 seja igual ao último
    if not (df2.iloc[0]['x'] == df2.iloc[-1]['x'] and df2.iloc[0]['y'] == df2.iloc[-1]['y']):
        df2 = pd.concat([df2, pd.DataFrame({'x': [df2.iloc[0]['x']], 'y': [df2.iloc[0]['y']]})], ignore_index=True)

    df = scale_contour_df(df, scale)
    df2 = scale_contour_df(df2, scale)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                             mode='lines',
                             name='Extrusion',
                             line=dict(color='black', width=1)))

    fig.add_trace(go.Scatter(x=df2['x'], y=df2['y'],
                             mode='lines',
                             name='Extruded Cut',
                             line=dict(color='red', width=1)))

    fig.update_layout(
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            scaleanchor="y"
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

@st.dialog("Sketch Preview ")
def show_sketch_preview(df, scale=1):
    df = df.copy().astype(float).round(6)
    df = scale_contour_df(df, scale)
    if not (df.iloc[0]['x'] == df.iloc[-1]['x'] and df.iloc[0]['y'] == df.iloc[-1]['y']):
        df = pd.concat([df, pd.DataFrame({'x': [df.iloc[0]['x']], 'y': [df.iloc[0]['y']]})], ignore_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['x'], y=df['y'],
                             mode='lines',
                             line=dict(color='black', width=1)))
    fig.update_layout(
        #xaxis_title='x',
        #yaxis_title='y',
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False,scaleanchor="y"),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    st.plotly_chart(fig, use_container_width=True)


def manter_extrude_button_ativo_img():
    st.session_state.extrude_button_img = True

def desativ_extrude_button_img():
    st.session_state.extrude_button_img = False

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


st.title("MecHub CAD v1.0.0", anchor=False)

st.subheader("Geometry From Image", divider="gray", anchor=False,
               help='Upload a .jpg, .jpeg or .png image to create an extrusion based on its external contour.')


col1, col2 = st.columns([1, 2])

if col1.button('Recommendations', use_container_width=True):
    show_uploading_recommendations()

#SETUP

if 'active_page_2' not in st.session_state:
    st.session_state.active_page_2 = '2_Geometry_From_Image'
    st.session_state.image_ok = False

    st.session_state.st_contours_to_remove = []
    st.session_state.st_add_contours_to_remove = []
    st.session_state.st_contours_ids = []

    st.session_state.st_try_smoother = True
    st.session_state.st_length_img = 5.

    st.session_state.st_contour_coordinates_ = []

    st.session_state.extrude_button_img = False

    st.session_state.st_sketch_img = None
    st.session_state.st_solid_img = None


def zerar_remov_contour():
    try:
        st.session_state['st_contours_ids'] = []
        st.session_state['st_contours_to_remove'] = []
    except:
        pass

## Uploading Image File
uploaded_file_img = col1.file_uploader('Extrude Contours From Images .jpg, .png or .jpeg',type=["jpg", "png", "jpeg"],on_change=zerar_remov_contour)
#uploaded_file_img = col1.file_uploader('Extrude Contours From Images .jpg, .png or .jpeg',type=["jpg", "png", "jpeg"])
if uploaded_file_img is not None:
    with st.spinner('Loading file...'):
        file_bytes_opencv = np.asarray(bytearray(uploaded_file_img.read()), dtype=np.uint8)
        file_img_name = str(uploaded_file_img.name)

        if file_img_name.endswith('.png') or file_img_name.endswith('.jpg') or file_img_name.endswith('.jpeg'):
            col1.image(uploaded_file_img)
            try:
                st.session_state['image_ok'] = True
            except:
                pass
        else:
            col1.error('Error: The file must be either .jpg, .png or .jpeg')
else:
    try:
        st.session_state['image_ok'] = False
        st.session_state['st_contours_ids'] = []
        st.session_state['st_contours_to_remove'] = []
    except:
        pass

##

## Extrusion Length

length_img = col1.number_input("Extrusion Length",format='%f',step=1.,help='For basic extrusion',key='st_length_img')

##

## Try Smoother?

try_smoother = col1.toggle("Try Smoother", key='st_try_smoother')

##


## Sketch Preview
if st.session_state['image_ok']:
    contour_coordinates_ = img_contour(file_bytes_opencv, st.session_state['st_contours_to_remove']) \
        if try_smoother is False else img_contour_smoother(file_bytes_opencv, st.session_state['st_contours_to_remove'])

    with col1:
        plot_all_contours_closed(contour_coordinates_)

##


## Remover Contornos Selecionados
if st.session_state.get('image_ok', False):
    disabled_contour_remov = False
else:
    disabled_contour_remov = True

valid_default_contours = list(set(st.session_state['st_contours_to_remove']) & set(st.session_state['st_contours_ids']))

contours_to_remove = col1.multiselect(
    "The contours you'd like to remove from the CAD",
    st.session_state['st_contours_ids'],
    key='st_contours_to_remove', disabled=disabled_contour_remov
)

##


## Run Button
if st.session_state['image_ok']:
    run_button = col2.button("Extrude",use_container_width = True)
else:
    run_button = False

if run_button:
    st.session_state.extrude_button_img = True


if st.session_state.extrude_button_img:
  # Saving the inputs
  try:
      st.session_state['st_contour_coordinates_'] = contour_coordinates_
  except:
      col2.error('Error: Contour Coordinates. Verify if the image was uploaded.')
      pass

  try:
      st.session_state['st_contours_to_remove'] = contours_to_remove
      st.session_state['st_try_smoother'] = try_smoother
      st.session_state['st_length_img'] = length_img
  except:
      pass


  with col2:

    with st.spinner('Loading...'):

        try:

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
                    #st.error("Error generating contour sketch")
                    continue

            ## Exportar o modelo como STL
            exporters.export(result, 'img_sketch_MHCAD.stl')

            ## 3D

            for contour in contour_coordinates_:

                if contour is None:
                    continue

                try:
                    solido1 = cq.Workplane("XY").polyline(contour).close().extrude(length_img)

                    try:
                        modelo_combinado = modelo_combinado.union(solido1)
                    except:
                        modelo_combinado = solido1

                except:
                    #st.error("Error generating solid")
                    continue

            ### Exportar como STL
            exporters.export(modelo_combinado, 'img_solid_MHCAD.stl')

            # Display
            try:
                stl_from_file(
                    file_path='img_solid_MHCAD.stl',
                    material='material',
                    auto_rotate=False,
                    opacity=1,
                    cam_h_angle=90,
                    height=610,
                    max_view_distance=100000,
                    color='#4169E1'
                )
                st.success("Sketch creation completed. Solid generated successfully.")
                st.session_state.st_sketch_img = result
                st.session_state.st_solid_img = modelo_combinado

            except:
                pass

            ############################### DOWNLOAD STL OR STEP

            st.divider()
            st.subheader("⬇️ Download", divider='gray', anchor=False)

            col21, col22 = col2.columns([1, 1])

            stl_file_sketch = str(path)[:-6] + "/" + 'img_sketch_MHCAD.stl'
            stl_file_solid = str(path)[:-6] + "/" + 'img_solid_MHCAD.stl'

            # Create a download button for STL
            ## Sketch
            col21.download_button(
                label="Sketch Surface .stl",
                #data=open(stl_file_sketch, "rb").read(),
                data=open('img_sketch_MHCAD.stl', "rb").read(),
                file_name='img_sketch_MHCAD.stl',
                mime="application/stl",
                on_click=manter_extrude_button_ativo_img,
                use_container_width=True
            )
            ## Solid
            col22.download_button(
                label="Solid .stl",
                #data=open(stl_file_solid, "rb").read(),
                data=open('img_solid_MHCAD.stl', "rb").read(),
                file_name='img_solid_MHCAD.stl',
                mime="application/stl",
                on_click=manter_extrude_button_ativo_img,
                use_container_width=True
            )

            with st.spinner('Creating .step File...'):
                exporters.export(result, 'img_sketch_MHCAD.step')
                exporters.export(modelo_combinado, 'img_solid_MHCAD.step')

                step_file_sketch = str(path)[:-6] + "/img_sketch_MHCAD.step"
                step_file_solid = str(path)[:-6] + "/img_solid_MHCAD.step"

                # Create a download button for STEP
                ## Sketch
                col21.download_button(
                    label="Sketch Surface .step",
                    #data=open(step_file_sketch, "rb").read(),
                    data=open('img_sketch_MHCAD.step', "rb").read(),
                    file_name='img_sketch_MHCAD.step',
                    mime="application/step",
                    on_click=manter_extrude_button_ativo_img,
                    use_container_width=True
                )
                ## Solid
                col22.download_button(
                    label="Solid .step",
                    #data=open(step_file_solid, "rb").read(),
                    data=open('img_solid_MHCAD.step', "rb").read(),
                    file_name='img_solid_MHCAD.step',
                    mime="application/step",
                    on_click=manter_extrude_button_ativo_img,
                    use_container_width=True
                )

            ############################### XLSX FILE

            try:

                file_name_img_sheet = 'img_contour_coordinates.xlsx'
                img_coord_dict = {
                    i: pd.DataFrame(coords, columns=['x', 'y'])
                    for i, coords in enumerate(contour_coordinates_)
                }

                buffer1 = io.BytesIO()
                with pd.ExcelWriter(buffer1, engine="xlsxwriter") as writer:

                    for contour in range(len(contour_coordinates_)):

                        try:
                            img_coord_ = pd.DataFrame(img_coord_dict[contour])
                            img_coord_.to_excel(writer, sheet_name='Contour ' + str(contour), index=False)
                            #img_coord_.to_excel(excel_writer, sheet_name='Contour ' + str(contour), index=False)
                        except Exception as e:
                            st.error(f"Error: Saving contour coordinates - {e}")
                            pass

                    writer.close()

                    col2.download_button(
                        label="Download Contour Coordinates",
                        data=buffer1,
                        file_name=file_name_img_sheet,
                        use_container_width=True,
                        on_click=manter_extrude_button_ativo_img,
                    )

            except Exception as e:
                st.error("Error: Generating the .xlsx coordinates file.")


        except Exception as e:
            st.error(f"Error in generating the geometry: {e}") # Keep
else:
  col2.markdown("")
