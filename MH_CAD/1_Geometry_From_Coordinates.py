import streamlit as st

st.session_state.update(st.session_state)
for k, v in st.session_state.items():
    st.session_state[k] = v

from PIL import Image
import os

path = os.path.dirname(__file__)
my_file = path + '/pages/images/mechub_logo.png'
img = Image.open(my_file)

st.set_page_config(
    page_title='MecHub CAD',
    layout="wide",
    page_icon=img
)

st.sidebar.image(img)
st.sidebar.markdown(
    "[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Mechub?sub_confirmation=1) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GitMechub)")

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

import cv2  # OpenCV

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
    elif extrusion_type.lower() == "twist":
        extrusion_type = "twist"
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

    # Remove last point if it's the same as the first (for Sketch compatibility)
    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates = coordinates[:-1]

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
    plt.figure(figsize=(10, 10))

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
    plt.show()


def plot_all_contours(contours):
    plt.figure(figsize=(10, 10))

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
    plt.show()


def plot_all_contours_closed(contours):
    plt.figure(figsize=(10, 10))

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
            plt.gca().invert_yaxis()
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
    plt.show()


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


def centralizar_contorno(contorno):
    if not contorno:
        return []

    x_coords, y_coords = zip(*contorno)
    x_centro = (max(x_coords) + min(x_coords)) / 2
    y_centro = (max(y_coords) + min(y_coords)) / 2

    contorno_centralizado = [(x - x_centro, y - y_centro) for (x, y) in contorno]
    return contorno_centralizado


def revolve_centralize_y(contour):
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

    # Último ponto
    if contour[-1][0] >= 0:
        novo_contorno.append(contour[-1])

    return novo_contorno


@st.dialog("Uploading a Coordinates File")
def show_uploading_instructions():
    st.markdown("""
    - The file must be saved in either **.xlsx** or **.csv** format.  
    - For **.xlsx** files, column A represents the x-axis, while column B represents the y-axis.  
    - For **.xlsx** files, decimal values should use a `.` as the separator, not a `,`.  
    - For **.csv** files, the first value represents the x-axis, and the second value represents the y-axis.  
    - **Coordinates in both formats must be in order, with each (x, y) point located between the previous and next points.**
    """)
    st.image(path + '/pages/images/example_mhcad.png')


def scale_contour_df(df, scale=1):
    df = df.copy().astype(float)
    # center = df[['x', 'y']].mean(axis=0)
    # scaled_df = (df[['x', 'y']] - center) * scale + center
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
        # xaxis_title='x',
        # yaxis_title='y',
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, scaleanchor="y"),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    st.plotly_chart(fig, use_container_width=True)


def manter_extrude_button_ativo():
    st.session_state.extrude_button = True


def desativ_extrude_button():
    st.session_state.extrude_button = False


############################################################


st.title("MecHub CAD v1.0.0", anchor=False)

st.subheader("Geometry From Coordinates", divider="gray", anchor=False,
             help='Set the geometry coordinates and extrude it by revolving through an axis or by basic extrusion. '
                  'You can do it manually or by uploading an .xlsx sheet with both "x" (Column A) and "y" (Column B) columns')

col1, col2 = st.columns([1, 2])

# SETUP

if 'image_ok' in st.session_state:
    try:
        st.session_state['extrude_button_img'] = False
    except:
        pass

if 'active_page' not in st.session_state:
    st.session_state.active_page = '1_Geometry_From_Coordinates'

    st.session_state.st_coordinates_x = [0., 0., 4., 4.]
    st.session_state.st_coordinates_y = [0., 4., 4., 0.]
    st.session_state.st_coordinates = [(0, 0), (0, 4), (4, 4), (4, 0)]
    st.session_state.st_upload_coordinates = False
    st.session_state.st_extrusion_type = "basic"
    st.session_state.st_length = 4.
    st.session_state.st_revolve_angle = 360.
    st.session_state.st_centralize = False
    st.session_state.st_twist_length = 4.
    st.session_state.st_twist_angle = 30.

    st.session_state.st_coordinates_cut = [(1, 1), (1, 3), (3, 3), (3, 1)]
    st.session_state.st_coordinates_x_cut = [1., 1., 3., 3.]
    st.session_state.st_coordinates_y_cut = [1., 3., 3., 1.]
    st.session_state.st_upload_coordinates_cut = False
    st.session_state.st_extruded_cut = False
    st.session_state.st_length_cut = 6.
    st.session_state.st_revolve_angle_cut = 360.
    st.session_state.st_centralize_cut = False
    st.session_state.st_twist_length_cut = 4.
    st.session_state.st_twist_angle_cut = 30.

    st.session_state.st_set_scale = False
    st.session_state.st_scale = 2.

    st.session_state.extrude_button = False

    st.session_state.st_sketch = None
    st.session_state.st_solid = None

col1.subheader('Extrude Geometry',
               help='Points must be listed in sequential order, where each point connects to the previous one.'
               , anchor=False)

upload_coordinates = col1.toggle("Upload Coordinates File",
                                 help='If you prefer to upload a file instead of setting the surface coordinates manually.'
                                      ' The points must be ordered.', key='st_upload_coordinates',
                                 on_change=desativ_extrude_button)

if not upload_coordinates:
    col1.write('Surface Coordinates (x, y)')
    df__coord = pd.DataFrame({'x': st.session_state['st_coordinates_x'], 'y': st.session_state['st_coordinates_y']})
    df__coord_input = col1.data_editor(df__coord, num_rows="dynamic",
                                       column_config={
                                           "x": st.column_config.NumberColumn(
                                               format="%f"
                                           ),
                                           "y": st.column_config.NumberColumn(
                                               format="%f"
                                           )})

else:
    if col1.button('Instructions', use_container_width=True):
        show_uploading_instructions()

    uploaded_file = col1.file_uploader('The file must be in either **.xlsx** or **.csv** format.', type=["xlsx", "csv"])
    if uploaded_file is not None:
        with st.spinner('Loading file...'):
            df__coord_input = process_coord_files(uploaded_file, str(uploaded_file.name))
            col1.dataframe(df__coord_input)

extrusion_type = col1.radio(
    "Extrusion Type",
    ["basic", "revolve", "twist"],
    key='st_extrusion_type', help="Revolve axis = 'y'"
)

if extrusion_type == "basic":
    length = col1.number_input("Extrusion Length", format='%f', step=1., key='st_length')

    revolve_angle = st.session_state['st_revolve_angle']
    centralize = st.session_state['st_centralize']
    twist_length = st.session_state['st_twist_length']
    twist_angle = st.session_state['st_twist_angle']

elif extrusion_type == "revolve":
    revolve_angle = col1.number_input("Revolve Angle °", format='%f', step=1., key='st_revolve_angle')
    centralize = col1.checkbox("Centralize", help="Rotate around the center of the object", key='st_centralize')

    length = st.session_state['st_length']
    twist_length = st.session_state['st_twist_length']
    twist_angle = st.session_state['st_twist_angle']

else:
    twist_length = col1.number_input("Twist Extrusion Length", format='%f', step=1., key='st_twist_length')
    twist_angle = col1.number_input("Twist Angle °", format='%f', step=1., key='st_twist_angle')

    length = st.session_state['st_length']
    revolve_angle = st.session_state['st_revolve_angle']
    centralize = st.session_state['st_centralize']

## Set Scale Input
set_scale = col1.toggle("Optional: Set Scale",
                        help='This option allows you to adjust the scale of the drawing contour that has been set.'
                        , value=False)

if not set_scale:
    set_scale = st.session_state['st_set_scale']
    scale = st.session_state['st_scale']
else:
    scale = col1.number_input("Scale", format='%f', step=0.5, min_value=0.01, key='st_scale')
##

## Extruded Cut Input
extruded_cut = col1.toggle("Optional: Extruded Cut",
                           help='If you want to perform an Extruded Cut, activate this option.'
                           , value=False, on_change=desativ_extrude_button)

if not extruded_cut:
    coordinates_cut = st.session_state['st_coordinates_cut']
    length_cut = st.session_state['st_length_cut']
    revolve_angle_cut = st.session_state['st_revolve_angle_cut']
    centralize_cut = st.session_state['st_centralize_cut']
    twist_length_cut = st.session_state['st_twist_length_cut']
    twist_angle_cut = st.session_state['st_twist_angle_cut']
else:
    col1.subheader('Extruded Cut',
                   help='Points must be listed in sequential order, where each point connects to the previous one.'
                   , anchor=False)

    upload_coordinates_cut = col1.toggle("Upload Coordinates File",
                                         help='If you prefer to upload a file instead of setting the surface coordinates manually.'
                                              ' The points must be ordered.', value=False,
                                         on_change=desativ_extrude_button)

    if not upload_coordinates_cut:
        col1.write('Surface Coordinates (x, y)')
        df__coord_cut = pd.DataFrame(
            {'x': st.session_state['st_coordinates_x_cut'], 'y': st.session_state['st_coordinates_y_cut']})
        df__coord_input_cut = col1.data_editor(df__coord_cut, num_rows="dynamic",
                                               column_config={
                                                   "x": st.column_config.NumberColumn(
                                                       format="%f"
                                                   ),
                                                   "y": st.column_config.NumberColumn(
                                                       format="%f"
                                                   )}, key=None)


    else:
        if col1.button('Instructions ', use_container_width=True):
            show_uploading_instructions()

        uploaded_file = col1.file_uploader('The file must be in either **.xlsx** or **.csv** format. ',
                                           type=["xlsx", "csv"])

        if uploaded_file is not None:
            with st.spinner('Loading file...'):
                df__coord_input_cut = process_coord_files(uploaded_file, str(uploaded_file.name))

                col1.dataframe(df__coord_input_cut)

    if extrusion_type == "basic":
        length_cut = col1.number_input("Extrusion Length", format='%f', step=1., key='st_length_cut')

        revolve_angle_cut = st.session_state['st_revolve_angle_cut']
        centralize_cut = st.session_state['st_centralize_cut']
        twist_length_cut = st.session_state['st_twist_length_cut']
        twist_angle_cut = st.session_state['st_twist_angle_cut']

    elif extrusion_type == "revolve":
        revolve_angle_cut = col1.number_input("Revolve Angle °", format='%f', step=1., key='st_revolve_angle_cut')
        #centralize_cut = col1.checkbox("Centralize", help="Rotate around the center of the object", key='st_centralize_cut')

        length_cut = st.session_state['st_length_cut']
        twist_length_cut = st.session_state['st_twist_length_cut']
        twist_angle_cut = st.session_state['st_twist_angle_cut']

    else:
        twist_length_cut = col1.number_input("Twist Extrusion Length", format='%f', step=1., key='st_twist_length_cut')
        twist_angle_cut = col1.number_input("Twist Angle °", format='%f', step=1., key='st_twist_angle_cut')

        length_cut = st.session_state['st_length_cut']
        revolve_angle_cut = st.session_state['st_revolve_angle_cut']
        centralize_cut = st.session_state['st_centralize_cut']

##

## Sketch Preview


preview_button = col1.button("Sketch Preview", use_container_width=True, on_click=desativ_extrude_button)

if preview_button:
    try:
        scale_ = 1 if not set_scale else scale
        show_sketch_preview_2(df__coord_input, df__coord_input_cut, scale_) if extruded_cut else show_sketch_preview(
            df__coord_input, scale_)
    except Exception as e:
        col1.error(f"Error in generating the surface: {e}")

##


run_button = col2.button("Extrude", use_container_width=True)

if run_button:
    st.session_state.extrude_button = True

if st.session_state.extrude_button:
    # Saving the inputs
    ## For coord's df
    try:
        st.session_state['st_coordinates_x'] = df__coord_input['x']
        st.session_state['st_coordinates_y'] = df__coord_input['y']
    except:
        col2.error('Error: Surface Coordinates.')
        pass
    try:
        st.session_state['st_coordinates_x_cut'] = df__coord_input_cut['x']
        st.session_state['st_coordinates_y_cut'] = df__coord_input_cut['y']
    except:
        pass

    with col2:

        with st.spinner('Loading...'):

            try:

                ################# RUNNING #################

                coordinates = [(round(float(df__coord_input['x'][i]), 6), round(float(df__coord_input['y'][i]), 6)) for
                               i in range(len(df__coord_input['x']))]
                try:
                    coordinates_cut = [
                        (round(float(df__coord_input_cut['x'][i]), 6), round(float(df__coord_input_cut['y'][i]), 6)) for
                        i in range(len(df__coord_input_cut['x']))]
                except:
                    pass

                coordinates, extrusion_type, length, revolve_angle, revolve_axis = proc_EG_input(coordinates,
                                                                                                 extrusion_type,
                                                                                                 length, revolve_angle)
                extrusion_type_cut = extrusion_type
                coordinates_cut, extrusion_type_cut, length_cut, revolve_angle_cut, revolve_axis_cut = proc_EG_input(
                    coordinates_cut, extrusion_type_cut, length_cut, revolve_angle_cut)

                
                centralize_cut = centralize


                if extrusion_type == "revolve" and centralize:
                    coordinates = centralizar_contorno(coordinates)
                    coordinates = revolve_centralize_y(coordinates)
                elif extrusion_type == "twist":
                    coordinates = centralizar_contorno(coordinates)

                try:
                    if extrusion_type == "revolve" and centralize_cut:
                        coordinates_cut = centralizar_contorno(coordinates_cut)
                        coordinates_cut = revolve_centralize_y(coordinates_cut)
                    elif extrusion_type == "twist":
                        coordinates_cut = centralizar_contorno(coordinates_cut)
                except:
                    pass


                ############################### FUNCTIONS (CODE)

                def extruded_cut_geometry(coordinates_cut, coordinates, solid, s_surface, length_cut, twist_length_cut, twist_angle_cut,
                                          extrusion_type_cut="basic", revolve_angle_cut=360, revolve_angle=360,
                                          revolve_axis_cut=(0, 1, 0), extruded_cut=False, set_scale=False, scale=1):

                    if os.path.exists('final_sketch_MHCAD_cut.stl'):
                        os.remove('final_sketch_MHCAD_cut.stl')
                        print(f"Existing file 'final_sketch_MHCAD_cut.stl' removed.")

                    if os.path.exists('final_solid_MHCAD_cut.stl'):
                        os.remove('final_solid_MHCAD_cut.stl')
                        print(f"Existing file 'final_solid_MHCAD_cut.stl' removed.")

                    if extruded_cut is True:
                        try:

                            # Sketch

                            ## Add segments between the points
                            sketch = cq.Sketch()

                            try:
                                for i in range(len(coordinates_cut) - 1):
                                    sketch = sketch.segment(coordinates_cut[i], coordinates_cut[i + 1])
                            except:  # Trying to reorder if error
                                coordinates_cut = reorder_clockwise(coordinates_cut)
                                for i in range(len(coordinates_cut) - 1):
                                    sketch = sketch.segment(coordinates_cut[i], coordinates_cut[i + 1])

                                ### If the coordinates for the extrude cut are the same as those for the solid
                                if coordinates_cut == coordinates and extruded_cut == True:
                                    st.error(
                                        "Error: The coordinates for the extrude cut are the same as those for the solid")
                                    extruded_cut = False
                                    return 'sketch_MHCAD.stl', 'solid_MHCAD.stl', solid

                            ## Close the sketch, if necessary
                            s_surface_cut = sketch.close().assemble(tag="face_")

                            if set_scale is True:
                                try:
                                    s_surface_cut = s_surface_cut.val().scale(scale)
                                except:
                                    st.error("Error during scaling - Extruded cut")
                                    pass

                            ## Result (Sketch)
                            result_sketch = s_surface - s_surface_cut

                            ## Export as STL
                            sketch_name = 'final_sketch_MHCAD_cut.stl'
                            exporters.export(result_sketch, sketch_name)

                            # 3D

                            ## Create the sketch in CadQuery
                            profile = [(x, y) for x, y in coordinates_cut]  # Swap to (r, z) for the XY plane
                            # sketch = cq.Workplane("XY").polyline(profile).close()  # Close the profile

                            sketch = cq.Workplane("XY").placeSketch(s_surface_cut)

                            if set_scale is True:
                                length_cut = length_cut * scale
                                twist_length_cut = twist_length_cut * scale

                            ## Extrude or Revolve around the axis (Y is the default axis for revolve!)
                            solid_cut = sketch.revolve(angleDegrees=revolve_angle_cut, axisStart=(0, 0, 0),
                                                       axisEnd=revolve_axis_cut) if extrusion_type_cut == "revolve" else sketch.twistExtrude(
                                twist_length_cut, twist_angle_cut) if extrusion_type_cut == "twist" else sketch.extrude(length_cut)

                            solid = solid.cut(solid_cut)

                            ## Export as STL
                            solid_name = 'final_solid_MHCAD_cut.stl'
                            exporters.export(solid, solid_name)

                            return sketch_name, solid_name, solid

                        except:
                            st.error("Error: Extruded Cut")
                            return 'sketch_MHCAD.stl', 'solid_MHCAD.stl', solid

                    else:
                        return 'sketch_MHCAD.stl', 'solid_MHCAD.stl', solid


                def create_and_export_sketch(coordinates, set_scale, scale, filename='sketch_MHCAD.stl'):

                    def attempt_sketch(coords, set_scale=False, scale=1):

                        try:
                            sketch = cq.Sketch()
                            for i in range(len(coords) - 1):
                                sketch = sketch.segment(coords[i], coords[i + 1])
                            s_surface = sketch.close().assemble(tag="face")
                            if set_scale is True:
                                try:
                                    s_surface = s_surface.val().scale(scale)
                                except:
                                    st.error("Error during scaling")
                                    pass
                            exporters.export(s_surface, filename)
                            if os.path.exists(filename):
                                return s_surface
                            return None
                        except Exception as e:
                            st.error(f"Error during sketch attempt: {e}")
                            return None

                    # Step 1: Remove the file if it already exists
                    if os.path.exists(filename):
                        try:
                            os.remove(filename)
                            print(f"Existing file '{filename}' removed.")
                        except Exception as e:
                            print(f"Error removing file '{filename}': {e}")
                            return coordinates, None  # Return original coordinates if cleanup fails

                    # Attempt 1: Use the original coordinates
                    print("Attempting to create sketch with original coordinates...")
                    s_surface = attempt_sketch(coordinates, set_scale, scale)
                    if s_surface:
                        # st.success("File successfully created on the first attempt!")
                        return coordinates, s_surface

                    # Attempt 2: Reorder the coordinates clockwise
                    try:
                        st.error("Reordering coordinates clockwise and retrying...")
                        coordinates_reordered = reorder_clockwise(coordinates)
                        s_surface = attempt_sketch(coordinates_reordered, set_scale, scale)
                        if s_surface:
                            # st.success("File successfully created after reordering!")
                            return coordinates_reordered, s_surface
                    except Exception as e:
                        st.error(f"Reordering failed: {e}")

                    # Attempt 3: Adjust coordinates by adding (0,0) and the last x-value point
                    try:
                        # st.warning("Adjusting coordinates by adding (0,0) and the last x-value...")
                        coordinates.insert(0, (0, 0))
                        coordinates.append((coordinates[-1][0], 0))  # Use x-value of the last point
                        coordinates_reordered = reorder_clockwise(coordinates)
                        s_surface = attempt_sketch(coordinates_reordered, set_scale, scale)
                        if s_surface:
                            # st.success("File successfully created after adjusting points!")
                            return coordinates_reordered, s_surface
                    except Exception as e:
                        st.error(f"Adjusting points failed: {e}")

                    # If all attempts fail
                    st.error("Error: Unable to create the sketch after all attempts.")
                    return coordinates, None


                ############################### RESULTS

                coordinates_, s_surface = create_and_export_sketch(coordinates, set_scale, scale)

                if s_surface:

                    try:

                        # st.success("Sketch creation completed. Surface generated successfully.")

                        sketch = cq.Workplane("XY").placeSketch(s_surface)

                        ## Create the sketch in CadQuery
                        # profile = [(x, y) for x, y in coordinates]  # Swap to (r, z) for the XY plane
                        # print(profile)
                        # sketch = cq.Workplane("XY").polyline(profile).close()  # Close the profile'''

                        if set_scale:
                            length_ = length * scale
                            twist_length_ = twist_length * scale
                        else:
                            length_ = length
                            twist_length_ = twist_length

                        ## Extrude or Revolve around the axis (Y is the default axis for revolve!)
                        solid = sketch.revolve(angleDegrees=revolve_angle, axisStart=(0, 0, 0),
                                               axisEnd=revolve_axis) if extrusion_type == "revolve" else sketch.twistExtrude(
                            twist_length_, twist_angle) if extrusion_type == "twist" else sketch.extrude(length_)


                        ## Export as STL
                        exporters.export(solid, 'solid_MHCAD.stl')

                        # Extruded Cut

                        sketch_name, solid_name, solid = extruded_cut_geometry(coordinates_cut, coordinates_, solid,
                                                                               s_surface, length_cut, twist_length_cut, twist_angle_cut,
                                                                               extrusion_type_cut, revolve_angle_cut,
                                                                               revolve_angle, revolve_axis_cut,
                                                                               extruded_cut, set_scale, scale)

                    except Exception as e:
                        st.error(
                            f"Error creating extrusion: {e}. Try to alternate between 'basic' and 'revolve,' or verify the order of the coordinates.")

                else:
                    st.error("Sketch creation failed.")

                try:
                    stl_from_file(
                        file_path=solid_name,
                        material='material',
                        auto_rotate=False,
                        opacity=1,
                        cam_h_angle=90,
                        height=610,
                        max_view_distance=100000,
                        color='#4169E1'
                    )
                    st.success("Sketch creation completed. Solid generated successfully.")
                    st.session_state.st_sketch = s_surface
                    st.session_state.st_solid = solid

                except:
                    pass

                ############################### DOWNLOAD

                st.divider()
                st.subheader("⬇️ Download", divider='gray', anchor=False)

                col21, col22 = col2.columns([1, 1])

                stl_file_sketch = str(path) + "/" + sketch_name
                stl_file_solid = str(path) + "/" + solid_name

                # Create a download button for STL
                ## Sketch
                col21.download_button(
                    label="Sketch Surface .stl",
                    data=open(sketch_name, "rb").read(),
                    # data=open(stl_file_sketch, "rb").read(),
                    file_name=sketch_name,
                    mime="application/stl",
                    on_click=manter_extrude_button_ativo,
                    use_container_width=True
                )
                ## Solid
                col22.download_button(
                    label="Solid .stl",
                    data=open(solid_name, "rb").read(),
                    # data=open(stl_file_solid, "rb").read(),
                    file_name=solid_name,
                    mime="application/stl",
                    on_click=manter_extrude_button_ativo,
                    use_container_width=True
                )

                with st.spinner('Creating .step File...'):
                    exporters.export(s_surface, 'sketch_MHCAD.step')
                    exporters.export(solid, 'solid_MHCAD.step')

                    step_file_sketch = str(path) + "/sketch_MHCAD.step"
                    step_file_solid = str(path) + "/solid_MHCAD.step"

                    # Create a download button for STEP
                    ## Sketch
                    col21.download_button(
                        label="Sketch Surface .step",
                        # data=open(step_file_sketch, "rb").read(),
                        data=open('sketch_MHCAD.step', "rb").read(),
                        file_name='sketch_MHCAD.step',
                        mime="application/step",
                        on_click=manter_extrude_button_ativo,
                        use_container_width=True
                    )
                    ## Solid
                    col22.download_button(
                        label="Solid .step",
                        # data=open(step_file_solid, "rb").read(),
                        data=open('solid_MHCAD.step', "rb").read(),
                        file_name='solid_MHCAD.step',
                        mime="application/step",
                        on_click=manter_extrude_button_ativo,
                        use_container_width=True
                    )

                ############################### XLSX FILE

                try:

                    file_name_sheet = 'contour_coordinates.xlsx'
                    # Contorno principal
                    df_contour = pd.DataFrame(coordinates_, columns=['x', 'y'])
                    df_contour = scale_contour_df(df_contour, scale) if set_scale else df_contour

                    # Contorno adicional, se extruded_cut for verdadeiro
                    df_coordinates_cut = None
                    if extruded_cut:
                        df_coordinates_cut = pd.DataFrame(coordinates_cut, columns=['x', 'y'])
                        df_coordinates_cut = scale_contour_df(df_coordinates_cut,
                                                              scale) if set_scale else df_coordinates_cut

                    buffer2 = io.BytesIO()
                    with pd.ExcelWriter(buffer2, engine="xlsxwriter") as writer:
                        df_contour.to_excel(writer, sheet_name='Contour', index=False)
                        if df_coordinates_cut is not None:
                            df_coordinates_cut.to_excel(writer, sheet_name='Cut Coordinates', index=False)

                        writer.close()

                        col2.download_button(
                            label="Download Contour Coordinates",
                            data=buffer2,
                            file_name=file_name_sheet,
                            use_container_width=True,
                            on_click=manter_extrude_button_ativo,
                        )

                except Exception as e:
                    st.error("Error: Generating the .xlsx coordinates file.")

            except Exception as e:
                st.error(f"Error in generating the geometry: {e}")
else:
    col2.markdown("")
