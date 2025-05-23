import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt




st.title("xG Shot Simulator")


import joblib

model, feature_names = joblib.load("../model/xg_model.pkl")
if "shot_df" in st.session_state and not st.session_state.shot_df.empty:
    df = st.session_state.shot_df.copy()
    df_features = pd.get_dummies(df[["x", "y", "LeftFoot", "RightFoot", "BigChance", "Assisted"]])

    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0
    df_features = df_features[feature_names]

    df["xG"] = model.predict_proba(df_features)[:, 1]

    # display
    st.subheader("📋 Shot-by-Shot xG")
    st.dataframe(df[["x", "y", "xG"]])









# Sidebar inputs
st.sidebar.header("Shot Options")
foot = st.sidebar.selectbox("Foot Used", ["LeftFoot", "RightFoot"])
big_chance = st.sidebar.checkbox("Big Chance")
assisted = st.sidebar.checkbox("Assisted")

# Load pitch image 
pitch = Image.open("half_pitch.png")




# Canvas setup
st.subheader("Click on the pitch to simulate a shot")
from PIL import Image
import numpy as np

# Load pitch image
from PIL import Image

# Load pitch image from file
pitch_img = Image.open("half_pitch.png")
img_width, img_height = pitch_img.size


canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    background_image=pitch_img,  # PIL image loaded from half_pitch.png
    update_streamlit=True,
    height=700,  
    width=600,  
    drawing_mode="point",
    point_display_radius=5,
    key="canvas"
)




# Initialize shots list
if "shots" not in st.session_state:
    st.session_state.shots = []

# Process canvas input
if canvas_result.json_data and canvas_result.json_data["objects"]:
    for obj in canvas_result.json_data["objects"]:
        if obj not in st.session_state.shots:
            # Raw canvas coords
            x = obj["left"]
            y = obj["top"]

        
            x_mapped = 60 - (x / 900) * 60
            y_mapped = (y / 600) * 80

            shot_data = {
                "x": x_mapped,
                "y": y_mapped,
                "LeftFoot": foot == "LeftFoot",
                "RightFoot": foot == "RightFoot",
                "BigChance": big_chance,
                "Assisted": assisted
            }

            st.session_state.shots.append(obj)

            if "shot_df" not in st.session_state:
                st.session_state.shot_df = pd.DataFrame([shot_data])
            else:
                st.session_state.shot_df = pd.concat(
                    [st.session_state.shot_df, pd.DataFrame([shot_data])],
                    ignore_index=True
                )









# table and calculate xG
if "shot_df" in st.session_state and not st.session_state.shot_df.empty:
    df = st.session_state.shot_df.copy()
    df_features = df[["x", "y", "LeftFoot", "RightFoot", "BigChance", "Assisted"]]
    df_features = pd.get_dummies(df_features)
    for col in model.feature_names_in_:
        if col not in df_features.columns:
            df_features[col] = 0
    df_features = df_features[model.feature_names_in_]
    df["xG"] = model.predict_proba(df_features)[:, 1]

    st.subheader("📋 Shot-by-Shot xG")
    st.dataframe(df[["x", "y", "xG"]])


st.subheader("📊 Total xG")
if "shot_df" in st.session_state and not st.session_state.shot_df.empty:
    df = st.session_state.shot_df.copy()
    df_features = pd.get_dummies(df[["x", "y", "LeftFoot", "RightFoot", "BigChance", "Assisted"]])

    
    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0
    df_features = df_features[feature_names]

    df["xG"] = model.predict_proba(df_features)[:, 1]
    st.session_state.shot_df = df  

    
    total_xg = df["xG"].sum()
    st.metric(label="Cumulative xG", value=round(total_xg, 3))
else:
    st.write("No shots recorded yet.") 