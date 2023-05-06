import streamlit as st
from st_clickable_images import clickable_images
from os import listdir
from pathlib import Path
import base64

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded_img = base64.b64encode(img_bytes).decode()
    return encoded_img

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

directory = r'images\\'
files = listdir(directory)
enc_images = []

for file in files:
    enc_images.append(f"data:image/jpeg;base64,{img_to_bytes(f'{directory}{file}')}")

clicked = clickable_images(
    enc_images,
    titles=[f"Image #{str(i)}" for i in range(len(enc_images))],
    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    img_style={"margin": "5px", "height": "200px"},
)

st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

if clicked != "":
    change_query = False
    if "last_clicked" not in st.session_state:
        st.session_state["last_clicked"] = clicked
        change_query = True
    else:
        if clicked != st.session_state["last_clicked"]:
            st.session_state["last_clicked"] = clicked
            change_query = True
    if change_query:
        # do something here
        st.experimental_rerun()