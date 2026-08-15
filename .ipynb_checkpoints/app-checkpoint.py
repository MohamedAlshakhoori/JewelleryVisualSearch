import os
import zipfile
import pickle
import numpy as np
import streamlit as st
from PIL import Image
import chromadb

import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array


# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(
    page_title="Jewelry Visual Search Engine",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Jewelry Visual Search Engine")
st.write("Upload a jewelry photo or take a picture to find visually similar catalog items.")

# Directory paths
DATA_DIR = "./data"
CHROMA_ZIP_PATH = os.path.join(DATA_DIR, "chroma_db.zip")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# Automatically extract ChromaDB zip if not extracted
if not os.path.exists(CHROMA_DB_PATH) and os.path.exists(CHROMA_ZIP_PATH):
    with zipfile.ZipFile(CHROMA_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(CHROMA_DB_PATH)


# 2. MODEL & VECTOR DB LOADING (CACHED)
# Use @st.cache_resource to prevent re-loading model on every interaction
@st.cache_resource
def load_feature_extractor():
    """Loads pretrained MobileNetV2 backbone without classification head."""
    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    model.trainable = False
    return model

@st.cache_resource
def load_chroma_collection():
    """Initializes persistent ChromaDB client and loads jewelry collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name="jewellery_catalog")
    return collection

try:
    backbone = load_feature_extractor()
    collection = load_chroma_collection()
    st.sidebar.success("Model & Vector Index loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or index: {e}")
    st.stop()


# 3. HELPER INFERENCE FUNCTIONS
def extract_query_embedding(pil_image):
    """Preprocesses user uploaded image and extracts 1280-D vector embedding."""
    # Resize to 224x224 expected by MobileNetV2
    img_resized = pil_image.resize((224, 224)).convert("RGB")
    img_array = img_to_array(img_resized)
    preprocessed_img = preprocess_input(img_array)
    expanded_img = np.expand_dims(preprocessed_img, axis=0)

    # Extract feature embedding
    embedding = backbone.predict(expanded_img, verbose=0)
    return embedding


# 4. SIDEBAR CONTROLS
st.sidebar.header("Search Settings")

# Retrieve top matches and set distance threshold
top_k = st.sidebar.slider("Number of Matches (Top K)", min_value=1, max_value=25, value=12)

# Distance Threshold: In Cosine distance, 0.0 means identical, values > 0.45 usually mean low similarity
distance_threshold = st.sidebar.slider(
    "Cosine Distance Threshold", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    help="Matches with distance higher than this value will be filtered out as irrelevant."
)


# 5. UI: INPUT SOURCES (FILE UPLOAD / CAMERA)
input_method = st.radio("Choose Input Method:", ("Upload Image", "Use Camera"))

query_image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose a jewelry image...", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a photo of a jewelry item")
    if camera_file is not None:
        query_image = Image.open(camera_file)


# 6. INFERENCE & RETRIEVAL DISPLAY
if query_image is not None:
    # Display Query Image
    st.subheader("Query Image")
    st.image(query_image, width=250)
    
    with st.spinner("Searching catalog for visually similar items..."):
        # 1. Extract embedding vector
        query_vec = extract_query_embedding(query_image)
        
        # 2. Query ChromaDB Collection
        results = collection.query(
            query_embeddings=query_vec.tolist(),
            n_results=top_k,
            include=["uris", "distances"]
        )
        
        retrieved_uris = results["uris"][0]
        retrieved_distances = results["distances"][0]

    # Filter results using the threshold
    filtered_results = [
        (uri, dist) for uri, dist in zip(retrieved_uris, retrieved_distances)
        if dist <= distance_threshold
    ]

    st.markdown("---")
    st.subheader(f"Retrieved Matches ({len(filtered_results)} items passed threshold)")

    if len(filtered_results) == 0:
        st.warning("No similar jewelry items found below the selected distance threshold. Try increasing the threshold in the sidebar!")
    else:
        # Display matches in a responsive grid layout (4 columns per row)
        cols_per_row = 4
        for i in range(0, len(filtered_results), cols_per_row):
            cols = st.columns(cols_per_row)
            batch = filtered_results[i : i + cols_per_row]
            
            for col, (uri, distance) in zip(cols, batch):
                with col:
                    # Dynamically convert Kaggle absolute path to relative local path
                    if "Jewellery_Data" in uri:
                        relative_subpath = uri.split("Jewellery_Data")[-1].lstrip("/\\")
                        local_path = os.path.join(DATA_DIR, "Jewellery_Data", relative_subpath)
                    else:
                        local_path = uri

                    # Render image if local path exists
                    if os.path.exists(local_path):
                        img = Image.open(local_path)
                        st.image(img, use_container_width=True)
                        st.caption(f"Distance: **{distance:.3f}**")
                    else:
                        st.error(f"Image not found at:\n`{local_path}`")