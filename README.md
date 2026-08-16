# Jewellery Visual Search Engine

An end-to-end computer vision and vector search pipeline designed to retrieve visually similar jewellery items from a catalog given a user-queried image or real-time camera capture.

---

## Project Overview

Finding specific jewellery designs through text queries can be difficult due to complex patterns, settings, and visual aesthetics. This application solves this by projecting product photos into a high-dimensional vector space where visual similarity can be computed using distance metrics.


The system consists of two main stages:

**1. Offline Embedding & Indexing:** Pre-extracting 1280-dimensional visual feature embeddings using a pretrained CNN backbone **(MobileNetV2)** and storing them in a persistent vector database **(ChromaDB)**.

**2. Interactive Search Interface:** A **Streamlit** web application that allows users to upload query photos or capture images via webcam, extracts query embeddings on the fly, and performs sub-second vector queries against the indexed catalog.

---

## Features

- **Transfer Learning:** Utilizes MobileNetV2 pretrained on ImageNet (classification head removed) for extracting rich visual embeddings.
- **Vector Search Database:** Integrated with ChromaDB (configured with Cosine Distance) to enable fast spatial similarity lookups.
- **Distance Threshold Filtering:** Filters out irrelevant catalog matches when query images fall outside the expected product domain.
- **Interactive UI:** Built with Streamlit, supporting file uploads, live webcam capture, and dynamic match filtering.
- **Quantitative Evaluation:** Includes Precision@K evaluation metric calculation across catalog categories.

---

## 🔗 Links

* **Live Web App:** [Try the App on Streamlit](https://jewelleryvisualsearch.streamlit.app/)

---

## Repository Structure

```text
JewelleryVisualSearch/
├── data/
│   ├── chroma_db/                         # Persistent ChromaDB vector storage index
│   └── Jewellery_Data/                    # Catalog image dataset organized by category
│       ├── necklace/
│       └── ring/
├── notebook/
│   └── jewellery-visual-search.ipynb/     # Offline feature extraction & indexing pipeline script
├── requirements.txt                       # Project dependencies
├── app.py                                 # Streamlit web application
├── .gitignore
└── README.md
