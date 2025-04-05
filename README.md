# Restaurant and Customer Recommendation System

Welcome to the **Restaurant and Customer Recommendation System** project! This project utilizes both **Content-Based Filtering** and **Collaborative Filtering** techniques to build a recommendation engine for restaurants and customers. Built using Python and Jupyter Notebook, the system leverages NLP and Deep Learning methods to enhance recommendation accuracy.

## Project Status

🚧 **Status**: `Completed!`

## Project Workflow

This project follows the **Machine Learning Life Cycle**, specifically for NLP-based recommendation systems:

- **Data Acquisition**
- **Text Cleaning and Pre-processing**
- **Feature Engineering**
- **Pengenalan IndoNLU**
- **Dataset Analisis Sentimen IndoNLU**
- **Sentiment Analysis with Deep Learning**
- **Model Configuration and Pre-trained Model Loading**
- **Sentiment Dataset Preparation**
- **Model Testing with Example Sentences**
- **Fine Tuning and Evaluation**
- **Sentiment Prediction**

### Recommendation Techniques Used:

#### 📌 Content-Based Filtering (TF-IDF & Cosine Similarity)
#### 📌 Collaborative Filtering (Embedding Layers with TensorFlow)

## Dataset

We use the **Restaurant & Consumer Data** available publicly at the UCI Machine Learning Repository:

🔗 [Download Dataset](https://archive.ics.uci.edu/dataset/232/restaurant+consumer+data)

## Technologies

### Content-Based Filtering
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### Collaborative Filtering
```python
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
```

## How to Run

The program can be run directly on the `.ipynb` files. The two methods are separated into two notebooks:

1. **`Reccommendation_System_Restaurant.ipynb`** — For **Content-Based Filtering** using TF-IDF.
2. **`Reccommendation_System_Restaurant_CF.ipynb`** — For **Collaborative Filtering** using embedding techniques.

You can run the notebooks:
- All at once, or
- Block by block for better understanding and modification.

---
