import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import gdown
import os
import numpy as np
import zipfile
import io
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Page Configuration --- 
st.set_page_config(page_title="Recipe Gen Predict Dashboard", layout="wide")

# --- Step 1: Load Data and Preprocessing ---
@st.cache_data(show_spinner=False, persist=False)
def load_data():
    file_id = "1MhU94QnZrfQV51cl6rIOY6Tvl7JJT53_"  # Replace with your actual Google Drive file ID

    url = f'https://drive.google.com/uc?id={file_id}'
    zip_bytes = io.BytesIO()
    gdown.download(url, zip_bytes, quiet=False)
    zip_bytes.seek(0)

    if zipfile.is_zipfile(zip_bytes):
        with zipfile.ZipFile(zip_bytes, 'r') as z:
            pickle_filename = z.namelist()[0]
            with z.open(pickle_filename) as pickle_file:
                balanced_df = pickle.load(pickle_file)
                balanced_df['num_unique_ingredients'] = balanced_df['ingredients_list'].apply(lambda x: len(set(x.split())))
                return balanced_df
    else:
        st.error("The downloaded file is not a valid zip file.")
        return None

# Load the dataset once and cache it
balanced_df = load_data()

# Define cuisine categories
majority_classes = ['italian', 'mexican', 'chinese', 'indian']
minority_classes = [
    'mediterranean', 'southern_us', 'spanish', 'japanese',
    'middle eastern', 'vietnamese', 'greek', 'french',
    'jamaican', 'moroccan', 'brazilian', 'cajun_creole'
]

# --- Step 2: Define Visualizations ---
def plot_visualizations():
    sampled_majority_df = balanced_df[balanced_df['cuisine'].isin(majority_classes)].sample(n=5000, random_state=42)
    sampled_minority_df = balanced_df[balanced_df['cuisine'].isin(minority_classes)].sample(n=2000, random_state=42)
    
    # 1. Bar Plot for Cuisine Distribution
    st.subheader("Cuisine Distribution")
    cuisine_counts = balanced_df['cuisine'].value_counts().to_dict()
    cuisine_counts_df = pd.DataFrame(list(cuisine_counts.items()), columns=['cuisine', 'count'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(data=cuisine_counts_df[cuisine_counts_df['cuisine'].isin(majority_classes)],
                x='cuisine', y='count', palette='viridis', ax=axes[0])
    axes[0].set_title('Majority Cuisine Distribution')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    axes[0].set_ylim(0, 2000000)

    sns.barplot(data=cuisine_counts_df[cuisine_counts_df['cuisine'].isin(minority_classes)],
                x='cuisine', y='count', palette='viridis', ax=axes[1])
    axes[1].set_title('Minority Cuisine Distribution')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    axes[1].set_ylim(0, 200000)
    
    st.pyplot(fig)

    # 2. Box Plot for Unique Ingredients (using sampled data)
    st.subheader("Unique Ingredients Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.boxplot(data=sampled_majority_df,
                x='cuisine', y='num_unique_ingredients', palette='viridis', ax=axes[0])
    axes[0].set_title('Majority Classes: Unique Ingredients')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

    sns.boxplot(data=sampled_minority_df,
                x='cuisine', y='num_unique_ingredients', palette='viridis', ax=axes[1])
    axes[1].set_title('Minority Classes: Unique Ingredients')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    
    st.pyplot(fig)

    # 3. Scatter Plot for Interactive Visualization (using sampled data)
    st.subheader("Scatter Plot: Unique Ingredients")
    fig_majority = px.scatter(sampled_majority_df,
                              x='num_unique_ingredients', y='cuisine', color='cuisine',
                              title='Majority Classes: Unique Ingredients')
    st.plotly_chart(fig_majority)

    fig_minority = px.scatter(sampled_minority_df,
                              x='num_unique_ingredients', y='cuisine', color='cuisine',
                              title='Minority Classes: Unique Ingredients')
    st.plotly_chart(fig_minority)

    # 4. Histogram for Ingredient Distribution (using sampled data)
    st.subheader("Ingredient Distribution Histogram")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.histplot(data=sampled_majority_df,
                 x='num_unique_ingredients', hue='cuisine', multiple='stack', palette='viridis', ax=axes[0])
    axes[0].set_title('Majority Classes: Unique Ingredients Distribution')

    sns.histplot(data=sampled_minority_df,
                 x='num_unique_ingredients', hue='cuisine', multiple='stack', palette='viridis', ax=axes[1])
    axes[1].set_title('Minority Classes: Unique Ingredients Distribution')

    st.pyplot(fig)

    # 5. Correlation Heatmap
    st.subheader("Correlation Heatmap")
    pivot_df = balanced_df.pivot_table(index='cuisine', values='num_unique_ingredients', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Mean Number of Unique Ingredients per Cuisine')
    st.pyplot(fig)

# --- Step 3: Load Trained Model --- 
@st.cache_resource  # Using st.cache_resource for model caching
def load_trained_model():
    file_id = "1iG2Mp00Jhen6PWUUqb1x1NYAW5nZyGpF"  # Replace with your pre-trained model's Google Drive file ID
    url = f'https://drive.google.com/uc?id={file_id}'

    model_bytes = io.BytesIO()
    gdown.download(url, model_bytes, quiet=False)
    model_bytes.seek(0)

    with model_bytes as f:
        model, le = pickle.load(f)
    return model, le

model, le = load_trained_model()

def predict_cuisine(ingredients):
    num_unique_ingredients = len(set(ingredients.split()))
    prediction = model.predict([[num_unique_ingredients]])
    cuisine = le.inverse_transform(prediction)[0]
    return cuisine

# --- Step 4: App Layout with Streamlit ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    h1, h2, h3 { text-align: center; }
    .stTextArea, .stButton button { border-radius: 8px; border: 1px solid #ccc; }
    footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Visualizations", "Cuisine Prediction"])

# Title for the app
st.title("Cuisine Analysis and Prediction App")

# Tab 1: Visualizations
if selection == "Visualizations":
    st.header("Explore Cuisine Visualizations")
    plot_visualizations()

# Tab 2: Cuisine Prediction
if selection == "Cuisine Prediction":
    st.header("Cuisine Predictor")
    ingredients_input = st.text_area("Enter Ingredients (separated by commas):", placeholder="e.g., tomato, pasta, olive oil")
    if st.button("Predict Cuisine"):
        if ingredients_input:
            predicted_cuisine = predict_cuisine(ingredients_input)
            st.success(f"The predicted cuisine is: **{predicted_cuisine}**")
        else:
            st.error("Please enter some ingredients to predict the cuisine.")
