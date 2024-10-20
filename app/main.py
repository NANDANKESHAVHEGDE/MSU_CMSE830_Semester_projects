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
import sys
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline  # Import the transformers library

# Add the parent directory to the Python path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config  # Now this will work

# --- Page Configuration ---
st.set_page_config(page_title="Recipe Gen Predict Dashboard", layout="wide")

# --- Step 1: Load Data and Preprocessing ---
@st.cache_data(show_spinner=False, persist=False)
def load_data():
    file_id = config.GDRIVE_DATA_FILE_ID
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

# --- Step 2: Project Overview ---
def project_overview():
    st.header("Project Overview")
    
    # Add some introductory icons or emojis
    st.markdown("""
    🥘 **Overview of the Recipe Prediction App**  
    This application provides a unique interface for predicting the cuisine type based on user-input ingredients and generates detailed recipes tailored to that cuisine. 
    It combines machine learning with advanced natural language processing to create a seamless cooking experience.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Data Collection Section
    st.subheader("1. Data Collection 🌍")
    st.markdown("""
    - **Sources**: The data was collected from two primary sources:
        - **Kaggle Datasets**: A rich repository of recipe data that includes various cuisines, ingredients, and preparation methods.
        - **Spoonacular API**: An extensive API that provides additional data for recipes, ingredients, and nutrition information.
    
    - **Data Size**: The dataset contains thousands of recipes categorized into various cuisines, ensuring a comprehensive analysis.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Data Cleaning and Preprocessing Section
    st.subheader("2. Data Cleaning and Preprocessing 🧹")
    st.markdown("""
    - **Null Value Handling**: Records with missing values were removed to maintain the integrity of the dataset.
    - **Duplicate Records**: Any duplicate recipes were identified and eliminated.
    - **Unique Ingredients Count**: A new feature, `num_unique_ingredients`, was created to capture the diversity of ingredients in each recipe. This feature is crucial for predicting the cuisine type.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Class Imbalance Management Section
    st.subheader("3. Class Imbalance Management ⚖️")
    st.markdown("""
    - To ensure a balanced representation of both majority and minority cuisines, techniques like **resampling** were employed. This ensures the model can effectively learn from all available classes.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Machine Learning Model Section
    st.subheader("4. Machine Learning Model for Cuisine Prediction 🤖")
    st.markdown("""
    - A **classification model** was developed using **scikit-learn**. The model leverages the number of unique ingredients to predict the cuisine type accurately.
    - The trained model is optimized to recognize patterns in ingredient usage across different cuisines.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Recipe Generation Section
    st.subheader("5. Recipe Generation using Pre-trained Model 📜")
    st.markdown("""
    - After predicting the cuisine, a pre-trained text generation model generates a recipe based on the predicted cuisine and user-provided ingredients.
    - The model outputs detailed step-by-step cooking instructions, enhancing user engagement and providing a practical cooking guide.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Visualizations Section
    st.subheader("6. Visualizations 📊")
    st.markdown("""
    - A variety of visualizations were created to analyze the data, including:
        - **Bar Plots**: Showcasing cuisine distributions.
        - **Box Plots**: Illustrating the distribution of unique ingredients among cuisines.
        - **Scatter Plots**: Visualizing the relationship between the number of unique ingredients and cuisine types.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Deployment Section
    st.subheader("7. Deployment 🚀")
    st.markdown("""
    - The application is deployed on **Streamlit Cloud**, making it accessible for users to explore recipes and cuisines based on their ingredient preferences.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Future Enhancements Section
    st.subheader("8. Future Enhancements 🔮")
    st.markdown("""
    - Integration of user feedback for recipe improvement.
    - Expansion of cuisine types and inclusion of more diverse recipes.
    - Potential collaboration with nutrition APIs for enhanced health information.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Usage Instructions Section
    st.subheader("9. Usage Instructions 📋")
    st.markdown("""
    - To use the application:
        1. Navigate to the **Cuisine Prediction** section.
        2. Enter a list of ingredients separated by commas.
        3. Click on **Predict Cuisine** to see the predicted cuisine.
        4. Generate a detailed recipe with the **Generate Recipe Instructions** button.
    """)

    st.markdown("---")  # Add a horizontal line for separation

    # Decorative Image
    # st.image("./../Background/recipe_background.jpg", caption="Delicious cuisines around the world", use_column_width=True)

    st.markdown("""
    ### Final remarks
    ***This excercise is primarily to motivate more students to attempt usecases in NLP and NLG domain and at best case learn atleast one useful recipe!***
    """)

# --- Step 3: Define Visualizations ---
def plot_visualizations():
    sampled_majority_df = balanced_df[balanced_df['cuisine'].isin(majority_classes)].sample(n=5000, random_state=42)
    sampled_minority_df = balanced_df[balanced_df['cuisine'].isin(minority_classes)].sample(n=2000, random_state=42)

    # Cuisine Distribution
    st.subheader("Cuisine Distribution")
    cuisine_counts = balanced_df['cuisine'].value_counts().to_dict()
    cuisine_counts_df = pd.DataFrame(list(cuisine_counts.items()), columns=['cuisine', 'count'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.barplot(data=cuisine_counts_df[cuisine_counts_df['cuisine'].isin(majority_classes)],
                x='cuisine', y='count', palette='viridis', ax=axes[0])
    axes[0].set_title('Majority Cuisine Distribution')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
    axes[0].set_ylim(0, 200000)

    sns.barplot(data=cuisine_counts_df[cuisine_counts_df['cuisine'].isin(minority_classes)],
                x='cuisine', y='count', palette='viridis', ax=axes[1])
    axes[1].set_title('Minority Cuisine Distribution')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    axes[1].set_ylim(0, 20000)

    st.pyplot(fig)

    # Box Plot for Unique Ingredients
    st.subheader("Unique Ingredients Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.boxplot(data=sampled_majority_df, x='cuisine', y='num_unique_ingredients', palette='viridis', ax=axes[0])
    axes[0].set_title('Majority Classes: Unique Ingredients')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

    sns.boxplot(data=sampled_minority_df, x='cuisine', y='num_unique_ingredients', palette='viridis', ax=axes[1])
    axes[1].set_title('Minority Classes: Unique Ingredients')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

    st.pyplot(fig)

    # Scatter Plot for Unique Ingredients
    st.subheader("Scatter Plot: Unique Ingredients")
    fig_majority = px.scatter(sampled_majority_df, x='num_unique_ingredients', y='cuisine', color='cuisine', title='Majority Classes: Unique Ingredients')
    st.plotly_chart(fig_majority)

    fig_minority = px.scatter(sampled_minority_df, x='num_unique_ingredients', y='cuisine', color='cuisine', title='Minority Classes: Unique Ingredients')
    st.plotly_chart(fig_minority)

    # Ingredient Distribution Histogram
    st.subheader("Ingredient Distribution Histogram")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sns.histplot(data=sampled_majority_df, x='num_unique_ingredients', hue='cuisine', multiple='stack', palette='viridis', ax=axes[0])
    axes[0].set_title('Majority Classes: Unique Ingredients Distribution')

    sns.histplot(data=sampled_minority_df, x='num_unique_ingredients', hue='cuisine', multiple='stack', palette='viridis', ax=axes[1])
    axes[1].set_title('Minority Classes: Unique Ingredients Distribution')

    st.pyplot(fig)

# --- Step 4: Load Trained Model ---
@st.cache_resource  # Using st.cache_resource for model caching
def load_trained_model():
    file_id = config.GDRIVE_MODEL_FILE_ID
    url = f'https://drive.google.com/uc?id={file_id}'

    model_bytes = io.BytesIO()
    gdown.download(url, model_bytes, quiet=False)
    model_bytes.seek(0)

    with model_bytes as f:
        model, le = pickle.load(f)
    return model, le

model, le = load_trained_model()

# --- Step 5: Pre-trained Model for Recipe Instructions ---
# Load a pre-trained text generation model from Hugging Face
recipe_generator = pipeline("text-generation", model="gpt2")  # Change this to a more suitable model if necessary

def generate_recipe_instructions(cuisine, ingredients):
    prompt = (
        f"Create a detailed recipe for a {cuisine} dish using the following ingredients: {ingredients}.\n"
        "Include a list of steps and cooking instructions."
    )
    
    # Generate the recipe instructions
    generated = recipe_generator(prompt, max_length=300, num_return_sequences=1, do_sample=True)
    recipe_instructions = generated[0]['generated_text']
    
    return recipe_instructions

def predict_cuisine(ingredients):
    num_unique_ingredients = len(set(ingredients.split(',')))
    prediction = model.predict([[num_unique_ingredients]])
    cuisine = le.inverse_transform(prediction)[0]
    return cuisine

# --- Step 6: App Layout with Streamlit ---
st.markdown(
    f"""
    <style>
    body {{
        background-size: cover; /* Cover the entire page */
        background-repeat: no-repeat; /* Do not repeat the image */
        background-attachment: fixed; /* Keep the background fixed while scrolling */
        color: #333;  /* Darker text color */
        font-family: Arial, sans-serif;
    }}
    .tab-content {{
        background-color: rgba(255, 255, 255, 0.9);  /* White background with transparency for tab content */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}
    h1, h2, h3 {{
        color: #4a4a4a;  /* Darker headers */
    }}
    .stButton>button {{
        background-color: #007BFF;  /* Primary button color */
        color: white;
    }}
    .stButton>button:hover {{
        background-color: #0056b3;  /* Darker on hover */
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Project Overview", "Visualizations", "Cuisine Prediction and Instruction generation"])

# Title for the app
st.title("Cuisine Analysis and Prediction App")

# Tab 1: Project Overview
if selection == "Project Overview":
    project_overview()

# Tab 2: Visualizations
if selection == "Visualizations":
    st.header("Explore Cuisine Visualizations")
    plot_visualizations()

# Tab 3: Cuisine Prediction
if selection == "Cuisine Prediction and Instruction generation":
    st.header("Cuisine Predictor")
    ingredients_input = st.text_area("Enter Ingredients (separated by commas):", placeholder="e.g., tomato, pasta, olive oil")

    if st.button("Predict Cuisine"):
        if ingredients_input:
            predicted_cuisine = predict_cuisine(ingredients_input)
            st.session_state.predicted_cuisine = predicted_cuisine  # Store predicted cuisine in session state
            st.success(f"The predicted cuisine is: **{predicted_cuisine}**")
        else:
            st.error("Please enter some ingredients to predict the cuisine.")
    
    if st.button("Generate Recipe Instructions"):
        if 'predicted_cuisine' in st.session_state and ingredients_input:
            recipe_instructions = generate_recipe_instructions(st.session_state.predicted_cuisine, ingredients_input)
            st.markdown(f"### Recipe Instructions\n{recipe_instructions}")
        else:
            st.error("Please predict the cuisine first and ensure you have entered ingredients.")

