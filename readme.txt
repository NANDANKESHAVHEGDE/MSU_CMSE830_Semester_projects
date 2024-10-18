# Cuisine Analysis and Prediction App

## Overview
This is a Streamlit-based web application designed to analyze the distribution of cuisines based on recipes and predict the cuisine type from a list of ingredients. The app supports visualizations of cuisine distributions and unique ingredient patterns, along with a simple machine learning model that predicts cuisine based on the number of unique ingredients.

## Features
- **Cuisine Visualizations:**
  - Bar charts showcasing the distribution of majority and minority cuisines.
  - Box plots and scatter plots for visualizing the unique ingredient distribution across cuisines.
  - Histograms that depict the ingredient distribution within majority and minority cuisine categories.
  - Heatmap for showing correlations between cuisine types and their unique ingredients.
  
- **Cuisine Prediction:**
  - Predict the cuisine type by entering a list of ingredients.
  - The prediction model is based on the number of unique ingredients and uses a Random Forest Classifier.

## Dataset
The app uses a **balanced sample of 1 million rows** extracted from a larger recipe dataset. The dataset includes various cuisine types, focusing on both majority and minority cuisines to ensure fair representation. The cuisines in the dataset include:
- Majority: `Italian`, `Mexican`, `Chinese`, `Indian`
- Minority: `Mediterranean`, `Southern US`, `Spanish`, `Japanese`, `Middle Eastern`, `Vietnamese`, `Greek`, `French`, `Jamaican`, `Moroccan`, `Brazilian`, `Cajun Creole`

## Prerequisites

Refer to requirements.txt file in the repository

## The app is hosted in the below url:

https://msucmse830semesterprojects-6f9j4vtb6tp5ziynhxeafp.streamlit.app/
