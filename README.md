# IMDb Movie Rating Prediction (India)  

## Overview  
This project predicts **IMDb ratings for Indian movies** using a **Linear Regression model**. The dataset contains movie information such as release year, duration, genre, director, actors, and votes. By preprocessing the data and encoding categorical features, the model learns relationships between movie attributes and ratings, enabling predictions for unseen movies.  

## Features  
- Load and clean the **IMDb Movies India** dataset  
- Handle missing values in features like **Rating, Duration, Votes, Genre, Director, Actors**  
- Convert relevant features into numerical format  
- Encode categorical features using **OneHotEncoder**  
- Split dataset into training and testing sets  
- Train a **Linear Regression model**  
- Evaluate performance using **Mean Squared Error (MSE)**  
- Compare **Actual vs Predicted Ratings**  
- Visualize predictions with a line plot
  
## Dataset  
- Dataset used: `IMDb Movies India.csv`  
- Key Columns:  
  - **Name** → Movie title  
  - **Year** → Release year  
  - **Duration** → Movie duration in minutes  
  - **Genre** → Movie genre  
  - **Director** → Director name  
  - **Actor 1, Actor 2, Actor 3** → Leading actors  
  - **Votes** → Number of votes  
  - **Rating** → IMDb rating (Target variable)  

- Preprocessing steps:  
  - Convert numeric columns to proper data types  
  - Extract numbers from duration  
  - Remove commas from votes  
  - Fill missing categorical data with `"Unknown"`  
  - Drop rows with missing ratings  

## Technologies Used  
- Python  
- Pandas  
- Matplotlib  
- Scikit-learn
- VS Code
