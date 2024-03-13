import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Load the generated dataset
# Use the code provided in the previous response to generate df
df = pd.read_csv('npkdataset.csv')

# Convert the NPK ratio string to separate columns
df[['N', 'P', 'K']] = df['NPK Ratio'].str.split(':', expand=True).astype(int)

# Features (X) and target (y)
X = df[['Land Area']]
y = df[['N', 'P', 'K']]

# Create a Decision Tree Regressor
regressor = DecisionTreeRegressor()

# Train the model
regressor.fit(X, y)

# Streamlit app
st.title("NPK Ratio Predictor")

# Input form
crop_types = df['Crop Type'].unique()
selected_crop_type = st.selectbox("Select Crop Type:", crop_types)
min_land_area = df['Land Area'].min()
max_land_area = df['Land Area'].max()
land_area = st.slider("Select Land Area:", min_value=min_land_area, max_value=max_land_area)

# Make prediction when the 'Predict' button is clicked
if st.button("Predict NPK Ratio"):
    # Convert crop type to lowercase for case-insensitive matching
    crop_type_lower = selected_crop_type.lower()

    # Filter the dataset based on the given crop type
    crop_data = df[df['Crop Type'].str.lower() == crop_type_lower]

    # If data for the specific crop type exists
    if not crop_data.empty:
        # Predict NPK ratio for the given land area
        predicted_npk = regressor.predict([[land_area]])
        st.success(f"Predicted NPK Ratio for {selected_crop_type} with Land Area {land_area}: {predicted_npk[0]}")
    else:
        st.error(f"No data found for the specified crop type: {selected_crop_type}")
