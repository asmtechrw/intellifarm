import numpy as np
import pandas as pd
import streamlit as st
import base64
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import io
from PIL import Image
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def video_preview(video_url, title, description):
    st.subheader(title)
    st.text(description)
    st.video(video_url)

def card(name, location, review):
    st.markdown(f"## {name}")
    st.markdown(f"**Location:** {location}")
    st.write(review)

# Load your machine learning model and data
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv("crop_prediction_model_one.csv")

# Load your machine learning model and data
model2 = pickle.load(open('model2.pkl', 'rb'))
df2 = pd.read_csv("Soil.csv")

# Load your machine learning model and data
# model3 = pickle.load(open('model3.pkl', 'rb'))
df3 = pd.read_csv("Fertilizer.csv")

# Update column names based on your fertilizer dataset
X_fertilizer = df3[['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y_fertilizer = df3['Fertilizer Name']

# Separate numerical and categorical features
numerical_features_fertilizer = ['Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features_fertilizer = ['Soil Type', 'Crop Type']

# Create transformers for numerical and categorical features
numeric_transformer_fertilizer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer_fertilizer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Use ColumnTransformer to apply transformers to the respective features
preprocessor_fertilizer = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_fertilizer, numerical_features_fertilizer),
        ('cat', categorical_transformer_fertilizer, categorical_features_fertilizer)
    ])

# Fit the preprocessor on the training data
preprocessor_fertilizer.fit(X_fertilizer)

# Define the SVC classifier and the parameter grid for RandomizedSearchCV
svc_fertilizer = SVC()

grid_fertilizer = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'degree': [2, 3, 4, 5]
}

# Setup RandomizedSearchCV for fertilizer prediction
rs_svc_fertilizer = RandomizedSearchCV(estimator=svc_fertilizer,
                                       param_distributions=grid_fertilizer,
                                       n_iter=10,
                                       cv=5,
                                       verbose=2,
                                       n_jobs=-1)

# Create a pipeline with preprocessing and SVC for fertilizer prediction
pipeline_fertilizer = Pipeline(steps=[('preprocessor', preprocessor_fertilizer),
                                       ('svc', rs_svc_fertilizer)])

# Split the data into training and testing sets for fertilizer prediction
X_train_fertilizer, X_test_fertilizer, y_train_fertilizer, y_test_fertilizer = train_test_split(X_fertilizer, y_fertilizer, test_size=0.2, random_state=42)

# Fit the pipeline on the training set for fertilizer prediction
pipeline_fertilizer.fit(X_train_fertilizer, y_train_fertilizer)

# Load the preprocessor and model from the pickle file
with open('model3.pkl', 'rb') as file:
    pipeline_fertilizer = pickle.load(file)

def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Temparature': [temperature],
        'Humidity ': [humidity],
        'Moisture': [moisture],
        'Soil Type': [soil_type],
        'Crop Type': [crop_type],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous]
    })

    # Make the prediction using the loaded pipeline
    prediction = pipeline_fertilizer.predict(input_data)

    return prediction[0]

    

def bar_plot_drawer(x, y):
    fig = plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid")
    sns.barplot(data=df3, x=x, y=y)
    plt.xlabel("Fertilizers", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def scatter_plot_drawer(x, y):
    fig = plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df3, x=x, y=y, hue="Fertilizer Name", size="Fertilizer Name", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def box_plot_drawer(x, y):
    fig = plt.figure(figsize=(20, 15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df3)
    sns.despine(offset=10, trim=True)
    plt.xlabel("Fertilizers", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

converts_dict = {
    'Nitrogen': 'N',
    'Phosphorus': 'P',
    'Potassium': 'K',
    'Temperature': 'temperature',
    'Humidity': 'humidity',
    'Rainfall': 'rainfall',
    'ph': 'ph'
}

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    input = np.array([[n, p, k, temperature, humidity, ph, rainfall]]).astype(np.float64)
    prediction = model.predict(input)
    return prediction[0]

def predict_soil(N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B):
    input = np.array([[N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B]]).astype(np.float64)
    prediction = model2.predict(input)
    return prediction[0]


def scatterPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def scatterPlotDrawer2(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df2, x=x, y=y, hue="label", size="label", palette="deep", sizes=(20, 200), legend="full")
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x=x, y=y)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def barPlotDrawer2(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.barplot(data=df2, x=x, y=y)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)        

def boxPlotDrawer(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df)
    sns.despine(offset=10, trim=True)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)

def boxPlotDrawer2(x,y):
    fig = plt.figure(figsize=(20,15))
    sns.set_style("whitegrid")
    sns.boxplot(x=x, y=y, data=df2)
    sns.despine(offset=10, trim=True)
    plt.xlabel("Crops", fontsize=22)
    plt.ylabel(y, fontsize=22)
    plt.xticks(rotation=90, fontsize=18)
    plt.legend(prop={'size': 18})
    plt.yticks(fontsize=16)
    st.write(fig)


@st.cache_data
def get_img_as_base64_with_transparency(file, transparency):
    img = Image.open(file)
    
    # Make the image transparent
    img.putalpha(int(255 * (1 - transparency)))  # 0 is fully transparent, 255 is fully opaque
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    return base64.b64encode(buffered.getvalue()).decode()

def main():
    transparency = 0.6  # 30% transparency
    img = get_img_as_base64_with_transparency("bg5.jpeg", transparency)

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)


    html_temp_vis = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Visualize crop Properties </h2>
    </div>
    """

    html_temp_pred = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Which Crop To Cultivate? </h2>
    </div>
    """

    html_temp_vis2 = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Visualize Soil Properties </h2>
    </div>
    """

    html_temp_pred2 = """
    <div style="padding:10px;margin-bottom:30px">
    <h2 style="color:white;text-align:center;"> Is soil good To Cultivate? </h2>
    </div>
    """

    st.sidebar.title("What's in Store?")
    select_type = st.sidebar.radio("", ('Home','crop prediction','soil prediction','Fertilizer Prediction','NPk Ratio'))
    if select_type == 'Home':
        st.title("Smart Farming with AI")
        st.image("./images/mainhome.jpg",width=590)
        st.markdown(
    """<p style="font-size:19px;">
            IntelliFarm is a precision farming solution designed to revolutionize modern agriculture by leveraging machine learning, Python, and Streamlit. The project aims to provide farmers with real-time, data-driven insights to enhance crop yield, optimize resource utilization, and foster sustainable farming practices.
        </p>
    """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Crop Prediction")
            st.image("./images/crop_prediction.webp", caption="About Crop Prediction",
                 width=200)
            st.markdown(
            """<p style="font-size:18px;">
            Visualize and predict the most suitable crop for your field using advanced machine learning algorithms. Make informed decisions to optimize your crop yield.
            </p>
            """, unsafe_allow_html=True)
        with col2:
            st.subheader("Soil Prediction")
            st.image("./images/soil_prediction.png", caption="About Soil Prediction",
                 width=200)
            st.markdown(
            """<p style="font-size:18px;">
            Gain insights into your soil health and make informed decisions by predicting soil conditions. Enhance your farming practices with accurate soil predictions.
            </p>
            """, unsafe_allow_html=True)

    
        with col3:
            st.subheader("Fertilizer Prediction")
            st.image("./images/fertilizer.jpeg", caption="About Fertilizer Prediction",
                 width=200)
            st.markdown(
            """<p style="font-size:18px;">
            Optimize your fertilizer usage by predicting the right type and quantity for your crops. Improve resource utilization and promote sustainable farming practices.
            </p>
            """, unsafe_allow_html=True)
        
    if select_type == 'crop prediction':
        st.sidebar.title("Let's go to....")
        select_type = st.sidebar.radio("", ('Home','Data Info','visualize crop', 'Predict Your Crop'))
         

        if select_type == 'Home':
            st.header("Crop Prediction")
            st.subheader("Visualizing Crops")
            st.image("./images/crophome.jpg", caption="Visualizing Crops",
             width=400)
            st.markdown(
        """<p style="font-size:16px;">
        Get a clear visualization of different crops and their growth stages. Understand the characteristics that affect crop health and yield.
        </p>
        """, unsafe_allow_html=True)
            st.write("\n")

        st.subheader("Predicting Crops")
        st.image("./images/crop_prediction.webp", caption="Predicting Crops",
             width=400)
        st.markdown(
        """<p style="font-size:16px;">
        Utilize advanced machine learning algorithms to predict the most suitable crops for your field. Make informed decisions to optimize your crop yield.
        </p>
        """, unsafe_allow_html=True)
        st.write("\n")

   


        if select_type == 'Data Info':
             # Add title to the page
            st.title("Data Info page")

            # Add subheader for the section
            st.subheader("View Data")

            # Create an expansion option to check the data
            with st.expander("View data"):
                st.dataframe(df)

            # Create a section to columns values
            # Give subheader
            st.subheader("Columns Description:")

            # Create a checkbox to get the summary.
            if st.checkbox("View Summary"):
                st.dataframe(df.describe())

            # Create multiple check box in row
            col_name, col_dtype, col_data = st.columns(3)

            # Show name of all dataframe
            with col_name:
                if st.checkbox("Column Names"):
                    st.dataframe(df.columns)

            # Show datatype of all columns 
            with col_dtype:
                if st.checkbox("Columns data types"):
                    dtypes = df.dtypes.apply(lambda x: x.name)
                    st.dataframe(dtypes)
            
            # Show data for each columns
            with col_data: 
                if st.checkbox("Columns Data"):
                    col = st.selectbox("Column Name", list(df.columns))
                    st.dataframe(df[col])

            # Add the link to you dataset
            st.markdown("""
                            <p style="font-size:24px">
                                <a 
                                    href="https://www.kaggle.com/uciml/pima-indians-diabetes-database"
                                    target=_blank
                                    style="text-decoration:none;"
                                >Get Dataset
                                </a> 
                            </p>
                        """, unsafe_allow_html=True
            )
        elif select_type == 'visualize crop':
            st.markdown(html_temp_vis, unsafe_allow_html=True)
            plot_type = st.selectbox("Select plot type", ('Bar Plot', 'Scatter Plot', 'Box Plot'))
            st.subheader("Relation between features")

            # Plot!
            x = ""
            y = ""

            if plot_type == 'Bar Plot':
                x = 'label'
                y = st.selectbox("Select a feature to compare between crops",
                    ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
            if plot_type == 'Scatter Plot':
                x = st.selectbox("Select a property for 'X' axis",
                    ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
                y = st.selectbox("Select a property for 'Y' axis",
                    ('Nitrogen', 'Phosphorus', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))
            if plot_type == 'Box Plot':
                x = "label"
                y = st.selectbox("Select a feature",
                    ('Phosphorus', 'Nitrogen', 'ph', 'Potassium', 'Temperature', 'Humidity', 'Rainfall'))

            if st.button("Visualize"):
                if plot_type == 'Bar Plot':
                    y = converts_dict[y]
                    barPlotDrawer(x, y)
                if plot_type == 'Scatter Plot':
                    x = converts_dict[x]
                    y = converts_dict[y]
                    scatterPlotDrawer(x, y)
                if plot_type == 'Box Plot':
                    y = converts_dict[y]
                    boxPlotDrawer(x, y)

        if select_type == "Predict Your Crop":
            st.markdown(html_temp_pred, unsafe_allow_html=True)
            st.header("To predict your crop give values")
            st.subheader("Drag to Give Values")
            n = st.slider('Nitrogen (N)', 0, 140)
            p = st.slider('Phosphorus (P)', 5, 145)
            k = st.slider('Potassium (K)', 5, 205)
            temperature = st.slider('Temperature', 8.83, 43.68)
            humidity = st.slider('Humidity', 14.26, 99.98)
            ph = st.slider('pH', 3.50, 9.94)
            rainfall = st.slider('Rainfall', 20.21, 298.56)
            
            if st.button("Predict your crop"):
                output=predict_crop(n, p, k, temperature, humidity, ph, rainfall)
                res = "“"+ output.capitalize() + "”"
                st.success('The most suitable crop for your field is {}'.format(res))
                

    elif select_type == 'soil prediction':
        st.sidebar.title("Let's go to....")
        select_type = st.sidebar.radio("", ('Home','Data Info','visualize soil columns', 'Predict Your soil'))


        if select_type == 'Home':

            st.header("Soil Prediction")

            st.subheader("Understanding Soil Health")
            st.image("./images/img1.jpg", caption="Understanding Soil Health",
             width=400)
            st.markdown(
        """<p style="font-size:16px;">
        Gain deep insights into your soil health. Understand key factors such as nutrient levels, moisture content, and pH that influence crop growth.
        </p>
        """, unsafe_allow_html=True)
            st.write("\n")

            st.subheader("Predicting Soil Conditions")
            st.image("./images/img5.jpg", caption="Predicting Soil Conditions",
             width=400)
            st.markdown(
        """<p style="font-size:16px;">
        Predict soil conditions with precision. Use cutting-edge algorithms to forecast changes in soil quality, helping you make informed decisions.
        </p>
        """, unsafe_allow_html=True)
            st.write("\n")

            st.subheader("Optimizing Soil Management")
            st.image("./images/img2.jpg", caption="Optimizing Soil Management",
             width=400)
            st.markdown(
        """<p style="font-size:16px;">
        Optimize your soil management practices based on predictive insights. Maximize crop yield by adjusting fertilization and irrigation strategies.
        </p>
        """, unsafe_allow_html=True)
            st.write("\n")


        if select_type == 'Data Info':
             # Add title to the page
            st.title("Data Info page")

            # Add subheader for the section
            st.subheader("View Data")

            # Create an expansion option to check the data
            with st.expander("View data"):
                st.dataframe(df2)

            # Create a section to columns values
            # Give subheader
            st.subheader("Columns Description:")

            # Create a checkbox to get the summary.
            if st.checkbox("View Summary"):
                st.dataframe(df2.describe())

            # Create multiple check box in row
            col_name, col_dtype, col_data = st.columns(3)

            # Show name of all dataframe
            with col_name:
                if st.checkbox("Column Names"):
                    st.dataframe(df2.columns)

            # Show datatype of all columns 
            with col_dtype:
                if st.checkbox("Columns data types"):
                    dtypes = df2.dtypes.apply(lambda x: x.name)
                    st.dataframe(dtypes)
            
            # Show data for each columns
            with col_data: 
                if st.checkbox("Columns Data"):
                    col = st.selectbox("Column Name", list(df2.columns))
                    st.dataframe(df2[col])

            # Add the link to you dataset
            st.markdown("""
                            <p style="font-size:24px">
                                <a 
                                    href="https://www.kaggle.com/uciml/pima-indians-diabetes-database"
                                    target=_blank
                                    style="text-decoration:none;"
                                >Get Dataset
                                </a> 
                            </p>
                        """, unsafe_allow_html=True
            )
        elif select_type == 'visualize soil columns':
            st.markdown(html_temp_vis2, unsafe_allow_html=True)
            plot_type = st.selectbox("Select plot type", ('Bar Plot', 'Scatter Plot', 'Box Plot'))
            st.subheader("Relation between features")

            # Plot!
            x = ""
            y = ""

            if plot_type == 'Bar Plot':
                x = 'Output'
                y = st.selectbox("Select a feature to compare between crops",
                    ('P', 'N', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))
            if plot_type == 'Scatter Plot':
                x = st.selectbox("Select a property for 'X' axis",
                    ('P', 'N', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))
                y = st.selectbox("Select a property for 'Y' axis",
                    ('N', 'P', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))
            if plot_type == 'Box Plot':
                x = "Output"
                y = st.selectbox("Select a feature",
                    ('P', 'N', 'pH', 'K', 'EC', 'OC', 'S', 'Zn', 'Fe', 'Cu', 'Mn', 'B'))

            if st.button("Visualize"):
                if plot_type == 'Bar Plot':
                    barPlotDrawer2(x, y)
                if plot_type == 'Scatter Plot':
                    scatterPlotDrawer2(x, y)
                if plot_type == 'Box Plot':
                    boxPlotDrawer2(x, y)

        if select_type == "Predict Your soil":
            st.markdown(html_temp_pred2, unsafe_allow_html=True)
            st.header("To predict your crop, give values")
            st.subheader("Drag to Give Values")
            # Calculate the minimum and maximum values from the dataset
            min_N = float(df2['N'].min())
            max_N = float(df2['N'].max())
            min_P = float(df2['P'].min())
            max_P = float(df2['P'].max())
            min_K = float(df2['K'].min())
            max_K = float(df2['K'].max())
            min_pH = float(df2['pH'].min())
            max_pH = float(df2['pH'].max())
            min_EC = float(df2['EC'].min())
            max_EC = float(df2['EC'].max())
            min_OC = float(df2['OC'].min())
            max_OC = float(df2['OC'].max())
            min_S = float(df2['S'].min())   
            max_S = float(df2['S'].max())       
            min_Zn = float(df2['Zn'].min())
            max_Zn = float(df2['Zn'].max())
            min_Fe = float(df2['Fe'].min())
            max_Fe = float(df2['Fe'].max())
            min_Cu = float(df2['Cu'].min())
            max_Cu = float(df2['Cu'].max())
            min_Mn = float(df2['Mn'].min())
            max_Mn = float(df2['Mn'].max())
            min_B = float(df2['B'].min())
            max_B = float(df2['B'].max())

            N = st.slider('Nitrogen (N)', min_value=min_N, max_value=max_N, value=min_N, step=1.0)
            P = st.slider('Phosphorus (P)', min_value=min_P, max_value=max_P, value=min_P, step=1.0)
            K = st.slider('Potassium (K)', min_value=min_K, max_value=max_K, value=min_K, step=1.0)
            pH = st.slider('pH', min_value=min_pH, max_value=max_pH, value=min_pH, step=0.01)
            EC = st.slider('EC (Emulsifiable concentrate) ', min_value=min_EC, max_value=max_EC, value=min_EC, step=0.01)
            OC = st.slider('OC', min_value=min_OC, max_value=max_OC, value=min_OC, step=0.01)
            S = st.slider('S', min_value=min_S, max_value=max_S, value=min_S, step=1.0)
            Zn = st.slider('Zn', min_value=min_Zn, max_value=max_Zn, value=min_Zn, step=1.0)
            Fe = st.slider('Fe', min_value=min_Fe, max_value=max_Fe, value=min_Fe, step=1.0)
            Cu = st.slider('Cu', min_value=min_Cu, max_value=max_Cu, value=min_Cu, step=1.0)
            Mn = st.slider('Mn', min_value=min_Mn, max_value=max_Mn, value=min_Mn, step=1.0)
            B = st.slider('B', min_value=min_B, max_value=max_B, value=min_B, step=0.01)

            if st.button("Predict your soil"):
                output = predict_soil(N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B)
                output_str = str(output).capitalize()
                if output_str == 0:
                    st.success(f'This soil is not Good for cultivation')
                else:
                    st.success(f'This soil is Good for cultivation')

    elif select_type == 'Fertilizer Prediction':
        st.sidebar.title("Know about Fertilizer")
        select_type = st.sidebar.radio("", ('Home', 'Data Info','Visualize Fertilizer', 'Predict Fertilizer'))
         

        if select_type == 'Home':

           st.header("Fertilizer Prediction")

           st.subheader("Optimizing Fertilizer Usage")
           st.image("./images/fertilizerhome.jpg", caption="Optimizing Fertilizer Usage",
             width=400)
           st.markdown(
        """<p style="font-size:16px;">
        Optimize your fertilizer usage with predictive analytics. Tailor the type and quantity of fertilizers to maximize crop nutrition and minimize waste.
        </p>
        """, unsafe_allow_html=True)
           st.write("\n")

           st.subheader("Balancing Nutrient Levels")
           st.image("./images/img8.jpg", caption="Balancing Nutrient Levels",
             width=400)
           st.markdown(
        """<p style="font-size:16px;">
        Achieve optimal nutrient balance in your soil. Predict the right composition of fertilizers to ensure healthy crop growth and minimize environmental impact.
        </p>
        """, unsafe_allow_html=True)
           st.write("\n")

           st.subheader("Promoting Sustainable Practices")
           st.image("./images/img6.jpg", caption="Promoting Sustainable Practices",
             width=400)
           st.markdown(
        """<p style="font-size:16px;">
        Embrace sustainable farming practices by predicting and implementing eco-friendly fertilizer strategies. Contribute to environmental conservation while maximizing yield.
        </p>
        """, unsafe_allow_html=True)
           st.write("\n")





        if select_type == 'Data Info':
            st.title("Data Info page")
            st.subheader("View Data")
            with st.expander("View data"):
                st.dataframe(df3)
            st.subheader("Columns Description:")
            if st.checkbox("View Summary"):
                st.dataframe(df3.describe())
            col_name, col_dtype, col_data = st.columns(3)
            with col_name:
                if st.checkbox("Column Names"):
                    st.dataframe(df3.columns)

            with col_dtype:
                if st.checkbox("Columns data types"):
                    dtypes = df3.dtypes.apply(lambda x: x.name)
                    st.dataframe(dtypes)

       
            with col_data:
                if st.checkbox("Columns Data"):
                    col = st.selectbox("Column Name", list(df3.columns))
                    st.dataframe(df3[col])
        elif select_type == 'Visualize Fertilizer':
            st.subheader("Visualize Fertilizer Properties")
            plot_type = st.selectbox("Select plot type", ('Bar Plot', 'Scatter Plot', 'Box Plot'))
            st.subheader("Relation between features")

            # Plot!
            x = ""
            y = ""

            if plot_type == 'Bar Plot':
                x = 'Fertilizer Name'
                y = st.selectbox("Select a feature to compare between fertilizers",
                    ('Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'))
            if plot_type == 'Scatter Plot':
                x = st.selectbox("Select a property for 'X' axis",
                    ('Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'))
                y = st.selectbox("Select a property for 'Y' axis",
                    ('Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'))
            if plot_type == 'Box Plot':
                x = 'Fertilizer Name'
                y = st.selectbox("Select a feature",
                    ('Temparature', 'Humidity ', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'))

            if st.button("Visualize"):
                if plot_type == 'Bar Plot':
                    bar_plot_drawer(x, y)
                if plot_type == 'Scatter Plot':
                    scatter_plot_drawer(x, y)
                if plot_type == 'Box Plot':
                    box_plot_drawer(x, y)

        if select_type == 'Predict Fertilizer':
            st.header("To predict fertilizer recommendation, provide the following values:")
            temperature = st.slider('Temperature', min_value=df3['Temparature'].min(), max_value=df3['Temparature'].max(), step=1, value=int(df3['Temparature'].mean()))
            humidity = st.slider('Humidity (%)', min_value=df3['Humidity '].min(), max_value=df3['Humidity '].max(), step=1, value=int(df3['Humidity '].mean()))
            moisture = st.slider('Moisture (%)', min_value=df3['Moisture'].min(), max_value=df3['Moisture'].max(), step=1, value=int(df3['Moisture'].mean()))
            soil_type = st.selectbox('Select Soil Type', df3['Soil Type'].unique())
            crop_type = st.selectbox('Select Crop Type', df3['Crop Type'].unique())
            nitrogen = st.slider('Nitrogen', min_value=df3['Nitrogen'].min(), max_value=df3['Nitrogen'].max(), step=1, value=int(df3['Nitrogen'].mean()))
            potassium = st.slider('Potassium', min_value=df3['Potassium'].min(), max_value=df3['Potassium'].max(), step=1, value=int(df3['Potassium'].mean()))
            phosphorous = st.slider('Phosphorous', min_value=df3['Phosphorous'].min(), max_value=df3['Phosphorous'].max(), step=1, value=int(df3['Phosphorous'].mean()))

            if st.button("Predict Fertilizer"):
                output = predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous)
                st.success(f"The recommended fertilizer is: {output}")

    elif select_type == 'NPk Ratio':
        npkdataset = pd.read_csv('npkdataset.csv')

        # Convert the NPK ratio string to separate columns
        npkdataset[['N', 'P', 'K']] = npkdataset['NPK Ratio'].str.split(':', expand=True).astype(int)

        # Features (X) and target (y)
        X = npkdataset[['Land Area']]
        y = npkdataset[['N', 'P', 'K']]

        # Create a Decision Tree Regressor
        regressor = DecisionTreeRegressor()

        # Train the model
        regressor.fit(X, y)

        # Streamlit app
        st.title("NPK Ratio Predictor")

        # Input form
        crop_types = npkdataset['Crop Type'].unique()
        selected_crop_type = st.selectbox("Select Crop Type:", crop_types)
        min_land_area = npkdataset['Land Area'].min()
        max_land_area = npkdataset['Land Area'].max()
        land_area = st.slider("Select Land Area:", min_value=min_land_area, max_value=max_land_area)

        # Make prediction when the 'Predict' button is clicked
        if st.button("Predict NPK Ratio"):
            # Convert crop type to lowercase for case-insensitive matching
            crop_type_lower = selected_crop_type.lower()

            # Filter the dataset based on the given crop type
            crop_data = npkdataset[npkdataset['Crop Type'].str.lower() == crop_type_lower]

            # If data for the specific crop type exists
            if not crop_data.empty:
                # Predict NPK ratio for the given land area
                predicted_npk = regressor.predict([[land_area]])
                st.success(f"Predicted NPK Ratio for {selected_crop_type} with Land Area {land_area}: {predicted_npk[0]}")
            else:
                st.error(f"No data found for the specified crop type: {selected_crop_type}")

if __name__ == '__main__':
    main()