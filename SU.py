import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from streamlit_folium import folium_static
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from twilio.rest import Client
import altair as alt
from bs4 import BeautifulSoup
import requests

# Custom CSS for full-screen background, fade-in effect, and other styling
def set_bg_as_image():
    st.markdown(
        """
        <style>
        /* Fullscreen pseudo-element with the background image */
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            background-image: url("https://cdna.artstation.com/p/assets/images/images/016/265/566/original/mikhail-gorbunov-ui-1.gif?1551525784");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: blur(8px);
        }

        /* Override Streamlit's default styling */
        .stApp {
            background-color: transparent;
        }

        /* Fade-in animation */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Fade-out animation */
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }

        /* Styling for text and content box */
        h1, .content {
            color: #ffffff;
            z-index: 1;
            position: relative;
            animation: fadeIn 2s ease-in-out;
        }

        .content {
            padding: 2rem;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            margin-top: 4rem;
        }

        /* Styling for the sidebar */
        .css-1lcbmhc {
            top: 0;
            position: fixed;
            z-index: 2;
        }
        
         /* Responsive styles */
        @media (max-width: 640px) {
            /* Fullscreen pseudo-element with the background image */
            .stApp::before {
                background-size: contain;
                background-position: top;
        }

        /* Content adjustments */
            .content {
                padding: 1rem; /* smaller padding */
                margin-top: 2rem; /* smaller margin */
                font-size: 14px; /* smaller font size */
        }

        /* Sidebar adjustments */
            .css-1lcbmhc, .stSidebar {
                width: 100% !important;
                z-index: 2;
        }

        /* Hide the hamburger menu to save space */
            .css-1v3fvcr {
                display: none;
        }

        /* Adjust specific Streamlit elements for smaller screens */
            .stButton > button, .stTextInput > div > div > input, .stSelectbox > select {
                width: 100% !important;
                font-size: 16px; /* Adjust font size as needed */
        }

        /* Make plotly charts responsive */
            .plotly-graph-div {
                width: 100% !important;
        }

        /* Adjust the title for smaller screens */
            h1 {
                font-size: 22px; /* Smaller font size for the title */
        }
        
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# Set the background
set_bg_as_image()

# Set up the sidebar
st.sidebar.title("Prediction Type")
sidebar_option = st.sidebar.selectbox("Choose an option:", ["Intenisty and Chart" , "Crime Map"])
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Display the heading
st.title("Welcome to Sentinel Mark 2 : Crime predictive Model ")

# Emoji for large headings
crime_chart_emoji = "📊"
intensity_map_emoji = "🗺️"

# Conditional display based on CSV upload
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Correct column names based on your CSV file
    crime_types = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']
    df_crime_types = df[crime_types + ['nm_pol', 'long', 'lat']]
    df_grouped = df_crime_types.groupby('nm_pol').sum().reset_index()
    df_grouped['Total Crime Intensity'] = df_grouped[crime_types].sum(axis=1)

    # If "Crime Map" is not selected, show the chart and intensity map
    if sidebar_option != "Crime Map":
        # Display Crime Type Chart
        st.subheader(f'{crime_chart_emoji} Crime Type Chart')
        df_melted = pd.melt(df_grouped, id_vars=['nm_pol'], value_vars=crime_types, var_name='Crime Type', value_name='Count')
        fig = px.bar(
            df_melted,
            x='nm_pol',
            y='Count',
            color='Crime Type',
            title=f'{crime_chart_emoji} Crime Distribution by Police Station',
            labels={'nm_pol': 'Police Station', 'Count': 'Number of Incidents'}
        )
        fig.update_layout(
            xaxis_title="Police Station",
            yaxis_title="Number of Incidents",
            barmode='stack',
            xaxis={'categoryorder': 'total descending'},
            legend_title_text='Crime Type'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display the Crime Intensity Map
        st.subheader(f'{intensity_map_emoji} Crime Intensity Map')
        fig_map = px.scatter_mapbox(
            df_grouped,
            lat='lat',
            lon='long',
            size='Total Crime Intensity',
            color='Total Crime Intensity',
            hover_name='nm_pol',
            title=f'{intensity_map_emoji} Crime Intensity Map',
            mapbox_style="open-street-map",
            zoom=10
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

#<------------------------------------------------- Random forest and Crime Map---------------------------------------------------------------------->  

    # If "Crime Map" is selected, only show the folium map
    elif sidebar_option == "Crime Map":
        st.subheader(f'{intensity_map_emoji} Crime Probability Map')
        X_train, X_test, y_train, y_test = train_test_split(df[['lat', 'long']], df[crime_types], test_size=0.3, random_state=42)
        class_weights = 'balanced'
        multi_output_rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
        multi_output_rf.fit(X_train, y_train)
        y_pred_proba = multi_output_rf.predict_proba(X_test)
        y_pred_proba_positive = np.array([proba[:, 1] for proba in y_pred_proba]).T

        map_center = [X_test['lat'].mean(), X_test['long'].mean()]
        crime_map_rf = folium.Map(location=map_center, zoom_start=12)
        high_probability_threshold = 0.5
        moderate_probability_threshold = 0.2

        for idx, (lat, long) in enumerate(zip(X_test['lat'], X_test['long'])):
            max_probability = y_pred_proba_positive[idx, :].max()
            if max_probability >= high_probability_threshold:
                color = 'red'
            elif max_probability >= moderate_probability_threshold:
                color = 'yellow'
            else:
                color = 'green'
            popup_info = '<br>'.join([f"{crime}: {y_pred_proba_positive[idx, i]:.2f}" for i, crime in enumerate(crime_types)])
            folium.CircleMarker(
                location=[lat, long],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                popup=folium.Popup(popup_info, max_width=300)
            ).add_to(crime_map_rf)
        folium_static(crime_map_rf)
    
# <------------------------------------------------- Model Training DNN ------------------------------------------------------>    
    
    if uploaded_file is not None:
        # Read the uploaded file
        data = df.copy()

        target_columns = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']

        feature_columns = ['lat', 'long']
        
        # Identifying numeric columns (excluding 'nm_pol' which might be a string)
        numeric_cols = [col for col in data.columns if data[col].dtype != 'object']

        # Preprocessing
        # Handling outliers: Remove outliers using IQR method for numeric columns
        Q1 = data[numeric_cols].quantile(0.25)
        Q3 = data[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Initializing separate scalers for features and target variables
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        # Splitting the dataset
        X = data[feature_columns]  # Features
        y = data[['lat', 'long'] + target_columns]   # Target variables (lat, long, and crime numbers)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit and transform the features
        X_train = feature_scaler.fit_transform(X_train)
        X_test = feature_scaler.transform(X_test)

        # Fit and transform the target variables
        y_train_scaled = target_scaler.fit_transform(y_train)
        y_test_scaled = target_scaler.transform(y_test)

        model = Sequential([
        Dense(128, activation='relu', input_shape=(2,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(9, activation='linear')  # Update to match the number of columns in target variable
        
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        model.fit(X_train, y_train_scaled, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluating the model
        y_pred_scaled = model.predict(X_test)
        mse = mean_squared_error(y_test_scaled, y_pred_scaled)
        rmse = mean_squared_error(y_test_scaled, y_pred_scaled, squared=False)

        st.subheader("Model Evaluation Metrics")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Root Mean Squared Error: {rmse}")

        # Inverse transform the scaled predictions to the original scale
        predictions_original_scale = pd.DataFrame(target_scaler.inverse_transform(y_pred_scaled), columns=['pred_lat', 'pred_long'] + target_columns)

        # Clip negative values to 0
        predictions_original_scale[['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']] = predictions_original_scale[['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']].clip(lower=0).astype(int)

        st.write("Predictions for Number of Crimes Per location")
        st.write(predictions_original_scale)


        # Inverse transform the scaled predictions to the original scale
        predictions_original_scale = pd.DataFrame(target_scaler.inverse_transform(y_pred_scaled), columns=['pred_lat', 'pred_long'] + target_columns)

        # Clip negative values to 0
        predictions_original_scale[['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']] = predictions_original_scale[['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']].clip(lower=0).astype(int)

#<----------------------------------------- Data Relation Chart------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->

       # Integration of Altair scatter plot code
        if uploaded_file is not None:

            data = df.copy()

            # Remove "lat" and "long" columns if they exist
            df = df.drop(["lat", "long"], axis=1, errors="ignore")

            # Select x and y-axis columns in the sidebar
            x_column = st.sidebar.selectbox("Select X-axis Column", df.columns, key="x_column")
            y_column = st.sidebar.selectbox("Select Y-axis Column", df.columns, key="y_column")

            # Check if exactly two columns are selected
            if x_column != y_column:
                # Scatter plot to compare the selected columns with custom colors
                st.write("### Data Relation Chart:")
                
                chart = alt.Chart(df).mark_circle().encode(
                    x=x_column,
                    y=y_column,
                    color=alt.Color("count()", scale=alt.Scale(range=['#FF69B4', '#ADD8E6'])),
                    tooltip=[x_column, y_column, "count()"]
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Please select two different columns for comparison.")


# <---------------------------------------------------- message sending code --------------------------------------- ----------------------------------------------------------------------->

    st.sidebar.header("Twilio SMS Sender")
    twilio_account_sid = st.sidebar.text_input("Twilio Account SID", "")
    twilio_auth_token = st.sidebar.text_input("Twilio Auth Token", "")
    twilio_phone_number = st.sidebar.text_input("Twilio Phone Number", "")
    recipient_phone_number = st.sidebar.text_input("Recipient Phone Number", "")

    # Create a button to trigger the modal inside the sidebar
    if st.sidebar.button("Report Suspicious Activity Anonymously"):
        # Create a modal dialog
        with st.sidebar.form(key="suspicious_activity_report_form"):
            st.header("Report Suspicious Activity Anonymously")
            st.write("Please provide details of the suspicious activity below:")
            report_text = st.text_area("Enter your anonymous suspicious activity report:")

            # Create a submit button within the modal
            submit_button = st.form_submit_button("Submit Report")

        # Handle the report submission outside the modal
        if submit_button:
            # Process and send the report (you can add your code here)
            # For testing purposes, you can print the report to the console
            print("Anonymous Suspicious Activity Report:")
            print(report_text)

 #<-------------------------------------------- Crime Against Woman Delhi Police Data ------------------------------------------------------------------------------------------------------------------------------>
    
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # Data manually extracted from the image
    data = {
        'Year': ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'],
        'RAPE': [706, 1636, 2166, 2199, 2153, 2168, 1699, 2076, 1033, 1100],
        'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY': [727, 3515, 4322, 5367, 3314, 2921, 2186, 2551, 1244, 1480],
        'INSULT TO THE MODESTY OF WOMEN': [214, 916, 1361, 1941, 599, 495, 434, 440, 229, 225],
        'KIDNAPPING OF WOMEN': [2048, 3286, 3604, 3738, 3482, 3471, 3761, 3758, 1880, 2197],
        'ABDUCTION OF WOMEN': [162, 323, 423, 556, 262, 201, 177, 325, 184, 105],
        'CRUELTY BY HUSBAND AND IN LAWS': [2046, 3045, 3194, 3536, 3416, 3792, 4557, 4731, 2096, 2704],
        'DOWRY DEATH': [134, 144, 153, 153, 116, 110, 141, 72, 69, 69],
        'DOWRY PROHIBITION ACT': [15, 15, 13, 20, 26, 16, 14, 16, 9, 7]
    }

    # Convert the dictionary to a pandas DataFrame and melt it for Plotly
    df = pd.DataFrame(data)
    df = df.melt(id_vars=['Year'], var_name='Crime', value_name='Cases')

    # Define a radio button in Streamlit
    option = st.sidebar.radio('Select a category:', ('Home', 'Crime Against Women', 'Crime News'))


    # Clear the main page when 'Crime Against Women' is selected
    if option == 'Crime Against Women':
        st.empty()  # This will clear anything that might have been on the screen before
        
        # Create the Plotly figure
        fig = px.bar(df, x='Year', y='Cases', color='Crime', title='Crime Against Women from 2012 to 2021')
        fig.update_layout(barmode='group')
        
        # Display the Plotly chart only
        st.plotly_chart(fig)

#<-------------------------------------- News Scrapping ----------------------------------------------------------------------->
    
    # Add the function to scrape news
    def fetch_crime_news():
        url = 'https://www.ndtv.com/topic/delhi-crime'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_containers = soup.find_all('div', class_='src_itm-ttl')
        news_articles = []
        for container in news_containers:
            news_link = container.find('a')
            if news_link:
                title = news_link.text.strip()
                link = news_link['href'].strip()
                news_articles.append({'title': title, 'url': link})
        return news_articles
    
    if option == 'Crime News':
        st.header("Latest Crime News")
        news_data = fetch_crime_news()
        for article in news_data:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.markdown("---")

        

       
else:
    st.markdown('''
        <div class="content">
            <p>"Sentinel Mark 2, the successor of its previous iteration, stands as a robust crime analysis and prediction tool. Engineered with precision, it harnesses cutting-edge analytics to forecast criminal activity with startling accuracy. This advanced tool aids law enforcement agencies in preemptive measures, ensuring public safety with proactive strategies. Its seamless integration with modern tech provides an unparalleled edge in the realm of crime prevention."</p>
        </div>
    ''', unsafe_allow_html=True)




