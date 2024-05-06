import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import hydralit_components as hc
import base64
import json
import requests  
from streamlit_lottie import st_lottie
import pydeck as pdk
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Set Page Icon, Title, and Layout
st.set_page_config(layout="wide", page_title="Capstone Project")

# Function to load a local CSS file and apply it within a Streamlit app
def local_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("CSS file not found.")

# Function to display Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Apply the local CSS for the overall style
local_css("styles.css")

# Additional styles for the navigation bar
nav_bar_styles = """
<style>
.nav-bar {
  background-color: #007bff; /* Blue */
  overflow: hidden;
  margin-top: -100px; /* Adjust the negative margin to reduce space */
}
.nav-link {
  color: white;
  padding: 10px 15px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
}
.nav-bar a {
  float: left;
  display: block;
  color: white;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
}
.nav-bar a:hover {
  background-color: #ddd;
  color: black;
}
</style>
"""
# Inject additional styles for the navigation bar into the app
st.markdown(nav_bar_styles, unsafe_allow_html=True)


# Function to convert image to Base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error("Logo image file not found. Please check the path.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Specify the correct path to your logo image
logo_path = r"logo.jpeg"
logo_base64 = get_image_base64(logo_path)


# Your navigation bar with the logo
nav_bar_html = f"""
<div class="nav-bar">
  <a class="nav-link" href="#home">🏠 Home</a>
  <a class="nav-link" href="#About">📊 About</a>
  <a class="nav-link" href="#Adresses Database">🔍 EDA </a>
  <a class="nav-link" href="#Classfication">🧠 Risks</a>
  <a class="nav-link" href="#Segmentation">🧠 Risks</a>
  <div style="float: right;">
    <img src="data:image/jpeg;base64,{logo_base64}" alt="logo" style="height: 60px;"> <!-- Adjust height as necessary -->
  </div>
</div>
"""


# Injecting the customized navigation bar with the logo into the app
st.markdown(nav_bar_html, unsafe_allow_html=True)

# Navigation Bar Design
menu_data = [
    {'label': "Home", 'icon': "bi bi-house"},
    {'label': "About", 'icon': "bi bi-clipboard-data"},
    {'label': 'Adresses Database', 'icon' : "bi bi-pin-map"},
    {'label': 'Classification', 'icon' : "bi bi-person-lines-fill"},
    {'label': 'Segmentation', 'icon' : "bi bi-segmented-nav"}
]

# Customize theme for the hydralit navbar
over_theme = {'txc_inactive': 'white', 'menu_background': '#0178e4', 'option_active': 'white'}
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=True,
    sticky_nav=True,
    sticky_mode='sticky',
)
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if menu_id == 'Home':
  # st.title("<h1 style='text-align:Welcome to RiskAdvisor",)
  st.markdown("<h1 style='text-align: center; color: black;'>Welcome to RiskAdvisor<i class='bi bi-heart-fill' style='color: red;'></i> </h2>", unsafe_allow_html=True)
  st.markdown("<hr style='border-top: 3px solid black;'>", unsafe_allow_html=True)
  # Adding custom style with Bradley Hand ITC font
  st.markdown("""
    <style>
      @font-face {
        font-family: 'Bradley Hand ITC';
        src: url('https://path-to-your-font/BradleyHandITC.ttf') format('truetype');
      }
      .bradley-font {
        font-family: 'Bradley Hand ITC';
        color: blue;
      }
    </style>
              
  <h2 style='text-align: center;' class="bradley-font">The tool that will help Securite Assurance assess the risk.</h2>
""", unsafe_allow_html=True)
# Load the Lottie animation
  lottie_hello = load_lottieurl("https://lottie.host/30cc4871-e0a0-4a9a-a32f-c667e9c1a9b2/15hDV4tBTu.json")

  # Create three columns with the middle one containing the Lottie animation
  col1, col2, col3 = st.columns([1,2,1])

  with col2:  # This is the middle column
      st_lottie(
          lottie_hello,
          speed=1,
          reverse=False,
          loop=True,
          quality="low",
          width=800  ,
          height=500 ,
          key=None,
      )
  st.markdown("""
  <div style="font-family: Arial; font-size: 25px; font-weight: bold; text-align: justify">
    A key challenge we face at Securite Assurance involves evaluating new risks in unfamiliar Lebanese cities. The goal of 
            this app is to mitigate this issue by employing machine learning techniques. These techniques will enable us to
            categorize each city in Lebanon as a low, medium, or high risk related to property and casualty insurance in Lebanon.
            By doing so, we can swiftly and accurately assess the risk level of any new city and give an appropriate price for 
            the premises guided by a model that continuously evolves through learning from past data. In addition, we
            will employ supervised learning techniques to develop a predictive model that forecasts the likelihood of a claim to happen. 
  </div> """, unsafe_allow_html=True)



if menu_id == 'About':
    # Insert a horizontal line across the page
    st.markdown("---")

    # Create two columns with a 2:1 width ratio
    col1, col2 = st.columns((2,1))

    with col1:
        # Content for the 'Securite Assurance' section on the left
        st.header("Securite Assurance")
        st.markdown("""
        <div style="text-align: justify">
        Founded in 1955 as the representative of Union des Assurances de Paris (UAP), Securite Assurance has grown to become one of Lebanon's 
        leading insurance companies. Throughout the decades, particularly during a transformative period in the 1990s, the company 
        has expanded its reach and influence across the nation. Today, Securite Assurance stands as a towering figure in the Lebanese
        insurance industry, issuing over 400,000 policies annually and insuring assets valued at more than a billion USD. With more
        than 10 branches spread across Lebanon and a dedicated team of over 350 insurance specialists, the company offers a wide range
        of services designed to meet the diverse needs of its clients.
                    
        </div>
        """, unsafe_allow_html=True)
        # Content for the 'Creator' section on the right
        st.markdown("---")
        st.header("Web Application Creator")
        st.write("Name: Emilia El Khoury")
        st.write("Education: 'Bachelor in Actuarial Sciences' and 'Master of Sciences in Business and Data Analytics'")
        st.write("Contact: emilia.khoury1@gmail.com or [Linkedin](https://www.linkedin.com/in/emilia-el-khoury/)")  

    st.write("Thanks for stopping by!")
    # Insert a horizontal line across the page
    st.markdown("---")
    with col2:
        # Load and display the logo image
        st.write("\n")
        st.write("\n")
        logo_path = r"logo2.png"
        try:
            image = Image.open(logo_path)
            st.image(image, width = 200)
        except FileNotFoundError:
            st.error("Logo image file not found. Please check the path.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")
        
        # Load and display the logo image
        logo_path = r"image3.png"
        try:
            image = Image.open(logo_path)
            st.image(image, width=200)
        except FileNotFoundError:
            st.error("Logo image file not found. Please check the path.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")  
#https://similobeta2.streamlit.app/#welcome-to-similo
#https://smartprep.streamlit.app/
#https://gpt-dataset-generator.streamlit.app/



# Load your data from an Excel file
data = pd.read_excel('streamlit locations.xlsx')
data2 = pd.read_excel('indemnity_streamlit.xlsx')
# st.write(data2)  # Uncomment this line if you want to display data2 in your app for debugging

if menu_id == 'Adresses Database':  
    st.markdown("<h3 style='text-align: Left; color: black;'>Choose The Location you need:<i class='bi bi-heart-fill' style='color: red;'></i> </h3>", unsafe_allow_html=True)


    # Dropdown to select an address
    sorted_addresses = sorted(data['Risk Adress'].unique())
    address_to_view = st.selectbox('Select an address to zoom in', sorted_addresses)
    # address_to_view = st.selectbox('Select an address to zoom in', data['Risk Adress'].unique())

    # Filter data to get the selected address
    selected_data = data[data['Risk Adress'] == address_to_view]

    # Show the map zoomed to the selected address
    st.map(selected_data)
    st.markdown("### Historical Claims Experience")
      # Create two columns with a 2:1 width ratio
    col1, col2 = st.columns((2,1))
    with col1:
            # Filter data2 by the selected address for 'Risk Adress'
            filtered_data2 = data2[data2['Risk Adress'] == address_to_view]

            # Generate a pie chart for 'Cause of Loss'
            if not filtered_data2.empty:
                  cause_of_loss_counts = filtered_data2['Cause of Loss'].value_counts()
                  fig, ax = plt.subplots()
                  wedges, texts, autotexts = ax.pie(cause_of_loss_counts, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})  # Smaller font size
                  ax.legend(wedges, cause_of_loss_counts.index, title="Cause of Loss", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
                  ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                  st.pyplot(fig)
            else:
                st.write("No data available for the selected address.")

    with col2:
        st.markdown("")        
        st.markdown("")
        st.markdown("")
        st.markdown("")
        
        # Filter data2 by the selected address for 'Risk Adress'
        filtered_data2 = data2[data2['Risk Adress'] == address_to_view]

        # Generate a histogram for 'Branch'
        if not filtered_data2.empty:
            fig, ax = plt.subplots()
            # Histogram plot
            hist_data = sns.histplot(filtered_data2['Branch '], kde=False, color='blue', ax=ax, binwidth=1)
            ax.set_xlabel('Branch ')
            ax.set_ylabel('Count')
            ax.set_title('Policies Type Related to This Address')  

            # Highlight the highest bar
            max_height = max([p.get_height() for p in hist_data.patches])  # Get the maximum bar height
            for p in hist_data.patches:  # Iterate through each bar
                if p.get_height() == max_height:  # If this bar's height is the maximum
                    p.set_color('red')  # Set the color of the bar to red

            st.pyplot(fig)
        else:
            st.write("No data available for the selected address.")



    col1, col2 = st.columns((2,1))        
    with col1:
  

      # Load your data
      data2 = pd.read_excel('indemnity_streamlit.xlsx')
      print("Data Loaded:", data2.head())

      # Ensure 'Loss Date' is a datetime type
      data2['Loss Date'] = pd.to_datetime(data2['Loss Date'])

      # Extract day of the week and year from 'Loss Date'
      data2['Day of Week'] = data2['Loss Date'].dt.day_name()
      data2['Year'] = data2['Loss Date'].dt.year

      # Group by 'Day of Week' and 'Year' to count occurrences
      grouped_data = data2.groupby(['Year', 'Day of Week']).size().reset_index(name='Count')

      # Sort days of the week in order
      days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
      grouped_data['Day of Week'] = pd.Categorical(grouped_data['Day of Week'], categories=days, ordered=True)
      grouped_data.sort_values('Day of Week', inplace=True)

      print("Grouped Data:", grouped_data)

      # Streamlit slider for selecting the year
      year_to_filter = st.slider('Select a year', int(grouped_data['Year'].min()), int(grouped_data['Year'].max()), int(grouped_data['Year'].max()))
      print("Year Selected:", year_to_filter)

      # Filter data based on selected year
      filtered_data = grouped_data[grouped_data['Year'] == year_to_filter]
      print("Filtered Data:", filtered_data)


  # Plotting
      if not filtered_data.empty:
          fig, ax = plt.subplots()
          ax.plot(filtered_data['Day of Week'], filtered_data['Count'], marker='o')
          ax.set_xlabel('Day of the Week')
          ax.set_ylabel('Count')
          ax.set_title('Counts of Claims by Day of the Week')
          st.pyplot(fig)
      else:
          st.write("No data available for the selected year.")
 
# if menu_id == "Classification":
#     # pickle_in = open('classifier.pkl', 'rb') 
#     # classifier = pickle.load(pickle_in)

#     def display_header():
#       st.markdown("<h1 style='text-align: center; color: black;'>Claims Classification Tool<i class='bi bi-heart-fill' style='color: red;'></i></h1>", unsafe_allow_html=True)
#       st.markdown("<hr style='border-top: 3px solid black;'>", unsafe_allow_html=True)
#       st.subheader("This classification tool will predict the likelihood of a policyholder causing a claim to Securite, based on various significant factors identified during the model deployment and assessment phase.")

#     def collect_user_input():
#       branch = st.number_input('Branch number:', min_value=1, value=1, step=1)
#       total_sum_insured = st.number_input('Total sum insured:', min_value=0.01, value=100000.00, step=0.01)
#       net_premium = st.number_input('Net premium:', min_value=1, value=1000, step=1)
#       active_policy = st.selectbox("Is the policy still active?", ("Yes", "No"))
#       active_policy = 1 if active_policy == "Yes" else 0
#       policy_counts = st.number_input('Number of endorsements:', min_value=0, value=0, step=1)
#       count_of_addresses = st.number_input('Number of locations:', min_value=1, value=1, step=1)
#       Effective_date = st.date_input("Effective date:")
#       year_eff = Effective_date.year
#       st.write(f"Effective Year: {year_eff}")  # Display the effective year
#       Expiry_date = st.date_input("Expiry date:")
#       policy_duration = (Expiry_date - Effective_date).days
#       st.write(f"Policy Duration: {policy_duration} days")  # Display the policy duration

#       return branch, total_sum_insured, net_premium, active_policy, policy_counts, count_of_addresses, policy_duration, year_eff

#     def setup_data_frame():
#         policy_types = ["FIR", "MUC", "MUP", "MUT", "PAR"]
#         currencies = ["LBP", "USD", "USF", "USL"]
#         categories = ["C", "I", "Other"]
#         endorsement_codes = ["AC", "AE", "AV", "CI", "CM", "CP", "DA", "DM", "DV", "DY", "ES", "MC", "PP", "PR", "RI", "RN", "RV", "SP"]
        
#         columns = [f'Policy _{type}' for type in policy_types] + \
#                     [f'Currency_{cur}' for cur in currencies] + \
#                     [f'Category_{cat}' for cat in categories] + \
#                     [f'EndCod_{code}' for code in endorsement_codes]
#         user_data = pd.DataFrame(0, index=np.arange(1), columns=columns)

#         user_data[f'Policy _{st.selectbox("Select the policy type:", policy_types)}'] = 1
#         user_data[f'Currency_{st.selectbox("Select the currency:", currencies)}'] = 1
#         user_data[f'Category_{st.selectbox("Select the category:", categories)}'] = 1
#         user_data[f'EndCod_{st.selectbox("Select the endorsement code:", endorsement_codes)}'] = 1
        
#         return user_data

#     def main():
#         display_header()
#         user_inputs = collect_user_input()
#         user_data = setup_data_frame()

#     st.markdown(" ")
#     if __name__ == '__main__':
#         main()
#     st.markdown(" ")
#     st.markdown(" ")
#     st.markdown(" ")
#     st.markdown(" ")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


if menu_id == "Classification":


# Load data
    train_data = pd.read_csv('dataset_train.csv')
    test_data = pd.read_csv('dataset_test.csv')

# Extract features and target
    X_train = train_data[['Branch ', 'Total Sum Insured', 'NetPemium', 'Policy Duration', 'Year_eff', 'Active Policy',
                      'policy_counts', 'count_of_addresses', 'Policy _FIR', 'Policy _MUC', 'Policy _MUP', 'Policy _MUT',
                      'Policy _PAR', 'Currency_LBP', 'Currency_USD', 'Currency_USF', 'Currency_USL', 'Category_C',
                      'Category_I', 'Category_Other', 'EndCod_AC', 'EndCod_AE', 'EndCod_AV', 'EndCod_CI', 'EndCod_CM',
                      'EndCod_CP', 'EndCod_DA', 'EndCod_DM', 'EndCod_DV', 'EndCod_DY', 'EndCod_ES', 'EndCod_MC',
                      'EndCod_PP', 'EndCod_PR', 'EndCod_RI', 'EndCod_RN', 'EndCod_RV', 'EndCod_SP']]
    y_train = train_data['target']

    # Model configuration (adjust these parameters as necessary)
    params = {
        'criterion': 'friedman_mse',
        'learning_rate': 0.1,
        'loss': 'exponential',
        'max_depth': 3,
        'max_features': 'sqrt',
        'min_samples_leaf': 2,
        'min_samples_split': 5,
        'n_estimators': 100,
        'subsample': 0.8,
        'warm_start': False
    }

    # Pipeline for scaling and classification
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(**params))
    ])

# Model training
    pipeline.fit(X_train, y_train)

    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier

    def display_header():
        st.title("Claims Classification Tool")
        st.markdown("### This tool predicts the likelihood of a policyholder causing a claim based on various significant factors.")

    def setup_data_frame():
        policy_types = ["FIR", "MUC", "MUP", "MUT", "PAR"]
        currencies = ["LBP", "USD", "USF", "USL"]
        categories = ["C", "I", "Other"]
        endorsement_codes = ["AC", "AE", "AV", "CI", "CM", "CP", "DA", "DM", "DV", "DY", "ES", "MC", "PP", "PR", "RI", "RN", "RV", "SP"]
        
        user_data = pd.DataFrame(0, index=np.arange(1), columns=[
            f'Policy _{type}' for type in policy_types] + 
            [f'Currency_{cur}' for cur in currencies] + 
            [f'Category_{cat}' for cat in categories] + 
            [f'EndCod_{code}' for code in endorsement_codes]
        )
        
        user_data[f'Policy _{st.selectbox("Select the policy type:", policy_types)}'] = 1
        user_data[f'Currency_{st.selectbox("Select the currency:", currencies)}'] = 1
        user_data[f'Category_{st.selectbox("Select the category:", categories)}'] = 1
        user_data[f'EndCod_{st.selectbox("Select the endorsement code:", endorsement_codes)}'] = 1
        
        return user_data

    def collect_user_input():
        branch = st.number_input('Branch number:', min_value=1, value=1, step=1)
        total_sum_insured = st.number_input('Total sum insured:', min_value=0.01, value=100000.00, step=0.01)
        net_premium = st.number_input('Net premium:', min_value=1, value=1000, step=1)
        active_policy = st.selectbox("Is the policy still active?", ("Yes", "No"))
        active_policy = 1 if active_policy == "Yes" else 0
        policy_counts = st.number_input('Number of endorsements:', min_value=0, value=0, step=1)
        count_of_addresses = st.number_input('Number of locations:', min_value=1, value=1, step=1)
        Effective_date = st.date_input("Effective date:")
        year_eff = Effective_date.year
        st.write(f"Effective Year: {year_eff}")  # Display the effective year
        Expiry_date = st.date_input("Expiry date:")
        policy_duration = (Expiry_date - Effective_date).days
        st.write(f"Policy Duration: {policy_duration} days")  # Display the policy duration
        
        user_input_data = [branch, total_sum_insured, net_premium, policy_duration, year_eff, active_policy,
                        policy_counts, count_of_addresses]
        user_data_frame = setup_data_frame()
        
        return user_input_data + user_data_frame.iloc[0].tolist()

    def make_prediction(user_inputs):
        df = pd.DataFrame([user_inputs], columns=X_train.columns)  # Ensure column names match training features
        prediction = pipeline.predict(df)
        return prediction

    def main():
        display_header()
        user_inputs = collect_user_input()
        if st.button("Predict Claim Likelihood"):
            prediction = make_prediction(user_inputs)
            st.write(f"Prediction: {'Claim Occurrence' if prediction[0] == 1 else 'No claim Occurrence'}")
            

    if __name__ == "__main__":
        main()

    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")






if menu_id == "Segmentation":

    # Import necessary libraries
    import pickle
    import numpy as np
    import streamlit as st

    # Load the models
    with open('seg.pkl', 'rb') as pickle_in:
        clustering = pickle.load(pickle_in)

    with open('scaler.pkl', 'rb') as pickle_in:
        scaler = pickle.load(pickle_in)

    # Web page headers and markdowns
    st.markdown("<h1 style='text-align: center; color: black;'>Risk Address Segmentation Tool<i class='bi bi-heart-fill' style='color: red;'></i></h2>", unsafe_allow_html=True)
    st.markdown("<hr style='border-top: 3px solid black;'>", unsafe_allow_html=True)
    st.subheader("This segmentation tool will cluster the addresses in Lebanon into 3 segments based on their level of Risk.")
    st.markdown("<h5>The classified risk will be related to three main Insurance Metrics:<i class='bi bi-heart-fill'</h5>", unsafe_allow_html=True)
    st.markdown("""
    - **Total Sum Insured**: The total amount of insurance coverage across all policies.
    - **Final Indemnity**: The total amount paid out in claims.
    - **Loss Ratio**: A ratio of claims paid to premiums received.
    """)

    # Prediction function
    def prediction(Total_sum_insured, Final_indemnity, Loss_Ratio):
        # Prepare input features
        Total_sum_insured = np.array(Total_sum_insured).reshape(-1, 1)
        Final_indemnity = np.array(Final_indemnity).reshape(-1, 1)
        Loss_Ratio = np.array(Loss_Ratio).reshape(-1, 1)
        X = np.concatenate((Total_sum_insured, Final_indemnity, Loss_Ratio), axis=1)
        X_scaled = scaler.transform(X)

        # Predict using the clustering model
        prediction = clustering.predict(X_scaled)
        if prediction == 0:
            pred = 'LOW'
        elif prediction == 1:
            pred = 'MEDIUM'
        elif prediction == 2:
            pred = 'HIGH'
        return pred

    # Main function to interact with the user
    def main():
        Total_sum_insured = st.number_input('Enter the total Sum insured', min_value=1, value=1, step=1)
        Final_indemnity = st.number_input('Enter the Final Indemnity paid for the customers', min_value=0.01, value=0.01, step=0.01)
        Loss_Ratio = st.number_input("What was the overall Loss Ratio", min_value=0.0, value=1.0, step=0.01)
        result = ""

        # Predict and display the results
        if st.button("Predict"):
            result = prediction(Total_sum_insured, Final_indemnity, Loss_Ratio)
            st.success('Your cluster is {}'.format(result))
            print(clustering.labels_)
        st.markdown(" ")
        st.markdown(" ")
    
        st.markdown(" ")
    if __name__ == '__main__':
        main()
    st.markdown(" ")
    st.markdown(" ")

