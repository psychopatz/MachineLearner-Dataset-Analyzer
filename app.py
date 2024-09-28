import streamlit as st
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai



# Set page configuration
st.set_page_config(
    page_title="MachineLearner's Dataset Analyzer",  # Custom app title
    page_icon="⚡",                   # You can use emojis or an image file as a favicon
    layout="centered",                 # Can be "centered" or "wide"
    initial_sidebar_state="expanded"   # Can be "expanded", "collapsed", or "auto"
)

# Define the path to the images
image_folder = os.path.join(os.path.dirname(__file__), "img")

# Load images
st.image(os.path.join(image_folder, "Tutorial1.png"), caption="Tutorial1")
st.image(os.path.join(image_folder, "Tutorial2.png"), caption="Tutorial2")
st.image(os.path.join(image_folder, "chatbot.png"), caption="chatbot")
st.image(os.path.join(image_folder, "user.png"), caption="user")


class KaggleDataUploader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df = None

    def set_kaggle_credentials(self, kaggle_json_content):
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
            f.write(kaggle_json_content)
        os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
        st.success("Kaggle API credentials authorized successfully.")

    def download_dataset(self):
        dataset_path = '.'
        os.system(f"kaggle datasets download -d {self.dataset_name} -p ./ --unzip")
        st.success(f"Dataset {self.dataset_name} downloaded successfully.")
        
        csv_filename = self._find_csv_file(dataset_path)
        if csv_filename:
            st.info(f"CSV file found: {csv_filename}")
            self.df = pd.read_csv(csv_filename)
            return self.df
        else:
            st.error("No CSV file found in the dataset.")

    def download_metadata(self):
        metadata_path = 'dataset-metadata.json'
        os.system(f"kaggle datasets metadata {self.dataset_name} -p ./")
        st.success(f"Metadata for {self.dataset_name} downloaded successfully.")
        
        with open(f'./{metadata_path}', 'r') as f:
            metadata = json.load(f)
            st.json(metadata)

    def _find_csv_file(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    return os.path.join(root, file)
        return None

class DataAnalyzer:
    def __init__(self):
        self.df = None
        self.df_cleaned = None
        self.numerical_df = None

    def set_data(self, df):
        self.df = df
        self.df_cleaned = self.df.copy()
        self.df_cleaned = self.df_cleaned.fillna(self.df_cleaned.mean(numeric_only=True))
        self.numerical_df = self.df_cleaned.select_dtypes(include=[np.number])

    def explore_data(self):
        st.subheader("Dataset Information")
        num_entries, num_columns = self.df.shape
        st.write(f"Number of Entries: {num_entries}")
        st.write(f"Number of Columns: {num_columns}")
        
        st.subheader("Data Columns")
        columns_info = self.df.dtypes.reset_index()
        columns_info.columns = ['Column', 'Data Type']
        non_null_counts = self.df.notnull().sum().values
        
        for idx, row in columns_info.iterrows():
            st.write(f"{row['Column']}: {non_null_counts[idx]} non-null {row['Data Type']}")

        st.subheader("Descriptive Statistics")
        st.dataframe(self.df.describe())

    def compute_statistics(self):
        st.subheader("Descriptive Statistics for Numerical Columns")
        stats_dict = {
            "Mean": self.numerical_df.mean(),
            "Median": self.numerical_df.median(),
            "Mode": self.numerical_df.mode().iloc[0],
            "Std Dev": self.numerical_df.std(),
            "Variance": self.numerical_df.var(),
            "Min": self.numerical_df.min(),
            "Max": self.numerical_df.max(),
            "Range": self.numerical_df.max() - self.numerical_df.min(),
            "25th Percentile": self.numerical_df.quantile(0.25),
            "50th Percentile": self.numerical_df.quantile(0.5),
            "75th Percentile": self.numerical_df.quantile(0.75)
        }

        stats_df = pd.DataFrame(stats_dict).transpose()
        st.dataframe(stats_df.style.format("{:.2f}"))

    def visualize_data(self):
        st.subheader("Visual Representation of Descriptive Statistics")

        st.write("Histograms:")
        fig, axes = plt.subplots(figsize=(12, 8))
        self.numerical_df.hist(edgecolor='black', bins=20, ax=axes)
        plt.suptitle('Histograms of Numerical Columns')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)

        st.write("Box Plots:")
        fig, axes = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=self.numerical_df, orient='h', ax=axes)
        plt.title('Box Plots of Numerical Columns', pad=20)
        plt.tight_layout()
        st.pyplot(fig)

        st.write("Correlation Matrix:")
        fig, axes = plt.subplots(figsize=(12, 10))
        correlation_matrix = self.numerical_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=axes)
        plt.title('Correlation Matrix', pad=20)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

comprehensive_analysis = ""
class Chatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = None
        self.chat_session = None
        self.set_api_key()

    def set_api_key(self):
        if not self.api_key:
            raise ValueError("API key not found. Please set the GOOGLE_API_KEY.")
        genai.configure(api_key=self.api_key)

    def create_model(self, knowledge):
        knowledge_input = "\n".join(knowledge)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=(
                "You are a professional Data Analyst and an expert at interpreting the results of Data Exploration. "
                "You are given a dataset's metadata, its table, and its results. Interpret this as detailed as possible. "
                "Make an Introduction, Key Statistics, Descriptive Statistics and insights "
                "and give your output in a proper markdown language format.\n"
                f"Here's the data given:\n{knowledge_input}"
            ),
        )

    def start_chat(self):
        self.chat_session = self.model.start_chat(history=[])

    def get_response(self, user_input: str):
        response = self.chat_session.send_message(user_input)
        return response.text


def load_dataset():
    global comprehensive_analysis
    st.title("Import your Kaggle Dataset")
    st.image("img/tutorial1.png", caption="Tutorial1")
    st.image("img/tutorial2.png", caption="Tutorial2")
    kaggle_command = st.text_input("Enter Kaggle API command (Example: kaggle datasets download -d hanaksoy/customer-purchasing-behaviors):")
    
    if st.button("Load Dataset"):
        if kaggle_command:
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            comprehensive_analysis = ""
            
            # Extract dataset name from the command
            dataset_name = kaggle_command.split()[-1]
            
            kaggle_data = KaggleDataUploader(dataset_name)
            
            # Set Kaggle credentials (you should securely handle this in a real application)
            kaggle_json_content = '''{
                    "username": "patz123456",
                    "key": "dfbe240463c495fd73afbd59042f34d9"
                }'''
            kaggle_data.set_kaggle_credentials(kaggle_json_content)
            
            # Download dataset
            df = kaggle_data.download_dataset()
            
            if df is not None:
                st.session_state.df = df
                st.session_state.dataset_loaded = True
                st.success("Dataset loaded successfully!")
            else:
                st.error("Failed to load the dataset. Please check your Kaggle command and try again.")
        else:
            st.warning("Please enter a valid Kaggle API command.")

def dashboard():
    st.title("Dataset Dashboard")
    
    if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first.")
        return
    
    df = st.session_state.df
    analyzer = DataAnalyzer()
    analyzer.set_data(df)
    
    # Display results
    analyzer.explore_data()
    analyzer.compute_statistics()
    analyzer.visualize_data()
    
    # Prepare knowledge for chatbot
    st.session_state.knowledge = [
        str(df.head()),
        str(df.describe()),
        str(df.info()),
        str(analyzer.numerical_df.corr())
    ]
    
    st.session_state.analysis_complete = True
    st.success("Analysis complete!")

class Chatbot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = None
        self.chat_history = []
        self.set_api_key()

    def set_api_key(self):
        if not self.api_key:
            raise ValueError("API key not found. Please set the GOOGLE_API_KEY.")
        genai.configure(api_key=self.api_key)

    def create_model(self, knowledge):
        knowledge_input = "\n".join(knowledge)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=(
                "You are a professional Data Analyst and an expert at interpreting the results of Data Exploration. "
                "You are given a dataset's metadata, its table, and its results. Interpret this as detailed as possible. "
                "Make an Introduction, Key Statistics, Descriptive Statistics and insights "
                "and give your output in a proper markdown language format.\n"
                f"Here's the data given:\n{knowledge_input}"
            ),
        )

    def get_response(self, user_input: str,isComprehensive = False):
        if isComprehensive:
            data = self.model.generate_content(user_input)
            return data.text
        
        response = self.model.generate_content(user_input)
        self.chat_history.append(("User", user_input))
        self.chat_history.append(("Assistant", response.text))
        return response.text

def load_dataset():
    st.title("Import your Kaggle Dataset ")
    st.image("img/tutorial1.png", caption="Tutorial1")
    st.image("img/tutorial2.png", caption="Tutorial2")
    kaggle_command = st.text_input("Enter Kaggle API command (Example: kaggle datasets download -d hanaksoy/customer-purchasing-behaviors):")
    
    load_button = st.button("Load Dataset", disabled=st.session_state.get('is_loading', False))
    
    if load_button:
        if kaggle_command:
            st.session_state.is_loading = True
            
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                if key != 'is_loading':
                    del st.session_state[key]
            
            # Extract dataset name from the command
            dataset_name = kaggle_command.split()[-1]
            
            with st.spinner("Loading dataset..."):
                kaggle_data = KaggleDataUploader(dataset_name)
                
                # Set Kaggle credentials (you should securely handle this in a real application)
                kaggle_json_content = '''{
                    "username": "patz123456",
                    "key": "dfbe240463c495fd73afbd59042f34d9"
                }'''
                kaggle_data.set_kaggle_credentials(kaggle_json_content)
                
                # Download dataset
                df = kaggle_data.download_dataset()
                
                if df is not None:
                    st.session_state.df = df
                    st.session_state.dataset_loaded = True
                    st.session_state.analysis_complete = False  # Reset analysis state
                    st.success("Dataset loaded successfully!")
                else:
                    st.error("Failed to load the dataset. Please check your Kaggle command and try again.")
            
            st.session_state.is_loading = False
        else:
            st.warning("Please enter a valid Kaggle API command.")
        

def dashboard():
    st.title("Dataset Dashboard")
    
    if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first.")
        return
    
    if not st.session_state.get('analysis_complete', False):
        if st.button("Analyze Data", disabled=st.session_state.get('is_analyzing', False)):
            st.session_state.is_analyzing = True
            
            with st.spinner("Analyzing data..."):
                df = st.session_state.df
                analyzer = DataAnalyzer()
                analyzer.set_data(df)
                
                # Display results
                analyzer.explore_data()
                analyzer.compute_statistics()
                analyzer.visualize_data()
                
                # Prepare knowledge for chatbot
                st.session_state.knowledge = [
                    str(df.head()),
                    str(df.describe()),
                    str(df.info()),
                    str(analyzer.numerical_df.corr())
                ]
                
                st.session_state.analysis_complete = True
                st.success("Analysis complete!")
            
            st.session_state.is_analyzing = False
    else:
        st.info("Analysis has already been completed. Load a new dataset to analyze again.")
        
        # Display the analysis results
        df = st.session_state.df
        analyzer = DataAnalyzer()
        analyzer.set_data(df)
        analyzer.explore_data()
        analyzer.compute_statistics()
        analyzer.visualize_data()

def chatbot():
    st.title("Dataset Chatbot")
    
    # Add CSS to make the input box and button float at the bottom
    st.markdown("""
        <style>
        .fixed-bottom {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            box-shadow: 0px -2px 10px rgba(0, 0, 100, 1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first.")
        return
    
    if 'analysis_complete' not in st.session_state or not st.session_state.analysis_complete:
        st.warning("Please complete the analysis in the Dashboard page first.")
        return
    
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            # Initialize chatbot
            google_api_key = "AIzaSyBlgD75p_eQ-doSPnNFoyrtDG1Z5BfSt-s"  # Replace with your actual Google API key
            st.session_state.chatbot = Chatbot(google_api_key)
            st.session_state.chatbot.create_model(st.session_state.knowledge)
    
    # Display chat history
    for role, message in st.session_state.chatbot.chat_history:
        if role == "User":
            col1, col2 = st.columns([6, 1])
            with col1:
                st.text_area("You:", value=message, height=100, max_chars=None, key=None, disabled=True)
            with col2:
                st.image("img/user.png", width=30)
        else:
            col1, col2 = st.columns([1, 6])
            with col1:
                st.image("img/chatbot.png", width=30)
            with col2:
                st.markdown(f" {message}")
    
    # Floating input box
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    user_input = st.text_input("Your question:")
    ask_button = st.button("Ask", disabled=st.session_state.get('is_asking', False))
    st.markdown('</div>', unsafe_allow_html=True)
    
    if ask_button:
        if user_input:
            st.session_state.is_asking = True
            with st.spinner("Generating response..."):
                response = st.session_state.chatbot.get_response(user_input)
                
                # Display user message
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.text_area("You:", value=user_input, height=100, max_chars=None, key=None, disabled=True)
                with col2:
                    st.image("img/user.png", width=50)
                
                # Display assistant response
                col1, col2 = st.columns([1, 6])
                with col1:
                    st.image("img/chatbot.png", width=50)
                with col2:
                    st.markdown(f"{response}")
            
            st.session_state.is_asking = False
        else:
            st.warning("Please enter a question.")


def getComprehensive_analysis():
    global comprehensive_analysis
    st.title("Comprehensive Analysis")
    if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first.")
        return
    
    if 'analysis_complete' not in st.session_state or not st.session_state.analysis_complete:
        st.warning("Please complete the analysis in the Dashboard page first.")
        return
    
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            # Initialize chatbot
            google_api_key = "AIzaSyBlgD75p_eQ-doSPnNFoyrtDG1Z5BfSt-s"  # Replace with your actual Google API key
            st.session_state.chatbot = Chatbot(google_api_key)
            st.session_state.chatbot.create_model(st.session_state.knowledge)
    
    # Display comprehensive analysis
    if comprehensive_analysis == "":
        comprehensive_analysis = st.session_state.chatbot.get_response("Explain the datasets to me comprehensively", isComprehensive = True)
    
    st.markdown(comprehensive_analysis)
    

def main():
    st.markdown("""
    <style>
    /* Customize the radio button text */
    .css-1cpxqw2 {
        font-size: 20px;            /* Font size */
        font-weight: bold;          /* Bold text */
        color: #ffffff;             /* White text for dark mode */
    }

    /* Customize the radio button container */
    .stRadio > div {
        background-color: #333333;  /* Dark gray background for the radio button container */
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #444444;  /* Subtle border to separate options */
    }

    /* Customize the selected radio option */
    input[type="radio"]:checked + div {
        background-color: #444444;  /* Darker gray for the selected option */
        color: #00ccff;             /* Light blue text for selected option */
    }
    
    
    </style>
    """, unsafe_allow_html=True)
    
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False
    if 'is_analyzing' not in st.session_state:
        st.session_state.is_analyzing = False
    if 'is_asking' not in st.session_state:
        st.session_state.is_asking = False
    
    st.sidebar.title("MachineLearner's Dataset Analyzer")
    page = st.sidebar.radio("Menu Options", ["Load Dataset", "Dataset Overview","Comprehensive Analysis", "Chatbot Assistant"])
    
    if page == "Load Dataset":
        load_dataset()
    elif page == "Dataset Overview":
        dashboard()
    elif page == "Comprehensive Analysis":
        getComprehensive_analysis()
    elif page == "Chatbot Assistant":
        chatbot()

if __name__ == "__main__":
    main()