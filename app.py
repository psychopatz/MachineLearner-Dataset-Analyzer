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
    page_title="MachineLearner's Dataset Analyzer",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the path to the images
image_folder = os.path.join(os.path.dirname(__file__), "img")

class KaggleDataUploader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.datasets_dir = "datasets"
        self.dataset_subdir = os.path.join(self.datasets_dir, self.dataset_name.replace('/', '_'))
        os.makedirs(self.dataset_subdir, exist_ok=True)

    def set_kaggle_credentials(self, kaggle_json_content):
        os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
        with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
            f.write(kaggle_json_content)
        os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

    def download_dataset(self):
        download_command = f"kaggle datasets download -d {self.dataset_name} -p {self.dataset_subdir} --unzip"
        os.system(download_command)
        
        csv_files = self._find_csv_files(self.dataset_subdir)
        if not csv_files:
            return None, "No CSV files found in the dataset."
        
        return csv_files, "Dataset downloaded successfully."

    def _find_csv_files(self, path):
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        return csv_files

    def download_metadata(self):
        metadata_path = os.path.join(self.dataset_subdir, f"dataset-metadata.json")
        metadata_command = f"kaggle datasets metadata {self.dataset_name} -p {self.dataset_subdir}"
        os.system(metadata_command)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            return None
            
class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.df_cleaned = df.fillna(df.mean(numeric_only=True))
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

    def visualize_data(self, filterDisplay=[]):
        # Define a mapping of visualizations to their identifiers
        visualizations = {
            "%HISTOGRAMS%": self.plot_histograms,
            "%BOXPLOTS%": self.plot_boxplots,
            "%CORRELATION MATRIX%": self.plot_correlation_matrix
        }

        # Check if filterDisplay has specific visualizations
        if filterDisplay:
            for key in filterDisplay:
                if key in visualizations:
                    visualizations[key]()  # Call the corresponding plot function
        else:
            # If filterDisplay is empty, display all visualizations
            st.subheader("Visual Representation of Descriptive Statistics")
            for key in visualizations:
                visualizations[key]()

    def plot_histograms(self):
        st.write("Histograms:")
        fig, axes = plt.subplots(figsize=(12, 8))
        self.numerical_df.hist(edgecolor='black', bins=20, ax=axes)
        plt.suptitle('Histograms of Numerical Columns')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)

    def plot_boxplots(self):
        st.write("Box Plots:")
        fig, axes = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=self.numerical_df, orient='h', ax=axes)
        plt.title('Box Plots of Numerical Columns', pad=20)
        plt.tight_layout()
        st.pyplot(fig)

    def plot_correlation_matrix(self):
        st.write("Correlation Matrix:")
        fig, axes = plt.subplots(figsize=(12, 10))
        correlation_matrix = self.numerical_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=axes)
        plt.title('Correlation Matrix', pad=20)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)


class Chatbot:
    def __init__(self,system_instruction=""):
        self.api_key = "AIzaSyBlgD75p_eQ-doSPnNFoyrtDG1Z5BfSt-s"
        self.model = None
        self.chat_history = []
        genai.configure(api_key=self.api_key)
        self.system_instruction = system_instruction

    def create_model(self):
                
        #Default System Instruction
        if not self.system_instruction:
            knowledge_input = "\n".join(st.session_state.knowledge)
            self.system_instruction = (
                "You are a professional Data Analyst and an expert at interpreting the results of Data Exploration. "
                "You are given a dataset's metadata, its table, and its results. Interpret this as detailed as possible. "
                "if the user asks about Visualizations like Histograms, Boxplots, Correlation Matrix, just use the findings as basis for the Visualizations. "
                "You must input your output in a proper markdown language format. use bold for the important information."
                "When dealing some statistics, always format it in markdown language"
                "At the end of your response, ask the user if they want some questions, add some emoji based on how you feel"
                f"\nHere's the data given:\n{knowledge_input}"
            )
            
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
            system_instruction= self.system_instruction,
        )

    def get_response(self, user_input: str, is_comprehensive=False):
        
        response = self.model.generate_content(user_input)
        if not is_comprehensive:
            self.chat_history.append(("User", user_input))
            self.chat_history.append(("Assistant", response.text))
        return response.text

def load_dataset():
    st.title("Import your Kaggle Dataset")
    st.image(os.path.join(image_folder, "tutorial1.png"), caption="Opening your Kaggle Dataset and press this button to open the menu")
    st.image(os.path.join(image_folder, "tutorial2.png"), caption="Click Copy API Command")
    kaggle_command = st.text_input("Enter Kaggle API command (Example: kaggle datasets download -d hanaksoy/customer-purchasing-behaviors):")
    
    # Initialize session state variables if they don't exist
    if 'csv_files' not in st.session_state:
        st.session_state.csv_files = []
    if 'selected_csv' not in st.session_state:
        st.session_state.selected_csv = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False

    # Load Kaggle dataset when 'Load Dataset' button is clicked
    if st.button("Load Dataset", disabled=st.session_state.get('is_loading', False)):
        if kaggle_command:
            st.session_state.is_loading = True
            
            dataset_name = kaggle_command.split()[-1]
            
            with st.spinner("Loading dataset..."):
                kaggle_data = KaggleDataUploader(dataset_name)
                kaggle_json_content = '''{
                    "username": "patz123456",
                    "key": "dfbe240463c495fd73afbd59042f34d9"
                }'''
                kaggle_data.set_kaggle_credentials(kaggle_json_content)
                csv_files, message = kaggle_data.download_dataset()
                
                if csv_files:
                    st.session_state.csv_files = csv_files
                    st.session_state.dataset_loaded = True
                    st.success(f"{message} Found {len(csv_files)} CSV files.")
                    
                    # Debug information
                    st.write("Debug: CSV files found:")
                    for csv_file in csv_files:
                        st.write(csv_file)
                    
                    # Download and display metadata
                    metadata = kaggle_data.download_metadata()
                    if metadata:
                        st.session_state['knowledge'] = ["dataset metadata\n```json\n" + json.dumps(metadata, indent=2) + "\n```"]
                    else:
                        st.warning("Metadata not found for the dataset.")
                else:
                    st.error(f"Failed to load the dataset. {message}")
            
            st.session_state.is_loading = False
        else:
            st.warning("Please enter a valid Kaggle API command.")

    # Display CSV selection if multiple CSV files are found
    if st.session_state.csv_files:
        st.subheader("Select CSV File")
        csv_options = [os.path.basename(csv) for csv in st.session_state.csv_files]
        
        # Debug information
        st.write(f"Debug: Number of CSV options: {len(csv_options)}")
        st.write("Debug: CSV options:", csv_options)
        
        selected_csv = st.selectbox("Choose a CSV file:", csv_options)
        
        if st.button("Load Selected CSV"):
            csv_path = next(csv for csv in st.session_state.csv_files if os.path.basename(csv) == selected_csv)
            try:
                df = pd.read_csv(csv_path)
                st.session_state.df = df
                st.session_state.df_edited = df.copy()
                st.session_state.selected_csv = csv_path
                st.session_state.analysis_complete = False
                st.success(f"Loaded CSV: {selected_csv}")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

    # Display current CSV information if available
    if st.session_state.get('selected_csv'):
        st.info(f"Currently loaded CSV: {os.path.basename(st.session_state.selected_csv)}")

    # If dataset is loaded, show additional options
    if 'df' in st.session_state and 'df_edited' in st.session_state:
        st.subheader("Edit Dataset")
        
        # Button to edit the CSV
        if st.button("Edit CSV"):
            st.session_state.df_edited = st.data_editor(st.session_state.df_edited, num_rows="dynamic")
        
        # Button to view changes
        if st.button("View Changes"):
            changes = st.session_state.df_edited.compare(st.session_state.df)
            if changes.empty:
                st.info("No changes detected.")
            else:
                st.write("Changes detected:")
                st.dataframe(changes)
        
        # Button to apply changes
        if st.button("Apply Changes"):
            st.session_state.df = st.session_state.df_edited.copy()
            st.success("Changes applied successfully!")
            st.session_state.analysis_complete = False

        # Display some information about the dataset
        st.subheader("Dataset Information")
        st.write(f"Number of rows: {len(st.session_state.df)}")
        st.write(f"Number of columns: {len(st.session_state.df.columns)}")
        st.write("Column names:", ", ".join(st.session_state.df.columns))

        # Display the first few rows of the dataset
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head())
def dashboard():
    st.title("Dataset Dashboard")
    
    if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first.")
        return
    
    if not st.session_state.get('analysis_complete', False):
        if st.button("Analyze Data", disabled=st.session_state.get('is_analyzing', False)):
            st.session_state.is_analyzing = True
            
            with st.spinner("Analyzing data..."):
                st.session_state.analyzer = DataAnalyzer(st.session_state.df)
                st.session_state.analyzer.explore_data()
                st.session_state.analyzer.compute_statistics()
                st.session_state.analyzer.visualize_data()
                
                # 1. Head of the DataFrame
                st.session_state['knowledge'].append("\nDataset Preview (First 5 rows):\n" + str(st.session_state.df.head()))
                # 2. Statistical Summary
                st.session_state['knowledge'].append("\nStatistical Summary of Numerical Columns:\n" + str(st.session_state.df.describe()))
                # 3. DataFrame Information
                st.session_state['knowledge'].append("\nDataFrame Information (Non-null counts, Data Types, Memory Usage):\n" + str(st.session_state.df.info()))
                # 4. Correlation Matrix
                st.session_state['knowledge'].append("\nNumerical Column Correlation Matrix:\n" + str(st.session_state.analyzer.numerical_df.corr()))
                
                # st.write(st.session_state['knowledge'][0])
                # st.write(st.session_state['knowledge'][1])
                # st.write(st.session_state['knowledge'][2])
                # st.write(st.session_state['knowledge'][3])
                # st.write(st.session_state['knowledge'][4])
                
                st.session_state.analysis_complete = True
                st.success("Analysis complete!")
            
            st.session_state.is_analyzing = False
    else:
        st.info("Analysis has already been completed. Load a new dataset to analyze again.")
        st.session_state.analyzer = DataAnalyzer(st.session_state.df)
        st.session_state.analyzer.explore_data()
        st.session_state.analyzer.compute_statistics()
        st.session_state.analyzer.visualize_data()

def chatbot():
    st.title("Dataset Chatbot")
    
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
            st.session_state.chatbot = Chatbot()
            st.session_state.chatbot.create_model()
    
    for role, message in st.session_state.chatbot.chat_history:
        col1, col2 = st.columns([6, 1]) if role == "User" else st.columns([1, 6])
        with col1:
            if role == "User":
                st.text_area("You:", value=message, height=100, max_chars=None, key=None, disabled=True)
            else:
                st.image(os.path.join(image_folder, "chatbot.png"), width=30)
        with col2:
            if role == "User":
                st.image(os.path.join(image_folder, "user.png"), width=30)
            else:
                st.markdown(f" {message}")
    
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    user_input = st.text_input("Your question:")
    ask_button = st.button("Ask", disabled=st.session_state.get('is_asking', False))
    st.markdown('</div>', unsafe_allow_html=True)
    
    if ask_button and user_input:
        st.session_state.is_asking = True
        with st.spinner("Generating response..."):
            response = st.session_state.chatbot.get_response(user_input)
            
            col1, col2 = st.columns([6, 1])
            with col1:
                st.text_area("You:", value=user_input, height=100, max_chars=None, key=None, disabled=True)
            with col2:
                st.image(os.path.join(image_folder, "user.png"), width=35)
            
            col1, col2 = st.columns([1, 6])
            with col1:
                st.image(os.path.join(image_folder, "chatbot.png"), width=35)
            with col2:
                st.markdown(f"{response}")
        
        st.session_state.is_asking = False
    elif ask_button:
        st.warning("Please enter a question.")
        
def formatMessage(message):
    # Mapping keywords to their display format
    visual_mapping = {
        "%HISTOGRAMS%": ["plot_histograms"],
        "%BOXPLOTS%": ["plot_boxplots"],
        "%CORRELATION MATRIX%": ["plot_correlation_matrix"]
    }
    
    formatted_parts = []
    current_index = 0
    
    while current_index < len(message):
        next_keyword_index = len(message)
        next_keyword = None
        
        for keyword in visual_mapping.keys():
            keyword_index = message.find(keyword, current_index)
            if keyword_index != -1 and keyword_index < next_keyword_index:
                next_keyword_index = keyword_index
                next_keyword = keyword
        
        if next_keyword:
            # Add text before the keyword
            if current_index != next_keyword_index:
                formatted_parts.append(("text", message[current_index:next_keyword_index].strip()))
            
            # Add the visualization
            formatted_parts.append(("visualization", next_keyword))
            current_index = next_keyword_index + len(next_keyword)
        else:
            # Add remaining text
            formatted_parts.append(("text", message[current_index:].strip()))
            break
    
    # Display formatted data and visualize
    for part_type, content in formatted_parts:
        if part_type == "text" and content:
            st.markdown(content)
        elif part_type == "visualization":
            with st.spinner(f"Generating visualization for {content}..."):
                for method_name in visual_mapping[content]:
                    method = getattr(st.session_state.analyzer, method_name)
                    method()
            st.write("")  # Add space after visualization
            

            
def get_comprehensive_analysis():
    st.title("Comprehensive Analysis")
    
    if 'dataset_loaded' not in st.session_state or not st.session_state.dataset_loaded:
        st.warning("Please load a dataset first.")
        return
    
    if 'analysis_complete' not in st.session_state or not st.session_state.analysis_complete:
        st.warning("Please complete the analysis in the Dashboard page first.")
        return
    
    if 'chatbot' not in st.session_state:
        knowledge_input = "\n".join(st.session_state.knowledge)
        system_instruction = (
                "You are a professional Data Analyst and an expert at interpreting the results of Data Exploration. "
                "You are given a dataset's metadata, its table, and its results. Interpret this as detailed as possible. "
                "In Visualizations and Interpretations part, replace the Histogram Title to %HISTOGRAMS% and the same for Boxplots its %BOXPLOTS% and %CORRELATION MATRIX% for Correlation Matrix."
                "The display must be as detailed as possible and in this order: Introduction, Key Statistics, Descriptive Statistics,Visualizations and Interpretations, insights and conclusions. "
                "You must input your output in a proper markdown language format\n"
                f"Here's the data given:\n{knowledge_input}"
            )
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = Chatbot(system_instruction)
            st.session_state.chatbot.create_model()
    
    # Check if we need to generate a new comprehensive analysis
    if 'comprehensive_analysis' not in st.session_state or st.session_state.get('new_dataset_loaded', False):
        with st.spinner("Generating comprehensive analysis..."):
            st.session_state.comprehensive_analysis = st.session_state.chatbot.get_response("Explain the datasets to me comprehensively", is_comprehensive=True)
        st.session_state.new_dataset_loaded = False  # Reset the flag
    
    formatMessage(st.session_state.comprehensive_analysis)
    


def main():
    st.markdown("""
    <style>
    .css-1cpxqw2 {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
    }
    .stRadio > div {
        background-color: #333333;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #444444;
    }
    input[type="radio"]:checked + div {
        background-color: #444444;
        color: #00ccff;
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
    page = st.sidebar.radio("Menu Options", ["Load Dataset", "Dataset Overview", "Comprehensive Analysis", "Chatbot Assistant"])
    
    if page == "Load Dataset":
        load_dataset()
    elif page == "Dataset Overview":
        dashboard()
    elif page == "Comprehensive Analysis":
        get_comprehensive_analysis()
    elif page == "Chatbot Assistant":
        chatbot()

if __name__ == "__main__":
    main()