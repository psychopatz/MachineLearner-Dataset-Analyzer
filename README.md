# Machine Learner Dataset Analyzer

A **Streamlit** app for analyzing Kaggle datasets. This application allows users to download datasets from Kaggle, perform data exploration, compute statistics, and visualize the results through intuitive charts and graphs.

## Features

- **Kaggle Dataset Import**: Import datasets directly from Kaggle using the Kaggle API.
- **Data Exploration**: Automatically compute and display dataset information, statistics, and descriptive summaries.
- **Visualization**: Visualize datasets using histograms, box plots, and correlation matrices.
- **AI-powered Assistant**: Leverage `google-generativeai` for enhanced insights and chatbot interaction.

## Installation

### Prerequisites

- **Python 3.7+**
- **Kaggle API**: You'll need to set up the Kaggle API on your machine. Follow the instructions [here](https://www.kaggle.com/docs/api).

### Step-by-Step Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/psychopatz/MachineLearner-Dataset-Analyzer.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd MachineLearner-Dataset-Analyzer
   ```

3. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate    # For Linux/Mac
   env\Scripts\activate       # For Windows
   ```

4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   The required libraries include:
   - `tabulate`
   - `seaborn`
   - `pandas`
   - `matplotlib`
   - `google-generativeai`
   - `kaggle`
   - `streamlit`

5. **Set up Kaggle API credentials**:
   - Download your `kaggle.json` API key from your Kaggle account.
   - Place the `kaggle.json` file in the appropriate directory:
     ```bash
     ~/.kaggle/kaggle.json     # For Linux/Mac
     C:\Users\<Username>\.kaggle\kaggle.json  # For Windows
     ```

6. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

   After running the above command, a browser window will open automatically, or you can visit `http://localhost:8501` in your browser.

## Usage

1. **Load Kaggle Dataset**: 
   - Enter the Kaggle dataset name or Kaggle API command to download and load the dataset.
   - Example: `kaggle datasets download -d hanaksoy/customer-purchasing-behaviors`.

2. **Dataset Overview**: 
   - Explore basic information about the dataset, including column details and summary statistics.

3. **Comprehensive Analysis**: 
   - View descriptive statistics, histograms, box plots, and a correlation matrix for numerical columns.

4. **Chatbot Assistant**: 
   - Interact with an AI-powered assistant for enhanced data insights.

## Requirements

Ensure the following Python packages are installed (included in `requirements.txt`):
- `tabulate`
- `seaborn`
- `pandas`
- `matplotlib`
- `google-generativeai`
- `kaggle`
- `streamlit`

## Project Structure

```
├── app.py                  # Main Streamlit application script
├── KaggleDataUploader.py    # Kaggle API interaction and dataset handling
├── DataAnalyzer.py          # Data analysis and visualization logic
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Contributions

Feel free to open issues or submit pull requests to contribute to the project.

## License

This project is licensed under the MIT License.

