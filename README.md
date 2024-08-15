# Wildfire Prediction Challenge

## Project Overview

Wildfires are a major environmental issue, particularly in Africa where they can cause extensive ecological damage and pose a risk to human life and property. Understanding the factors that contribute to wildfire spread and predicting the burned area can help in mitigating these risks.

This project aims to develop a machine-learning model that predicts the burned area in different locations across Africa for the years 2014 to 2016. The model is based on the XGBoost algorithm and is fine-tuned using RandomizedSearchCV to optimize performance.

## Folder Structure and File Locations

### Setting Up the Project in Google Drive

To run this project using Google Colab, you need to organize your files and folders within Google Drive as follows:


### Explanation of Files:
- **Train.csv**: Contains the training data, including features and the target variable `burn_area`.
- **Test.csv**: Contains the test data for which predictions need to be made.
- **SampleSubmission.csv**: A template submission file that outlines the required format for your predictions.
- **preprocessing.ipynb**: Jupyter notebook that handles data loading, feature extraction, and preprocessing.
- **model_training.ipynb**: Jupyter notebook that contains the code for model training, hyperparameter tuning, and evaluation.
- **submission.ipynb**: Jupyter notebook used for generating predictions on the test set and preparing the final submission file.
- **README.md**: The project documentation file you are currently reading.

## Order of Execution

### Step-by-Step Instructions

1. **Upload Files to Google Drive**:
   - First, upload the entire `wildfire-prediction` directory to your Google Drive under the `My Drive` root folder. This directory should include all the required CSV files (`Train.csv`, `Test.csv`, `SampleSubmission.csv`) and the Jupyter notebooks (`preprocessing.ipynb`, `model_training.ipynb`, `submission.ipynb`).

2. **Run the Notebooks in Sequence**:
   - **preprocessing.ipynb**:
     - **Purpose**: This notebook handles the loading and preprocessing of the dataset. It includes tasks such as parsing dates from the `ID` field, extracting year, month, and day as features, and standardizing the data.
     - **Expected Runtime**: Approximately 5 minutes.
     - **Execution**:
       1. Open the `preprocessing.ipynb` notebook in Google Colab.
       2. At the beginning of the notebook, mount your Google Drive by running:
          ```python
          from google.colab import drive
          drive.mount('/content/drive')
          ```
       3. Ensure that the file paths in the notebook point to your Google Drive location:
          ```python
          train = pd.read_csv('/content/drive/My Drive/wildfire-prediction/Train.csv')
          ```
       4. Run all cells in the notebook to complete the preprocessing steps.

   - **model_training.ipynb**:
     - **Purpose**: This notebook is responsible for training the XGBoost model. It includes hyperparameter tuning using RandomizedSearchCV, evaluating model performance using cross-validation, and selecting important features.
     - **Expected Runtime**: Approximately 30 minutes, depending on the available hardware (e.g., Colab GPU vs. local CPU).
     - **Execution**:
       1. Open the `model_training.ipynb` notebook in Google Colab.
       2. As before, ensure that your Google Drive is mounted and file paths are correctly set.
       3. Run all cells to train the model and select the best hyperparameters.

   - **submission.ipynb**:
     - **Purpose**: This notebook generates predictions for the test set and formats the results into a submission file. The predictions are constrained to be within the range (0, 1) as required by the competition guidelines.
     - **Expected Runtime**: Approximately 5 minutes.
     - **Execution**:
       1. Open the `submission.ipynb` notebook in Google Colab.
       2. Verify that the Google Drive paths are correct and run the cells to generate the final submission file.
       3. The generated CSV file will be saved back to your Google Drive.

## Features Used

### Detailed Feature Description

- **Year**: Extracted from the `ID` field to indicate the year when the fire occurred.
- **Month**: Extracted from the `ID` field to capture seasonal effects.
- **Latitude**: Geographic coordinate indicating the location's north-south position.
- **Longitude**: Geographic coordinate indicating the location's east-west position.
- **Other Environmental Data**: This may include other available features such as vegetation index, temperature, or humidity, depending on the dataset provided.

These features are standardized to ensure that the model training process is not biased by differences in feature scales. The model's performance is then evaluated, and only the most important features are selected for the final model.

## Environment Setup

### Google Colab Environment

Google Colab comes pre-installed with most of the necessary libraries, but here’s how you can install any missing packages:

- **Required Libraries**:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `scipy`
  - `jupyter` (if running locally)

- **Installation Command**:
  If you find that a library is missing, you can install it directly in a Colab cell:
  ```python
  !pip install numpy pandas scikit-learn xgboost scipy

