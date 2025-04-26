# Project3-Team10
This repository contains a Python Shiny web application to analyze a dataset of the user's choosing. 
Site: https://project3-team10-ee3r.onrender.com/ 

To test user engagment between the two sites, we implemented Google Analytics to collect user engagement data. The link assigns a version randomly for the user. 

## Project Installation
1. Clone the Repository
```{python} 
git clone https://github.com/yinhannn/Project3-Team10.git
```
2. Navigate to the project directory:
```{python} 
cd Project3-Team10
```
3. Install required libraries:
```{python}
pip install -r requirements.txt
```


## Choose dataset
The user first sees a screen that has a panel on the left side and can choose if they want to use the default database or upload a database on their own to analyze. The default database is from [Kaggle](https://www.kaggle.com/datasets/samikshadalvi/lungs-diseases-dataset) and contains detailed information about patients suffering from various lung conditions and if they have recovered from their lung disease. The uploaded datasets can be in various formats (e.g., CSV, Excel, JSON, and RDS). The user can then select the ***Reset Data*** button which will save the dataset to further analyze it. The user can reset the dataset they want to use by clicking the ***Reset Data*** button again in the panel.

## Data Cleaning and Preprocessing
The Cleaning & Preprocessing section provides tools to clean, transform, and prepare datasets for further analysis. Users can handle missing values, remove duplicates, detect and treat outliers, normalize numerical features, and encode categorical variables.  
#### Data Cleaning  
This section allows users to perform basic cleaning operations to ensure data consistency and accuracy:  
- Clean Strings & Convert Numbers automatically trims extra spaces, converts text-based numbers into numerical format, and standardizes string formatting.  
- Convert to Dates detects and converts date-like strings into proper date-time format.  
- Remove Duplicate Rows eliminates exact duplicate rows, ensuring data integrity.  
#### Missing Value Handling 
Users can choose from the following strategies to handle missing values  
- **Mean, Median, or Mode Imputation** replaces missing values based on statistical measures.  
- **Drop Missing Values** allows users to remove rows or specific columns with missing values.  
- **Bulk Selection for NA Removal** provides options to select or deselect all columns for missing value treatment.  
#### Outlier Handling  
This section provides tools to detect and handle outliers  
- **Detection Methods**: Users can choose between the Interquartile Range (IQR) method or Z-score to identify outliers.  
- **Threshold Adjustment**: A Z-score threshold slider helps fine-tune outlier detection sensitivity.  
- **Handling Options**: Users can choose to delete outliers, replace them with mean/median values, or apply Winsorization to cap extreme values.  
#### Normalization    
- **Enable Normalization** applies standard scaling techniques.  
- **Column Selection**: Users can select specific numeric columns or apply normalization across all numeric features.  
#### Encoding   
- This part allows users to encode categorical columns. A thershold is set to let user choose when should a column should apply lable encoder.  If a columnn's unique value bigger than the threshold, lable encoder will be used. 


## **Feature Engineering**
The Feature Engineering section allows users to create new features and modify existing features, providing visual feedback to display the impact of such transformations. Click the Update View button to see the effect on transformations on the preview data table.
#### Target Feature Transformation 
Select a column and transformation method (Log Transformation, Box-Cox, Yeo-Johnson) to see the impact of the transformation on the column. 
Note that the column must have missing values filled in from the data preprocessing step in order for the Box-Cox and Yeo-Johnson to yield results. 
Typically, the Box-Cox method requires that the column values are positive, but this has been considered such that non-positive values are accommodated for. 
#### Feature Selection
This method allows for dimensionality reduction. There are several feature selection methods:
- **PCA** allows the user to select number of PCA components and yields the variance explained by each component. Please ensure the data is properly preprocessed to yield results (e.g. handle missing values, scaling, hot-one encoding, etc.) 
- **Filter Zero-Var Variables** allows users to select variance threshold (from 0 to max(var)) for filtering and returns the features dropped at said threshold. Please ensure data is properly processed to yield results (e.g. fill missing values, etc.)
- **Manually Remove** allows users to select specific column(s) to manually remove from the table. 
#### Create New Features
This method allows users to create new features based on pre-existing features in the data. Input name of new feature, new feature formula, and the pre-existing features which are used in the new formula.
In the "Input New Formula," please ensure the columns are spelled correctly and the expression is a valid math expression. Also features should not have spaces in their names. 
                

## Exploratory Data Analysis 
**Exploratory Data Analysis** 
The EDA section allows users to explore data through interactive visualizations, summary statistics, and correlation analysis.  
Filters can be applied to focus on specific subsets of the dataset, and all outputs update dynamically based on user selections.                
#### Apply Filters  
Filters help refine the dataset for analysis. For numerical columns, sliders allow selection of value ranges, while categorical columns can be filtered using dropdown menus.  
When a filter is adjusted, all visualizations and statistical summaries update automatically.  
#### Visualization  
Several types of visualizations are available to help interpret the data:  
- **Histograms** display the distribution of a single numerical variable.  
- **Scatter plots** show relationships between two numerical variables.  
- **Box plots** highlight the spread of data and detect potential outliers.  
- **Correlation heatmaps** provide an overview of relationships between numerical features.  
#### Statistical Insights  
Summary statistics, including mean, median, minimum, maximum, and standard deviation, offer a quick overview of the dataset.  
A correlation table helps identify potential relationships between numerical variables, which can be useful for deeper analysis. 


