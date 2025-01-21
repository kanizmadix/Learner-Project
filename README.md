# Data Science and Web Scraping Learning Project

## Overview
This educational project combines fundamental data science concepts with practical web scraping techniques. Students will learn to work with classic machine learning datasets, implement different classification algorithms, and gather real-world data through web scraping.

## Learning Objectives
- Understanding basic data manipulation using pandas
- Implementing and comparing different machine learning algorithms
- Learning web scraping fundamentals with BeautifulSoup
- Working with real-world data from web sources
- Data export and file handling

## Prerequisites
- Python 3.x
- Google Colab account (recommended) or Jupyter Notebook
- Basic understanding of Python programming

## Required Libraries
```bash
pip install scikit-learn pandas numpy requests beautifulsoup4 openpyxl
```

## Project Structure
The project is divided into three main sections:

### 1. Data Science Fundamentals
- Loading and exploring the Iris dataset
- Implementation of classification algorithms:
  - Support Vector Machine (SVM)
  - Naive Bayes
- Model evaluation and performance metrics

### 2. Web Scraping
- Basic web scraping from Formula 1 website
- Data extraction and processing
- Saving data to Excel format

### 3. File Management
- Working with Excel files
- Downloading files in Google Colab

## Step-by-Step Guide

### Part 1: Data Science

1. **Loading the Iris Dataset**
   ```python
   from sklearn.datasets import load_iris
   import pandas as pd
   
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   df['species'] = iris.target
   ```

2. **Implementing SVM Classification**
   - Data splitting
   - Model training
   - Performance evaluation
   - Analysis of classification report

3. **Implementing Naive Bayes Classification**
   - Comparison with SVM results
   - Understanding different evaluation metrics

### Part 2: Web Scraping

1. **Basic Web Scraping**
   - Connecting to website
   - Extracting page title and headlines
   - Processing HTML content

2. **Data Processing**
   - Converting scraped data to DataFrame
   - Exporting data to Excel
   - File handling and downloads

## Expected Outputs
- Classification reports for both SVM and Naive Bayes models
- Excel file containing scraped F1 headlines
- Understanding of model performance metrics

## Performance Metrics Covered
- Accuracy
- Precision
- Recall
- F1-score

## Notes for Students
1. Always check website's robots.txt and terms of service before scraping
2. Pay attention to model performance differences between SVM and Naive Bayes
3. Try modifying parameters to understand their impact on model performance
4. Experiment with different datasets and websites

## Common Issues and Solutions
1. **Library Installation Issues**
   - Make sure all required libraries are installed
   - Check for version compatibility

2. **Web Scraping Errors**
   - Website might change structure
   - Connection issues
   - Rate limiting

3. **Model Performance**
   - Data splitting affects results
   - Parameter tuning might be needed

## Further Exploration
1. Try different classification algorithms
2. Experiment with feature engineering
3. Implement data visualization
4. Try scraping different websites
5. Add error handling to web scraping code

## Contributing
Feel free to fork this project and submit improvements via pull requests.

## License
This project is available for educational purposes under the MIT License.
