# Data Science & Web Scraping Learning Lab 🔬 📊

Learn data science fundamentals through hands-on practice! This educational project combines machine learning with the Iris dataset (using SVM & Naive Bayes) and web scraping techniques. Perfect for students learning classification models and real-world data collection with Python.

## Required Libraries
```bash
pip install scikit-learn pandas numpy requests beautifulsoup4 openpyxl matplotlib
```

## Project Structure
The project consists of three main components:
1. Classification Models with Visualizations
2. Web Scraping Implementation
3. Data Export & File Handling

## 1. Classification Models

### 1.1 Support Vector Machine (SVM) with Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the Iris dataset and select two features for simplicity
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use the first two features
y = iris.target

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM classifier
model = SVC(kernel='linear', C=1)
model.fit(X_train, y_train)

# Create a mesh grid for plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the mesh grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('SVM Decision Boundaries')
plt.show()
```

### 1.2 Naive Bayes with Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load the Iris dataset and select two features for simplicity
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use the first two features (sepal length and width)
y = iris.target

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Create a mesh grid for plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict the mesh grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Naive Bayes Decision Boundaries')
plt.show()
```

## 2. Web Scraping Implementation

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the Formula 1 website
url = 'https://www.formula1.com/'

# Send a GET request to the website
response = requests.get(url)

# Parse the website content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Extract all headlines (e.g., h2 or h3 tags)
headlines = soup.find_all(['h2', 'h3'])
headline_texts = [headline.text.strip() for headline in headlines[:10]]

# Save the scraped headlines to a pandas DataFrame
df = pd.DataFrame({'Headlines': headline_texts})

# Save the DataFrame to an Excel file
df.to_excel('f1_headlines.xlsx', index=False)
```

## Key Features
- Data visualization of classification boundaries
- Model performance evaluation
- Real-world web scraping
- Data export functionality

## Expected Outputs
1. Visual representations of SVM and Naive Bayes decision boundaries
2. Classification performance metrics
3. Excel file with scraped F1 headlines

## Learning Objectives
- Understanding classification algorithms through visualization
- Comparing different model decision boundaries
- Implementing web scraping
- Data handling and export

## Notes for Students
1. Experiment with different kernel functions in SVM
2. Compare the decision boundaries of both models
3. Try scraping different websites (check robots.txt first)
4. Modify visualization parameters

## Troubleshooting
1. Library Installation
```bash
pip install --upgrade scikit-learn matplotlib numpy
```

2. Common Issues
- MatplotLib backend errors in notebooks
- Web scraping connection timeouts
- File permission issues when saving Excel

## Next Steps
1. Try different feature combinations
2. Implement cross-validation
3. Add error handling to web scraping
4. Experiment with different visualization techniques

## License
MIT License - Feel free to use for educational purposes.

---
For questions or contributions, please open an issue or submit a pull request.
