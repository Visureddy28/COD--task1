
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# Preprocess the data
df = df.dropna()  # Dropping rows with missing values
df = pd.get_dummies(df, columns=['ocean_proximity'])  # Encoding categorical variable

# Define features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Linear Regression model
model = LinearRegression()

# Training the model on the training set
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Residuals (errors)
residuals = y_test - y_pred

# Boxplot of Residuals
plt.figure(figsize=(10, 5))
sns.boxplot(residuals)
plt.xlabel("Residuals")
plt.title("Boxplot of Residuals")
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, edgecolor='k')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# Histogram of Predicted Prices
plt.figure(figsize=(10, 5))
plt.hist(y_pred, bins=30, edgecolor='k')
plt.xlabel("Predicted Prices")
plt.ylabel("Frequency")
plt.title("Histogram of Predicted Prices")
plt.show()