import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df.columns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']

# Select features for regression
X = df[['Petal.Length']]
y = df['Sepal.Length']

# EDA: Basic Statistics
print("Correlation Coefficient:", df['Petal.Length'].corr(df['Sepal.Length']))
print(df[['Petal.Length', 'Sepal.Length']].describe())

# Scatter Plot
sns.scatterplot(x='Petal.Length', y='Sepal.Length', data=df)
plt.title('Scatter Plot: Petal.Length vs Sepal.Length')
plt.show()

# Linear Regression
model = LinearRegression()
model.fit(X, y)

# Regression Line
y_pred = model.predict(X)
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Petal.Length vs Sepal.Length')
plt.xlabel('Petal.Length')
plt.ylabel('Sepal.Length')
plt.legend()
plt.show()

# Model Metrics
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
print(f"R²: {r2:.3f}, RMSE: {rmse:.3f}")
