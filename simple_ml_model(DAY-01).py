import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {'Hours': [1, 2, 3, 4, 5], 'Marks': [40, 50, 60, 70, 80]}
df = pd.DataFrame(data)

# X = input, Y = output
X = df[['Hours']]
y = df['Marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
predicted = model.predict([[6]])
print(f"Predicted Marks for 6 hrs study: {predicted[0]:.2f}")

# Visualize
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Study vs Marks')
plt.show()
