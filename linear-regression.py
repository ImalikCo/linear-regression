import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fill in your csv file here
data = pd.read_csv('.csv')

# If csv file contains strings; change them into binary:
# If no strings are included, remove this line
data = pd.get_dummies(data, columns=['string_columns_here'])

# Fill in your x and y values here. X are the independent values, y is the dependent value
X = data[['']].values
y = data[''].values

#Split into training and testing set, change test_size and random_state to your own wishing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=0 )

model = LinearRegression()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)

#Plot data and linear regression line; change color to your own wishing
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_prediction, color = 'green')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Your Linear Regression Title here')
plt.show
