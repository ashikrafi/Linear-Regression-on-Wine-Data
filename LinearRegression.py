import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from IPython.core.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
'exec(%matplotlib inline)'

# Data Collection
dataset = pd.read_csv('Data/winequality-red.csv')
# Statistical Analysis
# display(dataset.describe())

# Data Cleaning
dataset.isnull().any()
values = {'fixed acidity': 0}
dataset = dataset.fillna(value=values)
# display(dataset)

# # dataset = dataset.fillna(method='ffill')

Datacolumns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
#
X = dataset[Datacolumns].values
y = dataset['quality'].values
#
plt.figure(figsize=(10, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['fixed acidity'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

X = dataset.values
coeff_df = pd.DataFrame(regressor.coef_, Datacolumns, columns=['Coefficient'])
display(coeff_df)

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
dataDisplay = df.head(20)
display(dataDisplay)
#
dataDisplay.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
#
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
