import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt

# Read the data set
dataSet = pd.read_csv("C:\\Users\\97059\\PycharmProjects\\AI-P3\\Height_Weight.csv")
print("\nThe Dataset")
print(dataSet)


# Print the data set information
print('\nData Set Information: ')
print("\n", dataSet.info())

# Check if there is a duplicated rows
duplicatedRows = dataSet[dataSet.duplicated()]
numOfDuplicatedRows = duplicatedRows.count()
print("\nNumber of duplicated rows: ", numOfDuplicatedRows)

# Generate the box plot for the Height
dataSet.boxplot(column=['Height'])
plt.title("Box Plot for Height")
plt.ylabel("Height")
plt.show()

# Generate the box plot for the Weight
dataSet.boxplot(column=['Weight'])
plt.title("Box Plot for Weight")
plt.ylabel("Weight")
plt.show()

# Clean the data set by replace the zero values and the outliers with the median
columnsToModify = ['Height', 'Weight']

for column in columnsToModify:
    median = dataSet[column].median()
    dataSet[column] = dataSet[column].replace(0, median)

zScores = np.abs((dataSet[columnsToModify] - dataSet[columnsToModify].median()) / dataSet[columnsToModify].std())
outliers = (zScores > 3)

for column in columnsToModify:
    median = dataSet[column].median()
    dataSet.loc[outliers[column], column] = median

print("\n The Clean Dataset")
print(dataSet)

# Part 1: Convert the height from inches to cms and the weight from pounds to kilograms.
print("\nPart 1:")
# Height in cm = Height in inc * 2.54
dataSet['Height'] = dataSet['Height'] * 2.54
# Weight in kg = Weight in pounds * 0.45359
dataSet['Weight'] = dataSet['Weight'] * 0.45359

print("Updated Dataset: ")
print(dataSet)

# Part 2: Print the main statistics of the features
print("\nPart 2: ")
dataSet1 = pd.read_csv("C:\\Users\\97059\\PycharmProjects\\AI-P3\\Height_Weight.csv")
dataSet1['Height'] = dataSet1['Height'] * 2.54
dataSet1['Weight'] = dataSet1['Weight'] * 0.45359

features = ['Height', 'Weight']
heightWeightColumns = dataSet1[features]
statistics = heightWeightColumns.describe()
print("Main statistics of the features before cleaning the data: ")
print(statistics)

features = ['Height', 'Weight']
heightWeightColumns = dataSet[features]
statistics = heightWeightColumns.describe()
print("\nMain statistics of the features after cleaning the data: ")
print(statistics)


# Part 4: Select a subset of 100 instances from randomly selected from the dataset
print("\nPart 4: ")
print('Model 1:')
randomSubsetM1 = dataSet.sample(n=100, random_state=42) # sample a method in pandas library
# print("Random Subset of 100 Instances:")
# print(randomSubsetM1)
# Generate the first model (called M1) and test this models performance using appropriate regression metrics.
XM1 = randomSubsetM1[['Height']]
YM1 = randomSubsetM1['Weight']

X_train_M1, X_test_M1, y_train_M1, y_test_M1 = train_test_split(XM1, YM1, test_size=0.3, random_state=42)
print("Split the data set: ")
print("Training set For M1: ", X_train_M1.shape[0])
print("Test set For M1: ", X_test_M1.shape[0])
M1 = LinearRegression()
M1.fit(X_train_M1, y_train_M1)
predictions = M1.predict(X_test_M1)
M1MSE = mean_squared_error(y_test_M1, predictions)
M1RMSE = np.sqrt(M1MSE)
M1MAE = mean_absolute_error(y_test_M1, predictions)
M1R2 = r2_score(y_test_M1, predictions)
print('Performance matrices:')
print('MSE For M1: ', M1MSE)
print('RMSE For M1: ', M1RMSE)
print('MAE For M1: ', M1MAE)
print('R^2 for M1: ', M1R2)
plt.scatter(y_test_M1, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions For M1')
plt.show()

# Part 5: Select a subset of 1000 instances from randomly selected from the dataset
print("\nPart 5: ")
print('Model 2:')
randomSubsetM2 = dataSet.sample(n=1000, random_state=42) # sample a method in pandas library
# print("Random Subset of 1000 Instances:")
# print(randomSubsetM2)
# Generate the second model (called M2) and test this models performance using appropriate regression metrics.
XM2 = randomSubsetM2[['Height']]
YM2 = randomSubsetM2['Weight']

X_train_M2, X_test_M2, y_train_M2, y_test_M2 = train_test_split(XM2, YM2, test_size=0.3, random_state=42)
print("Split the data set: ")
print("Training set For M2: ", X_train_M2.shape[0])
print("Test set For M2: ", X_test_M2.shape[0])
M2 = LinearRegression()
M2.fit(X_train_M2, y_train_M2)
predictions = M2.predict(X_test_M2)
M2MSE = mean_squared_error(y_test_M2, predictions)
M2RMSE = np.sqrt(M2MSE)
M2MAE = mean_absolute_error(y_test_M2, predictions)
M2R2 = r2_score(y_test_M2, predictions)
print('Performance matrices:')
print('MSE For M2: ', M2MSE)
print('RMSE For M2: ', M2RMSE)
print('MAE For M2: ', M2MAE)
print('R^2 for M2: ', M2R2)
plt.scatter(y_test_M2, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions For M2')
plt.show()

# Part 6: Select a subset of 5000 instances from randomly selected from the dataset
print("\nPart 6: ")
print('Model 3:')
randomSubsetM3 = dataSet.sample(n=5000, random_state=42) # sample a method in pandas library
# print("Random Subset of 5000 Instances:")
# print(randomSubsetM3)
# Generate the first model (called M3) and test this models performance using appropriate regression metrics.
XM3 = randomSubsetM3[['Height']]
YM3 = randomSubsetM3['Weight']

X_train_M3, X_test_M3, y_train_M3, y_test_M3 = train_test_split(XM3, YM3, test_size=0.3, random_state=42)
print("Split the data set: ")
print("Training set For M3: ", X_train_M3.shape[0])
print("Test set For M3: ", X_test_M3.shape[0])
M3 = LinearRegression()
M3.fit(X_train_M3, y_train_M3)
predictions = M3.predict(X_test_M3)
M3MSE = mean_squared_error(y_test_M3, predictions)
M3RMSE = np.sqrt(M3MSE)
M3MAE = mean_absolute_error(y_test_M3, predictions)
M3R2 = r2_score(y_test_M3, predictions)
print('Performance matrices:')
print('MSE For M3: ', M3MSE)
print('RMSE For M3: ', M3RMSE)
print('MAE For M3: ', M3MAE)
print('R^2 for M3: ', M3R2)
plt.scatter(y_test_M3, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions For M3')
plt.show()


# Part 7: Use the entire dataset and generate the first model (called M4)
print('\nPart 7:')
print('Model 4:')
X = dataSet[['Height']]
y = dataSet['Weight']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Split the data set: ")
print("Training set :", X_train.shape[0])
print("Test set :", X_test.shape[0])
M4 = LinearRegression()
M4.fit(X_train, Y_train)
predictions = M4.predict(X_test)
M4MSE = mean_squared_error(Y_test, predictions)
M4RMSE = np.sqrt(M4MSE)
M4MAE = mean_absolute_error(Y_test, predictions)
M4R2 = r2_score(Y_test, predictions)
print('Performance matrices:')
print('MSE For M4: ', M4MSE)
print('RMSE For M4: ', M4RMSE)
print('MAE For M4: ', M4MAE)
print('R^2 for M4: ', M4R2)
plt.scatter(Y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions For M4')
plt.show()

# # Part 7: Use the entire dataset and generate the first model (called M4)
# print('\nPart 7:')
# print('Model 4:')
# X = dataSet1[['Height']]
# y = dataSet1['Weight']
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print("Split the data set: ")
# print("Training set :", X_train.shape[0])
# print("Test set :", X_test.shape[0])
# M4 = LinearRegression()
# M4.fit(X_train, Y_train)
# predictions = M4.predict(X_test)
# M4MSE = mean_squared_error(Y_test, predictions)
# M4RMSE = np.sqrt(M4MSE)
# M4MAE = mean_absolute_error(Y_test, predictions)
# M4R2 = r2_score(Y_test, predictions)
# # print('Performance matrices:')
# # print('MSE For M4:  {M4MSE:.15f}')
# # print('RMSE For M4: {M4RMSE:.15f}')
# # print('MAE For M4: {M4MAE:.15f}')
# # print('R^2 for M4: {M4R2:.15f}')
# print('Performance metrics for Model 4:')
# print(f'Mean Squared Error (MSE) For M4: {M4MSE:.15f}')
# print(f'Root Mean Squared Error (RMSE) For M4: {M4RMSE:.15f}')
# print(f'Mean Absolute Error (MAE) For M4: {M4MAE:.15f}')
# print(f'R-squared (R^2) For M4: {M4R2:.15f}')
# plt.scatter(Y_test, predictions)
# plt.xlabel('True Values')
# plt.ylabel('Predictions')
# plt.title('True Values vs Predictions For M4')
# plt.show()
