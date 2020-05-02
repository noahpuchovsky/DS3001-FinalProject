from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from DataLoad import data
import numpy as np
import matplotlib.pyplot as plt

# Spliiting data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("G3", axis=1), data["G3"], test_size=0.30, random_state=0)# fitting the model

model = RandomForestRegressor(random_state=1)
model.fit(X_train, y_train)

# plotting feature importances
features = data.drop("G3", axis=1).columns
importance = model.feature_importances_

indices = np.argsort(importance)
plt.figure(figsize=(10,15))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()