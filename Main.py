from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from DataLoad import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectKBest, chi2

# Spliiting data into test and train sets
X = data.drop("G3", axis=1)
y = data["G3"]

X_new = SelectKBest(chi2, k=20).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.30, random_state=0)# fitting the model

reg = LinearRegression().fit(X_train, y_train)
y_p = reg.predict(X_test)
a = explained_variance_score(y_test, y_p)
print(a)

