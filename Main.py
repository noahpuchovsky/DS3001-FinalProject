from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import SGDRegressor, LinearRegression
from DataLoad import data, feature_names
from sklearn.model_selection import train_test_split, StratifiedKFold
from Visualize import print_results
from sklearn.feature_selection import SelectKBest, chi2, RFECV
import matplotlib.pyplot as plt
import numpy as np

# Spliiting data into test and train sets
X = data.drop("G3", axis=1)
# X = data[["Dalc", "Walc"]]
y = data["G3"]

# # Create the RFE object and compute a cross-validated score.
# est = ExtraTreesRegressor(n_estimators=100)
#
# rfecv = RFECV(estimator=est, step=1, cv=5,)
# rfecv = rfecv.fit(X, y)
# X_new = rfecv.fit_transform(X, y)
# print(X_new)
# print("Optimal number of features : %d" % rfecv.n_features_)

def optimize_features(k, X, y):
    print('K: ', k)
    select_k_best = SelectKBest(chi2, k=k)
    X_new = select_k_best.fit_transform(X, y)
    mask = select_k_best.get_support()  # list of booleans
    new_features = []  # The list of your K best features

    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)

    print(new_features)
    return X_new

def run_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)# fitting the model

    # reg1 = SVR(gamma='auto').fit(X_train, y_train)
    # y_pred1 = reg1.predict(X_test)
    # mse1 = print_results(y_pred1, y_test, "SVR")

    reg2 = SGDRegressor().fit(X_train, y_train)
    y_pred2 = reg2.predict(X_test)
    mse2 = print_results(y_pred2, y_test, "SGD")

    # reg3 = LinearRegression().fit(X_train, y_train)
    # y_pred3 = reg3.predict(X_test)
    # mse3 = print_results(y_pred3, y_test, "LR")

    return mse2


mean_square_errors = np.zeros(shape=(len(feature_names)))
run_times = 1000
for _ in range(run_times):
    for k in range(len(feature_names)):
        X_new = optimize_features(k+1, X, y)
        mse = run_models(X_new, y)
        mean_square_errors[k] += mse

mean_square_errors = mean_square_errors/run_times

l_np = np.asarray(mean_square_errors)
best_k = np.argmin(l_np)
best_mse = mean_square_errors[best_k]

print(feature_names)
print(best_k+1)
print(best_mse)

print(mean_square_errors)

#plt.plot("k", "mean square error", data=mean_square_errors)
print(X)
print(y)
X_new = optimize_features(30, X, y)
mse = run_models(X_new, y)
print(mse)


