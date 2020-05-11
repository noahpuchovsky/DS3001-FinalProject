import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataLoad import data, raw_data
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

def correlation_heatmap():
    data.corr()
    f,ax=plt.subplots(figsize=(15,15))
    sns.heatmap(data.corr(),annot=True,linewidth=0.5,fmt='.3f',ax=ax)
    plt.show()

def gender_bar_graph():
    sns.catplot(x="sex", kind="count",palette="magma", data=data, height = 6)
    plt.title("Gender of students : F - female,M - male")
    plt.show()

def age_histogram():
    data.age.unique()
    plt.figure(figsize=(10,5))
    plt.hist(data.age,bins=7,color="purple",width=0.8,density=True)
    plt.xlabel("Age")
    plt.ylabel("Percentage")
    plt.show()

def print_results(y_pred, y_test, model):
    mse = mean_squared_error(y_test, y_pred)
    # df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    # df1 = df.head(25)
    # df1.plot(kind='bar', figsize=(16, 10))
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    # plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # plt.show()
    # print(model, ': \n', 'mean squared error: ', mse)
    return mse

def studytime_boxplot():
    data['st_time'] = np.nan
    df = [data]

    for col in df:
        col.loc[col['studytime'] == 1 , 'st_time'] = '< 2 hours'
        col.loc[col['studytime'] == 2 , 'st_time'] = '2 to 5 hours'
        col.loc[col['studytime'] == 3, 'st_time'] = '5 to 10 hours'
        col.loc[col['studytime'] == 4, 'st_time'] = '> 10 hours'  

    plt.figure(figsize=(18,7))
    plt.title("Box plot for final grades,depending on the study time")
    sns.boxplot(y="st_time", x="G3", data = raw_data , orient="h")
    plt.show

def goout_boxplot():
    plt.figure(figsize=(18,7))
    plt.title("Box plot for final grades,depending on the amount a student goes out")
    sns.boxplot(y="goout", x="G3", data = raw_data , orient="h")
    plt.show
    
def failures_boxplot():
    plt.figure(figsize=(18,7))
    plt.title("Box plot for final grades,depending on number of failures")
    sns.boxplot(y="failures", x="G3", data = raw_data , orient="h")
    plt.show()
    
def show_dalc_consumption():
    pass

