import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataLoad import data

# correlation heatmap
data.corr()
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidth=0.5,fmt='.3f',ax=ax)
plt.show()

# gender bar graph
sns.catplot(x="sex", kind="count",palette="magma", data=data, height = 6)
plt.title("Gender of students : F - female,M - male")

# age histogram
data.age.unique()
plt.figure(figsize=(10,5))
plt.hist(data.age,bins=7,color="purple",width=0.8,density=True)
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.show()

