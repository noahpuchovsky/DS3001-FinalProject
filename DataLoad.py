import pandas as pd
from sklearn.preprocessing import LabelEncoder

# mat_data = pd.read_csv("Data/student-mat.csv")
data = pd.read_csv("Data.csv", index_col=0)

# data = (mat_data, por_data)
# data = pd.concat(data)
# data = data.drop_duplicates(["school","sex","age","address","famsize","Pstatus",
#                                   "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
# data = data.drop(["G1", "G2"], axis=1)
features = data.drop(["G3"], axis=1)
feature_names = features.columns


# le = LabelEncoder()
# #sex
# le.fit(data.sex.drop_duplicates())
# data.sex = le.transform(data.sex)
# #address
# le.fit(data.address.drop_duplicates())
# data.address = le.transform(data.address)
# #famsize
# le.fit(data.famsize.drop_duplicates())
# data.famsize = le.transform(data.famsize)
# #Pstatus
# le.fit(data.Pstatus.drop_duplicates())
# data.Pstatus = le.transform(data.Pstatus)
# #schoolsup
# le.fit(data.schoolsup.drop_duplicates())
# data.schoolsup = le.transform(data.schoolsup)
# #famsup
# le.fit(data.famsup.drop_duplicates())
# data.famsup = le.transform(data.famsup)
# #paid
# le.fit(data.paid.drop_duplicates())
# data.paid = le.transform(data.paid)
# #activities
# le.fit(data.activities.drop_duplicates())
# data.activities = le.transform(data.activities)
# #nursery
# le.fit(data.nursery.drop_duplicates())
# data.nursery = le.transform(data.nursery)
# #higher
# le.fit(data.higher.drop_duplicates())
# data.higher = le.transform(data.higher)
# #romantic
# le.fit(data.romantic.drop_duplicates())
# data.romantic = le.transform(data.romantic)
# #internet
# le.fit(data.internet.drop_duplicates())
# data.internet = le.transform(data.internet)
#
# #not binary features
# data = pd.get_dummies(data)
#
# data.to_csv('Data.csv')


