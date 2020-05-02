import pandas as pd
from sklearn import preprocessing

mat_data = pd.read_csv("Data/student-mat.csv")
por_data = pd.read_csv("Data/student-por.csv")
data = (mat_data, por_data)
data = pd.concat(data)
data = data.drop_duplicates(["school","sex","age","address","famsize","Pstatus",
                                  "Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
data.describe()
data = data.drop(["G1", "G2"], axis=1)
# "school","sex","address","famsize","Pstatus", "Mjob","Fjob","reason","guardian","schoolsup"
# "famsup","paid","activities","nursery","higher","internet","romantic"

char_cols = data.dtypes.pipe(lambda x: x[x == 'object']).index
label_mapping = {}

for c in char_cols:
    data[c], label_mapping[c] = pd.factorize(data[c])
