import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#import dataset
new_df=pd.read_csv("E:\\ML\\DATASET\\hr-analytics\\human-resources-analytics\\HR_comma_sep.csv")

#Quick verification
new_df.shape
new_df.columns

#we will use work accident and salary to predict the will the employ left or not
#for that we have to make them categorical variable

#check Datatypes of the columns
new_df.dtypes

#convert work_accident to categorical type
new_df["Work_accident"] = new_df["Work_accident"].astype("category") 

#convert salary to category then first change it to numerical value
# for that purpose we will use Label Encoder



from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
new_df["salary2"]=le.fit_transform(new_df[["salary"]])

#just a quick verification
new_df["salary"].unique()
new_df["salary2"].unique()

# make salary2 as category

new_df["salary2"]=new_df["salary2"].astype("category")
new_df["salary2"].dtype

#decision tree classifier
dt=DecisionTreeClassifier()
dt.fit(new_df[["salary2" , "Work_accident"]],new_df["left"])

#predict the output
dt.predict(new_df[["salary2" , "Work_accident"]])

#check how better the data is fitted to the model
dt.score(new_df[["salary2" , "Work_accident"]],new_df["left"])
