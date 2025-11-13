import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("Obesity Classification.csv")
print(df)

print(df.isnull().sum())

df=df.drop(["ID"],axis=1)
print(df)

le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
print(df["Gender"])

print(df.head())

print(df.duplicated().sum())

x=df.drop(["Label"],axis=1)
print(x)
y=df["Label"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True,stratify=y)
print("shape of x_train",x_train.shape)
print("shape of x_test",x_test.shape)
print("shape of y_train",y_train.shape)
print("shape of y_test",y_test.shape)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=10000)
model.fit(x_train,y_train)

y_predict=model.predict(x_test)
print(y_predict)

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_predict)
print(accuracy*100)
precision=precision_score(y_test,y_predict,average='weighted')
print(precision*100)
recall=recall_score(y_test,y_predict,average='weighted')
print(recall)
confusionmatrix=confusion_matrix(y_test,y_predict)
print(confusionmatrix)
f1=f1_score(y_test,y_predict,average='weighted')
print(f1)
clas=classification_report(y_test,y_predict)
print(clas)