# Import Required Libraries 
import pandas as pd 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_absolute_error 
import category_encoders as ce 
import joblib

#Read the CSV file.
data = pd.read_csv("House_Rent_dataset.csv") 
#View the first five rows. 
data.head()

#Separate independent and dependent variables
y = data["Rent"] 
X = data.drop( 
columns=[ 
"Rent", 
"Posted On", 
"Floor", 
"Area Locality" 
] 
) 

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split( 
X, 
y, 
test_size=0.25, 
random_state=42 
) 

pipeline = Pipeline([ 
("encoder", ce.LeaveOneOutEncoder()), 
("scaler", MinMaxScaler()), 
("model", LinearRegression()) 
])


#Train the entire pipeline. 
pipeline.fit(X_train, y_train) 

#Predict on the testing data. 
y_pred = pipeline.predict(X_test) 
#Calculate performance metrics. 
print("R² Score :", r2_score(y_test, y_pred)) 
print("MAE :", mean_absolute_error(y_test, y_pred))

joblib.dump( 
pipeline, 
"house_rent_pipeline.pkl" 
) 