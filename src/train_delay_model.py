import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#step1: Load Dataset
df = pd.read_csv('delivery_data.csv')
print("Data loaded successfully")
print(df.head)

#Step2 : Encode categorical columns
label_encoders = {}
for col in ['Traffic' , 'Weather']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#Step3 : Define features (x) and target (y)
x = df[['Distance_km', 'Traffic', 'Weather']]
y = df['Delay']

#Step 4 : split into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Step 5 : Train ML model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#Step6 : Evaluate model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#Step 7 : Save trained model and encoders
joblib.dump(model, 'delay_prediction_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("\n💾 Model and encoders saved successfully!")