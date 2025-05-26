from os_sklearn.ensemble import OSRandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from imblearn.datasets import fetch_datasets
us_crime = fetch_datasets()['us_crime']

X = us_crime.data
y = us_crime.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

rf = OSRandomForestClassifier(oversampling_strategy='BorderlineSMOTE',
                              print_indices_list=[1],
                              n_estimators=50,
                              data_name='us_crime',
                              iteration=1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("=== RF with SMOTE ===")
print(classification_report(y_test, y_pred))