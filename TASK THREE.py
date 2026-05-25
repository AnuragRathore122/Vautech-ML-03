# Model Training with Controlled Experiments

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# STEP 1: Load Dataset

df = pd.read_csv("C:\\Users\\Anurag Rathore\\Downloads\\adult.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# STEP 2: Data Preprocessing

# Handle missing values

df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Encode categorical features

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("\nData cleaned successfully!")

# STEP 3: Feature Sets
# Feature Set 1 (All features)

X1 = df.drop("income", axis=1)
y = df["income"]

# Feature Set 2 (Selected important features)

selected_features = ['age', 'educational-num', 'hours-per-week', 'capital-gain']
X2 = df[selected_features]

# STEP 4: Train-Test Split

X_train1, X_test1, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42)

X_train2, X_test2, _, _ = train_test_split(
    X2, y, test_size=0.2, random_state=42)

# STEP 5: Controlled Experiment 1
# Changing Hyperparameters

print("\n----- Experiment 1: Hyperparameter Change -----")

model_1 = RandomForestClassifier(n_estimators=50, random_state=42)
model_1.fit(X_train1, y_train)
pred1 = model_1.predict(X_test1)

acc1 = accuracy_score(y_test, pred1)
print("Accuracy with 50 trees:", acc1)


model_2 = RandomForestClassifier(n_estimators=150, random_state=42)
model_2.fit(X_train1, y_train)
pred2 = model_2.predict(X_test1)

acc2 = accuracy_score(y_test, pred2)
print("Accuracy with 150 trees:", acc2)

# STEP 6: Controlled Experiment 2
# Comparing Feature Sets

print("\n----- Experiment 2: Feature Set Comparison -----")

model_fs1 = RandomForestClassifier(random_state=42)
model_fs1.fit(X_train1, y_train)
pred_fs1 = model_fs1.predict(X_test1)

acc_fs1 = accuracy_score(y_test, pred_fs1)
print("Accuracy with ALL features:", acc_fs1)

model_fs2 = RandomForestClassifier(random_state=42)
model_fs2.fit(X_train2, y_train)
pred_fs2 = model_fs2.predict(X_test2)

acc_fs2 = accuracy_score(y_test, pred_fs2)
print("Accuracy with SELECTED features:", acc_fs2)

# STEP 7: Observations

print("\n----- Observations -----")

if acc2 > acc1:
    print("Increasing trees improved performance.")
else:
    print("More trees did NOT significantly improve performance.")

if acc_fs1 > acc_fs2:
    print("Using all features gives better accuracy.")
else:
    print("Selected features are sufficient and reduce complexity.")