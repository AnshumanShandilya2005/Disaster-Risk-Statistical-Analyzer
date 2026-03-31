# ==========================================
# DISASTER RISK PREDICTION SYSTEM (FINAL)
# ==========================================

# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ==============================
# LOAD DATA
# ==============================
climate = pd.read_csv('climate.csv')
disaster = pd.read_csv('disaster.csv')
flood = pd.read_csv('flood.csv')

# ==============================
# CHECK COLUMNS (DEBUG)
# ==============================
print("\nClimate Columns:", climate.columns)
print("Disaster Columns:", disaster.columns)
print("Flood Columns:", flood.columns)

# ==============================
# CLEAN CLIMATE DATA (FIXED)
# ==============================
climate = climate[['Country', 'Year', 'Avg Temperature (°C)', 'Rainfall (mm)', 'CO2 Emissions (Tons/Capita)']]
climate.columns = ['location', 'year', 'temperature', 'rainfall', 'co2']

# ==============================
# CLEAN DISASTER DATA (FIXED)
# ==============================
disaster = disaster[['Country', 'Year', 'Disaster Type']]
disaster.columns = ['location', 'year', 'disaster_type']

# Create target column
disaster['disaster_occurred'] = 1

# ==============================
# MERGE DATA
# ==============================
merged = pd.merge(climate, disaster, on=['location','year'], how='left')

# Fill missing (no disaster = 0)
merged['disaster_occurred'] = merged['disaster_occurred'].fillna(0)

print("\nMerged Data Preview:")
print(merged.head())

# ==============================
# HANDLE MISSING VALUES
# ==============================
merged = merged.ffill()

# ==============================
# FEATURE SELECTION
# ==============================
X = merged[['temperature','rainfall','co2']]
y = merged['disaster_occurred']

# ==============================
# EDA (3 GRAPHS)
# ==============================

# 1. Heatmap
plt.figure(figsize=(6,4))
numeric_data = merged.select_dtypes(include=[np.number])

plt.figure(figsize=(6,4))
sns.heatmap(numeric_data.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
plt.title("Correlation Heatmap")
plt.show()

# 2. Bar Graph
merged.groupby('location')['temperature'].mean().head(10).plot(kind='bar')
plt.title("Average Temperature (Top 10 Locations)")
plt.show()

# 3. Scatter Plot
plt.scatter(merged['rainfall'], merged['temperature'], c=y)
plt.xlabel("Rainfall")
plt.ylabel("Temperature")
plt.title("Disaster Pattern")
plt.show()

# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# ML MODELS
# ==============================
models = {
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

results = {}

print("\nMODEL RESULTS:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(name, "Accuracy:", acc)

# ==============================
# BEST MODEL
# ==============================
best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", results[best_model])

# ==============================
# MODEL COMPARISON GRAPH
# ==============================
plt.bar(results.keys(), results.values())
plt.title("Model Comparison")
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.show()

# ==============================
# FEATURE IMPORTANCE
# ==============================
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

importances = rf.feature_importances_

plt.bar(['temperature','rainfall','co2'], importances)
plt.title("Feature Importance")
plt.show()

# ==============================
# BONUS: FLOOD DATASET MODEL
# ==============================
print("\n--- Flood Dataset Model ---")

flood = flood.dropna()

Xf = flood.drop('occured', axis=1)
yf = flood['occured']

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    Xf, yf, test_size=0.2, random_state=42
)

rf.fit(Xf_train, yf_train)
pred = rf.predict(Xf_test)

print("Flood Model Accuracy:", accuracy_score(yf_test, pred))