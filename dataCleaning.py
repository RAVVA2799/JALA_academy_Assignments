import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor

filepath= "C:/Users/Microsoft/OneDrive/Desktop/pythonAssignments/IPL_Team_Performance.csv"
df = pd.read_csv(filepath)


#question21
numeric_cols = ['Team','State']

# Convert each column to numeric, forcing errors to NaN
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df_cleaned = df.dropna(subset=numeric_cols)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


#question22
# Check for duplicate rows
duplicates = df[df.duplicated()]
print("Duplicate rows:\n", duplicates)

#question23
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
df_scaled = scaler.fit_transform(df[numeric_cols])

# Convert back to DataFrame for readability
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

print("Standard Scaled Data:\n", df_scaled.head())


#question24
le = LabelEncoder()
df['State_encoded'] = le.fit_transform(df['State'])

# Step 3: View the updated DataFrame
print(df[['Team', 'State', 'State_encoded']])

#question25
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Step 3: View the results
print("Training Set:\n", train_set)
print("\nTesting Set:\n", test_set)

X = df[['Total Matches Played', 'Total Wins']]  # Features
y = df['Win Percentage']                        # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#question26
numeric_cols = df.select_dtypes(include='number')

# Step 3: Calculate summary statistics
summary = {
    'Mean': numeric_cols.mean(),
    'Median': numeric_cols.median(),
    'Mode': numeric_cols.mode().iloc[0],  # mode() returns a DataFrame
    'Standard Deviation': numeric_cols.std()
}

# Combine results into a single DataFrame
summary_df = pd.DataFrame(summary)
print("📊 Summary Statistics (EDA):\n")
print(summary_df)
    
#question27
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Win Percentage'], kde=True, color='skyblue', bins=10)
plt.title('Histogram of Win Percentage')
plt.xlabel('Win Percentage')
plt.ylabel('Frequency')

# Step 3: Plot Box Plot
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Win Percentage'], color='lightgreen')
plt.title('Box Plot of Win Percentage')
plt.xlabel('Win Percentage')

plt.tight_layout()
plt.show()

#question28
# Step 2: Compute the correlation matrix (only numeric columns)
corr_matrix = df[['Total Matches Played', 'Total Wins', 'Win Percentage']].corr()

# Step 3: Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('📊 Correlation Matrix Heatmap')
plt.show()

#question29
# Plot scatter matrix
scatter_matrix(df[['Total Matches Played', 'Total Wins', 'Win Percentage']], figsize=(10, 8), diagonal='kde', color='skyblue')
plt.suptitle("Scatter Plot Matrix - IPL Teams", fontsize=16)
plt.show()

# Optional: add team names as hue for color grouping
sns.pairplot(df[['Total Matches Played', 'Total Wins', 'Win Percentage']])
plt.suptitle("Seaborn Pairplot - IPL Teams", y=1.02, fontsize=16)
plt.show()


#question30
# Step 2: Feature Engineering
df['Losses'] = df['Total Matches Played'] - df['Total Wins']
df['Win Ratio'] = df['Total Wins'] / df['Total Matches Played']
df['Efficiency Score'] = df['Win Percentage'] * df['Win Ratio']

# Features and target
X = df[['Total Matches Played', 'Total Wins', 'Losses', 'Win Ratio', 'Efficiency Score']]
y = df['Win Percentage']

# Train model
model = RandomForestRegressor(random_state=0)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='teal')
plt.xlabel('Feature Importance')
plt.title('🔍 Feature Importance for Predicting Win Percentage')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()