import pandas as pd

#question11
filepath="C:/Users/Microsoft/OneDrive/Desktop/pythonAssignments/uscrime.csv"
df = pd.read_csv(filepath)
print(df.head(10))

#question12
# Group by a categorical column, e.g., 'State', and calculate mean and sum of numerical columns
grouped = df.groupby('M.F').agg(['mean', 'sum'])
print(grouped)

#question13
df_filled = df.fillna(df.mean(numeric_only=True))
print(df_filled.head(10))

#question14
df_states = pd.DataFrame({
    'State': ['Alabama', 'Alaska', 'California', 'Texas', 'Nevada'],
    'Region': ['South', 'West', 'West', 'South', 'West']
})

inner = pd.merge(df, df_states, on='State', how='inner')
print("=== Inner Join ===")
print(inner)

# Outer Join: All states from both DataFrames
outer = pd.merge(df, df_states, on='State', how='outer')
print("\n=== Outer Join ===")
print(outer)

# Left Join: All rows from df (uscrime.csv)
left = pd.merge(df, df_states, on='State', how='left')
print("\n=== Left Join ===")
print(left)

# Right Join: All rows from df_states
right = pd.merge(df, df_states, on='State', how='right')
print("\n=== Right Join ===")
print(right)

#question15
# Check data type before
print("Before conversion:", df['Crime'].dtype)

df['Crime'] = pd.to_numeric(df['Crime'], errors='coerce')

# Check data type after
print("After conversion:", df['Crime'].dtype)

# Optional: Check how many NaNs were introduced
print("Number of NaNs:", df['Crime'].isna().sum())

#question16
filtered_df = df[df['Crime'].between(700, 1500)]
print(filtered_df)

#question17
# Convert necessary columns to numeric in case of object types
df['Crime'] = pd.to_numeric(df['Crime'], errors='coerce')
df['Wealth'] = pd.to_numeric(df['Wealth'], errors='coerce')

# Create a pivot table (Group by 'State') with multiple aggregations
pivot_table = pd.pivot_table(
    df,
    index='Pop',
    values=['Crime', 'Wealth'],
    aggfunc={'Crime': ['mean', 'max'], 'Wealth': ['mean', 'min']}
)

print(pivot_table)

#question18
# Define a custom function
def classify_Crime_rate(value):
    if pd.isna(value):
        return 'Unknown'
    elif value > 1000:
        return 'High'
    elif value > 500:
        return 'Medium'
    else:
        return 'Low'

# Apply the function to the 'Crime' column
df['Crime_Level'] = df['Crime'].apply(classify_Crime_rate)

# Display the updated DataFrame
print(df[['Crime', 'Crime_Level']])

#question19
# Define bin edges and labels
bins = [0, 500, 1000, float('inf')]
labels = ['Low', 'Medium', 'High']

# Create a new column for binned categories
df['newCrime_Category'] = pd.cut(df['Crime'], bins=bins, labels=labels, right=True)

# View the result
print(df[['Crime', 'newCrime_Category']])

#question20
df['newCrime_Category'] = df['newCrime_Category'].replace({
    'Low': 'Safe',
    'High': 'Dangerous',
    'Medium': 'Moderate'
})