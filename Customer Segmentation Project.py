import pandas as pd
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the new file to check its content
df_cust = pd.read_csv('customer_segmentation_data.csv')
print("--- HEAD ---")
print(df_cust.head())
print("\n--- INFO ---")
print(df_cust.info())



df = pd.read_csv('customer_segmentation_data.csv')
X = df[['income', 'spending_score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Let's inspect the profile of each cluster to give meaningful business insights
summary = df.groupby('Cluster')[['income', 'spending_score', 'age']].mean()
print(summary)



# 1. Load data
df = pd.read_csv('customer_segmentation_data.csv')

# 2. Extract features & scale them
X = df[['income', 'spending_score']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Map numeric cluster labels to real-world business names based on data averages
cluster_map = {
    0: "Low Income, Low Spenders",
    1: "High Income, Low Spenders (Sajvers)",
    2: "Low Income, High Spenders",
    3: "High Income, High Spenders (VIPs)"
}
df['Segment Name'] = df['Cluster'].map(cluster_map)

# 4. Print Business Insight Summary Metrics
print("SCIKIT-LEARN CUSTOMER SEGMENTATION BREAKDOWN")
summary_table = df.groupby('Segment Name')[['income', 'spending_score', 'age']].agg({
    'income': 'mean',
    'spending_score': 'mean',
    'age': 'count'
}).rename(columns={'age': 'Customer Count'})

# Format table numbers nicely
summary_table['income'] = summary_table['income'].apply(lambda x: f"${x:,.2f}")
summary_table['spending_score'] = summary_table['spending_score'].apply(lambda x: f"{x:.1f}")
print(summary_table)
print("==================================================================\n")

# 5. Generate the Pure Python Dashboard Visual
plt.figure(figsize=(11, 6.5))
sns.set_theme(style="whitegrid")

sns.scatterplot(
    data=df, 
    x='income', 
    y='spending_score', 
    hue='Segment Name', 
    palette='Set1', 
    alpha=0.85,
    s=80,
    edgecolor='w'
)

# Customize the chart layout
plt.title('Customer Segments Generated via Scikit-Learn K-Means', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Annual Income ($)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.legend(title='Business Segments', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()