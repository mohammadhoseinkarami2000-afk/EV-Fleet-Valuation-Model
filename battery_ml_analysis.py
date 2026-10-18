import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print("1. Loading EV dataset...")
df = pd.read_csv("ev_cars_subset.csv")

# =============================================================
# PART 1: UNSUPERVISED LEARNING (K-Means Fleet Segmentation)
# =============================================================
print("\n2. Running K-Means Clustering to group EV fleet...")

# Select features for clustering
cluster_features = ['battery_health_percent', 'vehicle_age', 'odometer_km', 'listing_price']
df_cluster = df.dropna(subset=cluster_features).copy()

# Scale variables so large numbers (like odometer) don't overpower small ones (like age)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cluster[cluster_features])

# Train K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cluster['Cluster'] = kmeans.fit_predict(scaled_features)

# Label clusters dynamically based on average battery health
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_features)
centers['Cluster'] = range(3)
centers = centers.sort_values(by='battery_health_percent', ascending=False)

cluster_mapping = {
    centers.iloc[0]['Cluster']: "1 - Premium Health (Resale)",
    centers.iloc[1]['Cluster']: "2 - Mid-Life (Monitoring)",
    centers.iloc[2]['Cluster']: "3 - Second-Life & Recycling Target"
}
df_cluster['Segment'] = df_cluster['Cluster'].map(cluster_mapping)

# Save Cluster Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_cluster,
    x='odometer_km',
    y='battery_health_percent',
    hue='Segment',
    palette='viridis',
    alpha=0.7
)
plt.axhline(80, color='red', linestyle='--', label='80% Second-Life Health Threshold')
plt.title('EV Fleet Segmentation: Identifying Second-Life Opportunities')
plt.xlabel('Odometer (km)')
plt.ylabel('Battery Health (%)')
plt.legend()
plt.tight_layout()
plt.savefig('ev_clusters.png')
plt.close()
print("   -> Saved scatter plot as 'ev_clusters.png'")

# Merge segment labels back into main dataset and save for Power BI
df = df.merge(df_cluster[['vehicle_id', 'Segment']], on='vehicle_id', how='left')
df.to_csv("ev_cars_with_clusters.csv", index=False)
print("   -> Saved enriched dataset as 'ev_cars_with_clusters.csv' (Ready for Power BI!)")

# =============================================================
# PART 2: SUPERVISED LEARNING (Random Forest Feature Importance)
# =============================================================
print("\n3. Training Random Forest to determine drivers of battery degradation...")

target = 'battery_health_percent'
features = [
    'vehicle_age', 'odometer_km', 'annual_mileage_km',
    'dc_fast_charge_time_minutes', 'electric_range_km', 'battery_capacity_kwh'
]

# Drop rows missing target or feature values
df_ml = df.dropna(subset=features + [target])

X = df_ml[features]
y = df_ml[target]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

# Calculate Feature Importance
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# Save Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette='mako')
plt.title('What Drives Battery Degradation? (Random Forest Feature Importance)')
plt.xlabel('Relative Importance')
plt.ylabel('Vehicle Attributes')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()
print("   -> Saved feature importance plot as 'rf_feature_importance.png'")

print("\n--- SUMMARY OF RESULTS ---")
print("\nFeature Importances (Top factors degrading battery health):")
for feature, imp in importances.items():
    print(f" - {feature}: {imp:.2%}")