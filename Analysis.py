import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Set visualization styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})


# =============================================================
# STEP 1: LOAD DATASET & NULL VALUE REPORT
# =============================================================
def load_and_inspect_data(file_name):
    print("=== 1. LOADING DATASET & NULL VALUE ANALYSIS ===")

    # Automatically locate CSV relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    file_path = os.path.join(script_dir, file_name)

    df = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.\n")

    # Calculate missing counts and percentages
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    null_df = pd.DataFrame({'Missing_Count': null_counts, 'Missing_Percentage (%)': null_pct})

    # Filter for columns that actually have missing values
    null_report = null_df[null_df['Missing_Count'] > 0].sort_values(by='Missing_Count', ascending=False)

    print("Columns with Missing Values:")
    print(null_report.head(15))
    print("\n" + "=" * 60 + "\n")
    return df


# =============================================================
# STEP 2: SUMMARY STATISTICS & DESCRIPTIVE ANALYSIS
# =============================================================
def generate_summary_statistics(df, numerical_cols):
    print("=== 2. DESCRIPTIVE STATISTICS ===")
    stats = df[numerical_cols].describe().T
    stats['skewness'] = df[numerical_cols].skew()
    stats['median'] = df[numerical_cols].median()

    # Format and reorder columns
    ordered_stats = stats[['count', 'mean', 'median', 'std', 'min', '25%', '75%', 'max', 'skewness']]
    print(ordered_stats.round(2))
    print("\n" + "=" * 60 + "\n")
    return ordered_stats


# =============================================================
# STEP 3: EDA - DISTRIBUTIONS & CORRELATIONS
# =============================================================
def generate_eda_plots(df, numerical_cols):
    print("=== 3. GENERATING EDA VISUALIZATIONS ===")

    # Plot Distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    sns.histplot(df['battery_health_percent'].dropna(), kde=True, ax=axes[0, 0], color='#4C72B0')
    axes[0, 0].set_title('Battery Health Distribution (%)')
    axes[0, 0].set_xlabel('Battery Health (%)')

    sns.histplot(df['odometer_km'].dropna(), kde=True, ax=axes[0, 1], color='#DD8452')
    axes[0, 1].set_title('Odometer Distribution (km)')
    axes[0, 1].set_xlabel('Odometer (km)')

    sns.histplot(df['listing_price'].dropna(), kde=True, ax=axes[0, 2], color='#55A868')
    axes[0, 2].set_title('Listing Price Distribution ($)')
    axes[0, 2].set_xlabel('Listing Price ($)')

    sns.histplot(df['electric_range_km'].dropna(), kde=True, ax=axes[1, 0], color='#C44E52')
    axes[1, 0].set_title('Electric Range Distribution (km)')
    axes[1, 0].set_xlabel('Electric Range (km)')

    sns.histplot(df['vehicle_age'].dropna(), kde=True, ax=axes[1, 1], color='#8172B3')
    axes[1, 1].set_title('Vehicle Age Distribution (Years)')
    axes[1, 1].set_xlabel('Vehicle Age (Years)')

    sns.histplot(df['dc_fast_charge_time_minutes'].dropna(), kde=True, ax=axes[1, 2], color='#9370DB')
    axes[1, 2].set_title('DC Fast Charge Time (Minutes)')
    axes[1, 2].set_xlabel('Charge Time 10-80% (mins)')

    plt.tight_layout()
    plt.savefig('ev_data_distributions.png', dpi=300)
    plt.close()
    print("   -> Saved 'ev_data_distributions.png'")

    # Plot Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Core EV Variables', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('ev_correlation_matrix.png', dpi=300)
    plt.close()
    print("   -> Saved 'ev_correlation_matrix.png'\n")
    print("=" * 60 + "\n")


# =============================================================
# STEP 4: UNSUPERVISED LEARNING (K-Means Clustering)
# =============================================================
def run_kmeans_clustering(df, cluster_features):
    print("=== 4. RUNNING K-MEANS FLEET SEGMENTATION ===")

    # Drop rows missing clustering features
    df_cluster = df.dropna(subset=cluster_features).copy()

    # Scale numerical values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster[cluster_features])

    # Fit K-Means
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

    # Scatter Plot
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
    plt.legend(title='Vehicle Segment')
    plt.tight_layout()
    plt.savefig('ev_clusters.png', dpi=300)
    plt.close()
    print("   -> Saved scatter plot as 'ev_clusters.png'")

    # Merge segments back to original dataframe and save
    df_enriched = df.merge(df_cluster[['vehicle_id', 'Segment']], on='vehicle_id', how='left')
    df_enriched.to_csv("ev_cars_with_clusters.csv", index=False)
    print("   -> Saved enriched dataset as 'ev_cars_with_clusters.csv'\n")
    print("=" * 60 + "\n")
    return df_enriched


# =============================================================
# STEP 5: SUPERVISED LEARNING (Random Forest Feature Importance)
# =============================================================
def run_random_forest_importance(df, features, target):
    print("=== 5. TRAINING RANDOM FOREST DEGRADATION MODEL ===")

    df_ml = df.dropna(subset=features + [target])

    X = df_ml[features]
    y = df_ml[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)

    r2_score = rf.score(X_test, y_test)
    print(f"Model Training Complete. Test Set R^2 Score: {r2_score:.4f}\n")

    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='mako')
    plt.title('What Drives Battery Degradation? (Random Forest Feature Importance)')
    plt.xlabel('Relative Importance')
    plt.ylabel('Vehicle Attributes')
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300)
    plt.close()
    print("   -> Saved feature importance plot as 'rf_feature_importance.png'")

    print("\nFeature Importances Summary:")
    for feature, imp in importances.items():
        print(f" - {feature}: {imp:.2%}")
    print("\n" + "=" * 60 + "\n")


# =============================================================
# MAIN EXECUTION PIPELINE
# =============================================================
if __name__ == "__main__":
    file_name = "ev_cars_subset.csv"

    # 1. Load Data
    ev_df = load_and_inspect_data(file_name)

    # 2. Key variables for statistical inspection
    core_variables = [
        'battery_health_percent', 'vehicle_age', 'odometer_km', 'listing_price',
        'annual_mileage_km', 'dc_fast_charge_time_minutes', 'electric_range_km', 'battery_capacity_kwh'
    ]

    # 3. Descriptive Stats & Plotting
    generate_summary_statistics(ev_df, core_variables)
    generate_eda_plots(ev_df, core_variables)

    # 4. K-Means Clustering
    cluster_cols = ['battery_health_percent', 'vehicle_age', 'odometer_km', 'listing_price']
    enriched_df = run_kmeans_clustering(ev_df, cluster_cols)

    # 5. Random Forest Regressor
    ml_features = [
        'vehicle_age', 'odometer_km', 'annual_mileage_km',
        'dc_fast_charge_time_minutes', 'electric_range_km', 'battery_capacity_kwh'
    ]
    run_random_forest_importance(enriched_df, ml_features, target='battery_health_percent')