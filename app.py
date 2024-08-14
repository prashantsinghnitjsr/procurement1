import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import io

# Streamlit app title
st.title("Procurement SKU Classifications")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load CSV data
    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1', low_memory=False)

    # Process button
    if st.button("Process Data"):
        # Ensure Description column is of string type and handle missing values
        data['Description'] = data['Description'].astype(str).fillna('')

        # Rename columns to remove any leading/trailing spaces
        data.columns = data.columns.str.strip()

        # Handle missing or non-numeric values in 'Unit Cost'
        data['Unit Cost'] = pd.to_numeric(data['Unit Cost'], errors='coerce')
        data['Unit Cost'].fillna(data['Unit Cost'].mean(), inplace=True)  # Replace NaNs with mean

        # Classify based on Description using TF-IDF and K-means Clustering
        tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = tfidf.fit_transform(data['Description'])

        # Determine the number of clusters
        n_clusters = max(100, len(data) // 50)  # At least 100 clusters, or more for larger datasets

        # Using K-means for Description Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data['Classification - Description'] = kmeans.fit_predict(tfidf_matrix)

        # Initialize a column for storing price classification results
        data['Classification - Price'] = 'OK'

        # Perform price clustering and anomaly detection within each description cluster
        scaler = StandardScaler()
        for cluster in range(n_clusters):
            cluster_indices = data['Classification - Description'] == cluster
            cluster_data = data[cluster_indices]
            
            if len(cluster_data) > 1:  # Ensure there are at least two data points in the cluster
                # Scale the Unit Cost within this cluster
                unit_cost_scaled = scaler.fit_transform(cluster_data[['Unit Cost']])
                
                # Always mark the most extreme value as an anomaly
                most_extreme_index = np.argmax(np.abs(unit_cost_scaled))
                anomalies = np.zeros(len(cluster_data), dtype=int)
                anomalies[most_extreme_index] = -1
                
                if len(cluster_data) > 5:  # For larger clusters, use Isolation Forest for additional anomalies
                    # Calculate the number of additional anomalies to detect (up to 30% of the cluster)
                    n_additional_anomalies = max(0, int(0.3 * len(cluster_data)) - 1)
                    if n_additional_anomalies > 0:
                        contamination = n_additional_anomalies / (len(cluster_data) - 1)  # Exclude the already marked anomaly
                        
                        # Use Isolation Forest to detect additional anomalies within this cluster
                        iso_forest = IsolationForest(contamination=contamination, random_state=42)
                        additional_anomalies = iso_forest.fit_predict(np.delete(unit_cost_scaled, most_extreme_index, axis=0))
                        
                        # Combine the results, ensuring we don't overwrite the most extreme anomaly
                        anomalies[np.arange(len(cluster_data)) != most_extreme_index] = additional_anomalies
                
                # Mark anomalies as 'Red Flag' and others as 'OK'
                data.loc[cluster_indices, 'Classification - Price'] = np.where(anomalies == -1, 'Red Flag', 'OK')

        # Create a download link for the Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, index=False)
        output.seek(0)
        
        st.download_button(
            label="Download Classified Data",
            data=output,
            file_name="classified_procurement_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("Data processing completed. You can now download the classified data.")