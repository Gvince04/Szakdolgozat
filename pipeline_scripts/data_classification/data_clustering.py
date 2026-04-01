#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import sys
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import plotly.express as px

import json

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shared_utils.status_manager as status_manager


# In[2]:


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster and Visualize Embeddings (1024D -> 2D).")

    defaults = {
        "input_dir": "data/zengo_embedded",
        "output_dir": "data/zengo_results",
        "generate_plot": False,
        "algorithm": "kmeans",
        "n_clusters": 2,
        "eps": 10.0,
        "min_samples": 3,
        "reducer": "pca",
        "perplexity": 30.0
    }

    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=defaults["input_dir"], 
        help=f"Path to the root directory containing the embedded .npy files (default: { defaults['input_dir'] })"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=defaults["output_dir"], 
        help=f"Path to the root directory where the plot will be saved (default: { defaults['output_dir'] })"
    )

    parser.add_argument(
        "--generate_plot", 
        action="store_true", 
        default=defaults["generate_plot"], 
        help=f"If set, generates a plot (default: { defaults['generate_plot'] })"
    )

    parser.add_argument(
        "--algorithm", 
        type=str, 
        default=defaults["algorithm"], 
        choices=["kmeans", "dbscan"], 
        help=f"Clustering algorithm to use (default: { defaults['algorithm'] })"
    )

    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=defaults["n_clusters"], 
        help=f"Number of clusters for K-Means (default: { defaults['n_clusters'] })"
    )

    parser.add_argument(
        "--eps", 
        type=float, 
        default=defaults["eps"], 
        help=f"DBSCAN epsilon distance. Higher = fewer clusters, less noise. (default: { defaults['eps'] })"
    )

    parser.add_argument(
        "--min_samples", 
        type=int, 
        default=defaults["min_samples"], 
        help=f"DBSCAN min_samples. Minimum points to form a dense region. (default: { defaults['min_samples'] })"
    )

    parser.add_argument(
        "--reducer", 
        type=str, 
        default=defaults["reducer"], 
        choices=["pca", "tsne"], 
        help=f"Method to reduce 1024D -> 2D (default: { defaults['reducer'] })"
    )

    parser.add_argument(
        "--perplexity", 
        type=float, 
        default=defaults["perplexity"], 
        help=f"t-SNE perplexity (default: { defaults['perplexity'] })"
    )

    return parser.parse_args()


# In[ ]:


def load_data_recursive(input_dir):
    embeddings_list = []
    filenames = []

    print(f"Scanning '{input_dir}' for embeddings...")

    files_found = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)

                try:
                    data = np.load(file_path)

                    if data.ndim == 0:
                        continue
                    elif data.ndim == 2:
                        data = data.squeeze()

                    if data.shape != (1024,):
                        continue

                    embeddings_list.append(data)
                    filenames.append(os.path.basename(root))
                    files_found += 1

                except Exception as e:
                    print(f"Error loading {file}: {e}")

    if files_found == 0:
        return None, None

    print(f"Found {files_found} valid files.")

    X = np.vstack(embeddings_list)
    return X, filenames


# In[ ]:


def main():
    args = parse_args()

    # LOAD DATA
    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' not found.")
        sys.exit(0)

    X, file_labels = load_data_recursive(args.input_dir)

    if X is None:
        print("No valid embeddings found. Check your directory.")
        sys.exit(0)

    print(f"Final Matrix Shape: {X.shape} (Samples: {X.shape[0]}, Features: {X.shape[1]})")

    status_manager.update_progress("Clustering", 1/6 * 100)

    # PREPROCESSING
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    status_manager.update_progress("Clustering", 2/6 * 100)

    # CLUSTERING
    print(f"Running Clustering ({args.algorithm})...")

    if args.algorithm == "kmeans":
        print(f"KMeans with {args.n_clusters} clusters")
        model = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

    elif args.algorithm == "dbscan":
        print(f"DBSCAN with eps={args.eps}, min_samples={args.min_samples}")
        model = DBSCAN(eps=args.eps, min_samples=3)
        labels = model.fit_predict(X_scaled)

    status_manager.update_progress("Clustering", 3/6 * 100)

    # DIMENSIONALITY REDUCTION
    print(f"Reducing dimensions using {args.reducer.upper()}...")

    if args.reducer == "pca":
        reducer = PCA(n_components=2)
        X_2d = reducer.fit_transform(X_scaled)

    elif args.reducer == "tsne":
        pca_50 = PCA(n_components=min(50, X.shape[1]))
        X_pca = pca_50.fit_transform(X_scaled)

        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, init='pca', learning_rate='auto')
        X_2d = tsne.fit_transform(X_pca)

    status_manager.update_progress("Clustering", 4/6 * 100)

    # VISUALIZATION
    df_plot = pd.DataFrame(X_2d, columns=['Component 1', 'Component 2'])

    df_plot['Cluster'] = labels.astype(str) 
    df_plot['Source'] = file_labels

    if args.generate_plot:
        print("Generating Interactive Plot...")

        title = f"MOMENT Embeddings: {args.reducer.upper()} Projection<br>({args.algorithm.capitalize()})"

        fig = px.scatter(
            df_plot,
            x='Component 1',
            y='Component 2',
            color='Cluster',
            symbol='Cluster',
            hover_name='Source',
            hover_data={'Cluster': True, 'Component 1': False, 'Component 2': False},
            title=title,
            labels={
                'Component 1': f"{args.reducer.upper()} Dimension 1",
                'Component 2': f"{args.reducer.upper()} Dimension 2"
            },
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')), opacity=0.8)
        fig.update_layout(template="plotly_white")

        os.makedirs(args.output_dir, exist_ok=True)
        filename = f"plot_{args.algorithm}_{args.reducer}.html"
        save_path = os.path.join(args.output_dir, filename)

        fig.write_html(save_path)
        print(f"Interactive plot saved to: {save_path}")

    status_manager.update_progress("Clustering", 5/6 * 100)

    # DATA JSON
    print("Exporting raw data for GUI...")
    json_save_path = os.path.join(args.output_dir, "real_chart_data.json")
    chart_data = []

    for idx, row in df_plot.iterrows():
        chart_data.append({
            "x": row['Component 1'],
            "y": row['Component 2'],
            "cluster": row['Cluster'],
            "source": row['Source']
        })

    with open(json_save_path, 'w') as f:
        json.dump(chart_data, f, indent=4)

    print(f"GUI data saved to: {json_save_path}")

    status_manager.update_progress("Clustering", 100)



# In[ ]:


# print("Generating Plot...")

#     df_plot = pd.DataFrame(X_2d, columns=['Component 1', 'Component 2'])
#     df_plot['Cluster'] = labels
#     df_plot['Source'] = file_labels 

#     plt.figure(figsize=(12, 10))

#     sns.scatterplot(
#         data=df_plot,
#         x='Component 1',
#         y='Component 2',
#         hue='Cluster',
#         palette='viridis',
#         style='Cluster',
#         s=100,
#         alpha=0.8,
#         edgecolor='k'
#     )

#     algo_params = f"k={args.n_clusters}" if args.algorithm == "kmeans" else f"eps={args.eps}"
#     title = f"MOMENT Embeddings: {args.reducer.upper()} Projection\n({args.algorithm.capitalize()}, {args.n_clusters} Clusters)"
#     plt.title(title, fontsize=16)
#     plt.xlabel(f"{args.reducer.upper()} Dimension 1")
#     plt.ylabel(f"{args.reducer.upper()} Dimension 2")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Cluster ID")

#     # SAVE
#     os.makedirs(args.output_dir, exist_ok=True)
#     filename = f"plot_{args.algorithm}_{args.reducer}.png"
#     save_path = os.path.join(args.output_dir, filename)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     print(f"Plot saved to: {save_path}")

#     # OPTIONAL: Save the combined data if you want to reuse it
#     # np.save(os.path.join(args.output_dir, "X_combined.npy"), X)


# In[ ]:


if __name__ == "__main__":
    main()

