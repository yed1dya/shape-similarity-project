#!/usr/bin/env python3
# tsne_unlabeled.py

import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go


def main():
    # --- 1. Load CSV ---
    csv_filename = "shape14_features_20250718_174813.csv"
    if not os.path.isfile(csv_filename):
        raise FileNotFoundError(f"Could not find '{csv_filename}' in {os.getcwd()}")
    df = pd.read_csv(csv_filename, dtype={'child_id': str})

    # --- 2. Identify ID and feature columns ---
    id_col = 'child_id'
    if id_col not in df.columns:
        raise KeyError(f"'{id_col}' column not found in CSV")
    feature_cols = [c for c in df.columns if c != id_col]

    # --- 3. Drop feature columns with any NaNs ---
    feat_df = df[feature_cols].dropna(axis=1)
    filtered_cols = feat_df.columns.tolist()
    print(f"Dropped {len(feature_cols) - len(filtered_cols)} columns with NaNs.")

    # --- 4. Remove highly correlated features ---
    corr = feat_df[filtered_cols].corr().abs()
    mask = np.tril(np.ones_like(corr, dtype=bool), k=0)
    upper = corr.where(~mask)
    threshold = 0.90
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    final_cols = [c for c in filtered_cols if c not in to_drop]
    print(f"Dropped {len(to_drop)} highly correlated features (>{threshold}).")

    # --- 5. Prepare data for t-SNE ---
    X = feat_df[final_cols].values
    ids = df[id_col].tolist()

    # --- 6. Run t-SNE to 3D ---
    tsne = TSNE(n_components=3, random_state=42)
    embedded = tsne.fit_transform(X)

    # --- 7. Rank by Euclidean distance from reference '0000' ---
    if '0000' not in ids:
        raise ValueError("Reference ID '0000' not found in child_id column")
    ref_idx = ids.index('0000')
    ref_point = embedded[ref_idx]
    distances = np.linalg.norm(embedded - ref_point, axis=1)
    order = np.argsort(distances)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))

    rank_df = pd.DataFrame({
        'child_id': ids,
        'rank': ranks
    })
    rank_df_sorted = rank_df.sort_values('rank')
    rank_csv = 'shape14_tsne_ranking.csv'
    rank_df_sorted.to_csv(rank_csv, index=False)
    print(f"Saved ranking CSV to '{rank_csv}'")

    # --- 8. Build Plotly 3D scatter plot ---
    colors = ['red' if cid == '0000' else 'gray' for cid in ids]
    fig = go.Figure(
        data=go.Scatter3d(
            x=embedded[:, 0],
            y=embedded[:, 1],
            z=embedded[:, 2],
            mode='markers',
            marker=dict(size=4, opacity=0.8, color=colors),
            text=[f"Point {ranks[i]}: {cid}" for i, cid in enumerate(ids)],
            hoverinfo='text'
        )
    )
    fig.update_layout(
        title="3D t-SNE of shape14_features (unlabeled)",
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3",
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # --- 9. Save interactive HTML ---
    output_html = "shape14_tsne_unlabeled_3d.html"
    fig.write_html(output_html)
    print(f"Saved 3D t-SNE plot to '{output_html}'")

if __name__ == "__main__":
    main()
