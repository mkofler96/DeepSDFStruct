import torch
import streamlit as st
import os
import io
import meshio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from streamlit_stl import stl_from_text

import random


from DeepSDFStruct.pretrained_models import get_model, PRETRAINED_MODELS_DIR
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.mesh import create_3D_mesh

from mlflow import MlflowClient
from mlflow.entities import Experiment

client = MlflowClient(tracking_uri="sqlite:///mlruns.db")

all_exp = client.search_experiments()
experiments = {exp.name: exp for exp in all_exp}


def get_runs_from_experiment(experiment: Experiment):
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    return runs


st.set_page_config(page_title="Cluster Shape Explorer", layout="wide")

# -------------------------------------------------------------------------
# Sidebar: Model Selector
# -------------------------------------------------------------------------

available_dirs = ["Pretrained Models"] + list(experiments.keys())

model_dir = st.sidebar.selectbox("Select experiment", available_dirs, 0)

available_models = {}

if model_dir == "Pretrained Models":
    # Local folder models
    for name in os.listdir(PRETRAINED_MODELS_DIR):
        path = os.path.join(PRETRAINED_MODELS_DIR, name)
        if os.path.isdir(path):
            available_models[name] = path

else:
    # MLflow experiment (runs)
    ml_exp = experiments[model_dir]
    runs = client.search_runs(experiment_ids=[ml_exp.experiment_id])

    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        available_models[run_name] = run.info.artifact_uri
model_name = st.sidebar.selectbox("Select a model", available_models)

local_path = available_models[model_name]
epoch_dir = os.path.join(local_path, "ModelParameters")

available_epochs = [
    file[:-4]
    for file in os.listdir(epoch_dir)
    if file.endswith(".pth") or file.endswith(".pt")
]

if "cluster_index" not in st.session_state.keys():
    st.session_state.cluster_index = 0

epoch = st.sidebar.selectbox(
    "Select epoch", available_epochs, len(available_epochs) - 1
)


# -------------------------------------------------------------------------
# Load model once
# -------------------------------------------------------------------------
@st.cache_resource
def load_model(model_name: str, epoch: str):
    """
    model_dir: 'deepsdf_experiments' or mlflow experiment name
    model_name: run-name or folder-name
    epoch: checkpoint epoch
    """
    model_path = available_models[model_name]
    model = get_model(model_path, checkpoint=epoch)
    sdf = SDFfromDeepSDF(model)
    return model, sdf


model, sdf = load_model(model_name, epoch)
latent_vectors = model._trained_latent_vectors.detach().cpu().numpy()


# -------------------------------------------------------------------------
# Cluster computation (TSNE + KMeans)
# -------------------------------------------------------------------------
@st.cache_resource
def compute_clusters(latent_vectors, n_clusters=10, perplexity=3):
    X_embedded = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=perplexity
    ).fit_transform(latent_vectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_embedded)
    return kmeans.labels_, X_embedded


n_clusters = st.sidebar.slider(
    "Number of clusters", min_value=2, max_value=20, value=10
)
labels, X_embedded = compute_clusters(latent_vectors, n_clusters=n_clusters)

# -------------------------------------------------------------------------
# Sidebar: Cluster selector
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Navigation buttons (main window)
# -------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Prev"):
        st.session_state.cluster_index = max(0, st.session_state.cluster_index - 1)
with col3:
    if st.button("Next"):
        st.session_state.cluster_index = min(
            n_clusters - 1, st.session_state.cluster_index + 1
        )
with col2:
    st.text(f"Cluster {st.session_state.cluster_index+1}/{n_clusters}")

indices_in_cluster = np.argwhere(labels == st.session_state.cluster_index).flatten()


# -------------------------------------------------------------------------
# STL generation function
# -------------------------------------------------------------------------
def generate_stl(lat_vec, color_rgb):
    lat_vec_torch = torch.tensor(lat_vec, dtype=torch.float32, device=model.device)
    sdf.set_latent_vec(lat_vec_torch)
    with torch.inference_mode():
        surf_mesh, _ = create_3D_mesh(
            sdf, 100, differentiate=False, mesh_type="surface", device=model.device
        )
    surf_mesh_gus = surf_mesh.to_gus()
    surf_mesh_gus.show_options["c"] = color_rgb  # apply color
    mesh = meshio.Mesh(
        points=surf_mesh_gus.vertices, cells=[("triangle", surf_mesh_gus.faces)]
    )
    output = io.StringIO()
    meshio.write(output, mesh, file_format="stl", binary=False)
    return output


# -------------------------------------------------------------------------
# Display selected geometry
# -------------------------------------------------------------------------
def rgb_to_hex(rgb):
    """Convert an RGB list [R,G,B] to hex string."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


indices = [random.choice(list(indices_in_cluster)) for i in range(4)]
n_select = min(5, len(indices_in_cluster))

indices = random.sample(list(indices_in_cluster), n_select)
columns_row1 = st.columns(3)
columns_row2 = st.columns(3)

fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(
    X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="tab10", alpha=0.6
)

# Map cluster id to color
cmap = plt.get_cmap("tab10")
cluster_color = [int(255 * c) for c in cmap(st.session_state.cluster_index)[:3]]

ax.scatter(
    X_embedded[indices, 0],
    X_embedded[indices, 1],
    edgecolors="k",
    facecolor=cmap(st.session_state.cluster_index),
    s=120,
    linewidths=2,
)
plt.colorbar(scatter, ax=ax, label="Cluster")
columns_row1[0].pyplot(fig)

for selected_idx, column in zip(indices, columns_row1[1:] + columns_row2):
    lat_vec = latent_vectors[selected_idx]

    # st.subheader(
    #     f"Latent vector index: {selected_idx} (Cluster {st.session_state.cluster_index})"
    # )
    stl_io = generate_stl(lat_vec, cluster_color)
    with column:
        stl_from_text(
            text=stl_io.getvalue(),
            color=rgb_to_hex(cluster_color),
            material="material",
            auto_rotate=True,
            opacity=1,
            shininess=10,
            cam_v_angle=60,
            cam_h_angle=-90,
            height=500,
        )
