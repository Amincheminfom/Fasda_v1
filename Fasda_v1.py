import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import io
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests

# Author : Dr. Sk. Abdul Amin

logo_url = "https://github.com/Amincheminfom/Fasda_v1/raw/main/Fasda_logo.jpg"

st.set_page_config(
    page_title="Fasda_v1",
    layout="wide",
    page_icon=logo_url
)

st.sidebar.image(logo_url)

# Title and About
st.title("Fingerprint-assisted  scaffold diversity analysis")
about_expander = st.expander("What is Fasda_v1?", expanded=True)
with about_expander:
    st.write('''
             Fasda_v1 is a python package for exploring molecular profiles and scaffold diversity. 
    
    Features include
    - Fingerprint generation for ECFP types.
    - Tanimoto similarity calculation.
    - Dimensionality reduction with PCA or t-SNE.
    - Clustering with KMeans.
    - Bemis-Murcko scaffold analysis.
    - Visualization and CSV export.
    
    For more details
    You can find more details in our [paper](https://www.scopus.com/authid/detail.uri?authorId=57190176332).''')

# Author : Dr. Sk. Abdul Amin
st.subheader('Select the Dataset')
dataset_choice = st.selectbox('Choose to upload your dataset or use a sample file:',
                              ('Sample Dataset', 'Upload Dataset'))

sample_file_url = "https://github.com/Amincheminfom/Fasda_v1/raw/main/Fasda_v1_dataset.csv"

if dataset_choice == 'Sample Dataset':
    uploaded_file = sample_file_url
else:
    uploaded_file = st.file_uploader("Upload your CSV file with column 'Smiles'", type=["csv"])

st.sidebar.header("Parameters")
fingerprint_radius = st.sidebar.selectbox("Fingerprint Radius", options=[2, 3, 4, 6, 8], index=1)
num_bits = st.sidebar.slider("Fingerprint Bit Size", 512, 4096, 2048, step=512)
reduction_method = st.sidebar.selectbox("Dimensionality Reduction Method", options=["PCA", "t-SNE"])
num_clusters = st.sidebar.selectbox("Number of Clusters", options=[3, 4, 5, 6, 7, 8], index=2)

def generate_ecfp_fingerprint(smiles, radius, nBits):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) if mol else None

def bemis_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) if mol else None

if uploaded_file:
    # Author : Dr. Sk. Abdul Amin
    data = pd.read_csv(uploaded_file)
    if 'Smiles' in data.columns:
        data['Fingerprint'] = data['Smiles'].apply(generate_ecfp_fingerprint, radius=fingerprint_radius, nBits=num_bits)

        similarity_matrix = np.zeros((len(data), len(data)))
        for i, fp1 in enumerate(data['Fingerprint']):
            for j, fp2 in enumerate(data['Fingerprint']):
                similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fp1, fp2) if fp1 and fp2 else 0

        reducer = PCA(n_components=2, random_state=42) if reduction_method == "PCA" else TSNE(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(similarity_matrix)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data['Cluster'] = kmeans.fit_predict(reduced_data)

        # Author : Dr. Sk. Abdul Amin
        data['Scaffold'] = data['Smiles'].apply(bemis_murcko_scaffold)
        cluster_summary = []
        for cluster_id in range(num_clusters):
            cluster_data = data[data['Cluster'] == cluster_id]
            scaffold_counts = Counter(cluster_data['Scaffold'])
            cluster_summary.append({
                'Cluster': cluster_id,
                'Number of Compounds': len(cluster_data),
                'Unique Scaffolds': len(scaffold_counts),
                'Singleton Scaffolds': sum(1 for count in scaffold_counts.values() if count == 1),
                'Singleton Ratio': sum(1 for count in scaffold_counts.values() if count == 1) / len(scaffold_counts) if scaffold_counts else 0
            })
        summary_df = pd.DataFrame(cluster_summary)

        # Author : Dr. Sk. Abdul Amin
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Dataset Preview")
            st.dataframe(data.head())
            st.write("### Cluster Summary")
            st.dataframe(summary_df)
            st.download_button("Download Summary CSV", data=summary_df.to_csv(index=False), file_name='cluster_summary.csv')

        with col2:
            st.write("### Visualizing Clusters")

            response = requests.get(logo_url)
            logo_img = plt.imread(BytesIO(response.content), format="JPG")

            fig, ax = plt.subplots(figsize=(10, 8))

            cmap = plt.cm.get_cmap('viridis', num_clusters)  # num_clusters is the total number of clusters
            scatter = ax.scatter(
                reduced_data[:, 0],
                reduced_data[:, 1],
                c=data['Cluster'],
                cmap=cmap,
                s=50,
                edgecolor='k'
            )
            ax.set_xlabel(f"{reduction_method} Component 1")
            ax.set_ylabel(f"{reduction_method} Component 2")
            ax.set_title(f"{reduction_method} Visualization with Clusters")

            cbar = plt.colorbar(scatter, ax=ax, boundaries=np.arange(-0.5, num_clusters + 0.5, 1), ticks=np.arange(0, num_clusters, 1))
            cbar.set_label("Cluster")
            cbar.ax.set_yticklabels([str(i) for i in range(num_clusters)])

            # Add logo below the colorbar
            box = OffsetImage(logo_img, zoom=0.05)
            ab = AnnotationBbox(
                box,
                (0.97, -0.08),
                xycoords=cbar.ax.transAxes,
                frameon=False
            )
            fig.add_artist(ab)


            st.pyplot(fig)

            # Author : Dr. Sk. Abdul Amin
            plot_buffer = io.BytesIO()
            fig.savefig(plot_buffer, format="png", bbox_inches="tight")
            plot_buffer.seek(0)

            st.download_button(
                label="Download Plot",
                data=plot_buffer,
                file_name="cluster_visualization.png",
                mime="image/png"
            )

    else:
        st.error("The uploaded file must contain the column 'Smiles'.")

# Author : Dr. Sk. Abdul Amin
contacts = st.expander("Contact", expanded=False)
with contacts:
    st.write('''
             #### Report an Issue 

             You are welcome to report a bug or contribute to the web 
             application by filing an issue on [Github](https://github.com/Amincheminfom).

             #### Contact

             For any question, you can contact us through email:

             - [Dr. Amin](mailto:pharmacist.amin@gmail.com)
             ''')