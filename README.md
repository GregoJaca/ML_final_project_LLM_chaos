This project investigates the "chaotic" behavior of Large Language Models (LLMs) by analyzing the evolution of their internal hidden states during text generation. It aims to quantify the sensitivity of LLM responses to initial conditions by examining the trajectories of hidden states.

## Key Features

*   **LLM Prompting & Data Extraction:** Utilizes DeepSeek-R1 via Hugging Face Transformers to generate text from various prompts, extracting hidden states from different layers (first, middle, last) and generated tokens. It also calculates cosine similarity and L2 (Euclidean) distance matrices between these hidden states.
*   **Dimensionality Reduction & Trajectory Visualization:** Employs t-SNE and PCA to reduce the dimensionality of hidden states, enabling 2D and 3D visualization of the LLM's internal state trajectories.
*   **Recurrence Analysis:** Generates recurrence plots to identify recurring patterns and structures within the hidden state trajectories, indicating potential self-similar or cyclical behavior.
*   **Fractal Dimension Calculation:** Computes pointwise and correlation dimensions to characterize the complexity and fractal nature of the hidden state trajectories.


## Project Structure

*   `run_llm_and_generate_data.py`: Handles LLM interaction, text generation, and extraction of hidden states and similarity/distance matrices.
*   `pointwise_and_correlation_dimensions_good.py`: Dedicated script for calculating correlation dimension.
*   `recurrence_plot_text_WORD_maker.py`: Generates recurrence plots and compiles them into document formats (Word/PDF/HTML).

## Clustering and Visualization of Hidden States

This project includes a comprehensive pipeline for clustering and visualizing the hidden state representations of LLMs. The main goals are to identify semantically meaningful clusters in the high-dimensional hidden state space and to distinguish them from structural clusters (e.g., tokens like '**', ':**', or punctuation) that do not carry semantic meaning.

### Approach

- **Clustering Algorithms:** The code implements and compares several clustering algorithms, including KMeans, DBSCAN, Spectral Clustering, and Hierarchical Clustering. Each method is evaluated using standard clustering quality metrics such as Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Score.
- **Structural Cluster Filtering:** Special logic is used to identify and filter out clusters that correspond to structural tokens (punctuation, formatting, etc.), ensuring that the analysis focuses on semantically relevant groups.
- **Dimensionality Reduction:** For visualization, the pipeline supports PCA, t-SNE, and UMAP. These methods reduce the high-dimensional hidden states to 2D for effective plotting and exploration.
- **Visualization:** The results include interactive visualizations of clusters, cluster size distributions, and comparisons of clustering quality across methods. The pipeline also provides tools to inspect sample tokens from each cluster.

### Key Findings

- **Best Visualization Method:** t-SNE was found to be the most effective dimensionality reduction technique for visualizing clusters, as it preserves local structure and separates similar tokens well.
- **Best Clustering Methods:** KMeans and Hierarchical Clustering produced the most semantically meaningful clusters for the first layer hidden states. DBSCAN and Spectral Clustering were less effective with default parameters.
- **Layer-wise Differences:** Clustering was most effective on the first layer hidden states, which are more representative of individual token meaning. Last layer hidden states, which incorporate more context, were harder to cluster meaningfully.
- **Structural vs. Semantic Clusters:** The pipeline successfully identifies and removes structural clusters, allowing for a clearer analysis of semantic groupings.

### Example Usage

- The notebook demonstrates how to run the clustering pipeline, compare algorithms, and visualize results. It also provides commentary on the strengths and weaknesses of each method, with practical guidance for interpreting cluster quality and structure.