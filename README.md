This project investigates the "chaotic" behavior of Large Language Models (LLMs) by analyzing the evolution of their internal hidden states during text generation. It aims to quantify the sensitivity of LLM responses to initial conditions by examining the trajectories of hidden states.

## Key Features

*   **LLM Prompting & Data Extraction:** Utilizes DeepSeek-R1 via Hugging Face Transformers to generate text from various prompts, extracting hidden states from different layers (first, middle, last) and generated tokens. It also calculates cosine similarity and L2 (Euclidean) distance matrices between these hidden states.
*   **Dimensionality Reduction & Trajectory Visualization:** Employs t-SNE and PCA to reduce the dimensionality of hidden states, enabling 2D and 3D visualization of the LLM's internal state trajectories.
*   **Recurrence Analysis:** Generates recurrence plots to identify recurring patterns and structures within the hidden state trajectories, indicating potential self-similar or cyclical behavior.
*   **Fractal Dimension Calculation:** Computes pointwise and correlation dimensions to characterize the complexity and fractal nature of the hidden state trajectories.
*   **Network Analysis:** Constructs complex networks from recurrence matrices to analyze community structures, centrality measures, and transitions between communities within the LLM's internal state space.
*   **Autoencoder for Trajectory Compression:** Includes an autoencoder to compress hidden state trajectories into a lower-dimensional latent space while preserving their structure, facilitating further analysis.

## Project Structure

*   `run_llm_and_generate_data.py`: Handles LLM interaction, text generation, and extraction of hidden states and similarity/distance matrices.
*   `traj3.ipynb` / `traj4.ipynb`: Jupyter notebooks for comprehensive analysis and visualization of extracted data, including dimensionality reduction, recurrence plots, and fractal dimension calculations.
*   `network_analysis.py` / `network.py`: Scripts for performing network analysis on recurrence matrices, including community detection and various network metrics.
*   `pointwise_and_correlation_dimensions_good.py`: Dedicated script for calculating correlation dimension.
*   `encoder_stacked.py`: Implements an autoencoder for dimensionality reduction of hidden state trajectories.
*   `recurrence_plot_text_WORD_maker.py`: Generates recurrence plots and compiles them into document formats (Word/PDF/HTML).