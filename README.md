# ERM-scale

This repository contains the Julia code for the elife paper:
Wang Zezhen, Mai Weihao, Chai Yuming, Qi Kexin, Ren Hongtai, Shen Chen, Zhang Shiwu, Tan Guodong, Hu Yu, Wen Quan (2025) The Geometry and Dimensionality of Brain-wide Activity eLife 14:RP100666

https://doi.org/10.7554/eLife.100666.2

## Project Structure

-   `src/`: Contains the core Julia source code, including utility functions, subsampling methods, MDS implementations, and potentially the main ERM logic.
    -   `ERM.jl`: The central ERM implementation.
    -   `mds.jl`, `mds_cca_batch.jl`: Multidimensional scaling functions.
    -   `subsampling_functions.jl`: Functions related to data subsampling.
    -   `t_pdf_FT.jl`: t-distribution probability density functions or Fourier transforms.
    -   `util.jl`, `util2.jl`: Utility functions.
-   `fig_plot/`: Contains scripts to generate figures. Subdirectories correspond to specific figures (Fig 2, Fig 3, Fig 4, Fig 5) and their supplementary materials. Scripts for Figure 6 and kernel regression are also present.
-   `load_data/`: Contains scripts for loading data, specifically `load_fish_data2.jl`.
-   `Project.toml`: Defines the Julia project environment and dependencies.
-   `Manifest.toml`: Records the exact versions of all dependencies used in the project environment.

## Setup and Usage

1.  **Install Julia:** Ensure you have Julia installed on your system. You can download it from [julialang.org](https://julialang.org/).
2.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ERM-scale
    ```
3.  **Instantiate the environment:** Open a Julia REPL in the `ERM-scale` directory and activate the project environment. Then, instantiate it to install all necessary packages specified in `Project.toml` and `Manifest.toml`.
    ```julia
    julia> using Pkg
    julia> Pkg.activate(".")
    julia> Pkg.instantiate()
    ```
4.  **Running the code:**
    -   Explore the scripts in the `src/` directory for core functionalities.
    -   Run scripts in the `fig_plot/` directory to reproduce specific figures. For example, to generate Figure 6:
        ```julia
        julia> include("fig_plot/fig_6.jl")
        ```
    -   Use `load_data/load_fish_data2.jl` potentially as part of the figure generation or analysis scripts.

## Dependencies

The project relies on several Julia packages for numerical computation, machine learning, plotting, and data handling. Key dependencies include:

-   Flux.jl
-   CUDA.jl
-   Plots.jl
-   Distributions.jl
-   Statistics.jl
-   Clustering.jl
-   MultivariateStats.jl
-   ... and others listed in `Project.toml`.

The `Manifest.toml` file ensures reproducibility by locking the exact versions of all dependencies.

