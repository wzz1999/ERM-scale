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
-   `fig_plot/`: Contains scripts to generate figures. Subdirectories correspond to specific figures: `fig2_experimental_observation/`, `fig3_and_supp/`, `fig4_and_supp/`, `fig5_and_supp/`, and `fig6_oldversion/`.
-   `load_data/`: Contains scripts for loading data, specifically `load_fish_data2.jl`.
-   `Project.toml`: Defines the Julia project environment and dependencies.
-   `Manifest.toml`: Records the exact versions of all dependencies used in the project environment.

## Setup and Usage

1.  **Install Julia:** Ensure you have Julia installed on your system. You can download it from [julialang.org](https://julialang.org/).
2.  **Clone the repository:**
    ```bash
    git clone https://github.com/wzz1999/ERM-scale.git
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
    -   Run scripts in the `fig_plot/` directory to reproduce specific figures. For example, to generate Figure 3 from your terminal:
        ```bash
        julia fig_plot/fig3_and_supp/fig3.jl
        ```
    -   Use `load_data/load_fish_data2.jl` potentially as part of the figure generation or analysis scripts.

## Data Requirements

The analysis scripts, particularly `load_data/load_fish_data2.jl`, expect experimental data (e.g., spike data, ROI coordinates) in `.mat` format.

-   **Directory Structure:** The data for a specific experiment (identified by `ifish`) should reside in a subdirectory named after that ID (e.g., `210106/`) within a main data root directory.
-   **Expected Files:** Key files like `spike_OASIS.mat`, `Judge.mat`, `Result.mat`, and `center_Coord.mat` are expected within the `<fish_id>` subdirectory.
-   **Data Path Configuration:** The script `load_data/load_fish_data2.jl` attempts to automatically find the data root directory by checking predefined paths (e.g., `D:/Fish-Brain-Behavior-Analysis/results`, `/home/data2/wangzezhen/fishdata`). Other scripts, particularly in the `src/` directory like `PR_ratio.jl`, might depend on a base path defined by the `ERM_ROOT` constant.
-   **Action Required:** 
    - If your data is located elsewhere, **you must manually edit the `dataroot` and `dataroot2` path definitions** near the beginning of the `load_data/load_fish_data2.jl` script. 
    - Additionally, **check the `ERM_ROOT` constant defined in `src/util2.jl`** (currently hardcoded to `/home/wenlab-user/wangzezhen/ERM-scale`) and update it to your project's root directory if necessary, as other scripts might use it to locate results or other data files. You may also need to adjust the default `ifish` variable within the script or set it before running dependent scripts.

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

