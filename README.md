[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/framunoz/simulated-annealing-for-traveling-salesman-problem)

# Simulated Annealing for Traveling Salesman Problem

This project is a **hybrid Python-Julia library** designed to solve the **Traveling Salesman Problem (TSP)** using the **Simulated Annealing** algorithm. It leverages the ease of use of Python for the interface and orchestration, while harnessing the high-performance capabilities of Julia for the core computational kernels.

The kernels implemented in this library are based on the research:
> [A Hybrid Simulated Annealing Algorithm for Travelling Salesman Problem with Three Neighbor Generation Structures](https://hal.science/hal-01962049/document)

## Features

-   **Hybrid Architecture**: Seamless integration of Python and Julia using `juliacall`.
-   **Simulated Annealing**: Robust implementation of the SA metaheuristic.
-   **Performance**: Critical path operations implemented in Julia for maximum speed.
-   **Visualization**: Jupyter notebooks provided for visualizing the TSP solutions.
-   **Modern Tooling**: Managed with `uv` for fast and reliable dependency handling.

## Prerequisites

-   **Python**: 3.13 or higher.
-   **Julia**: The project uses `juliacall`, which will automatically handle the Julia installation (typically ~1.10+).
-   **Git**: For cloning the repository.
-   **uv**: A modern Python package installer and resolver.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for efficient dependency management.

1.  **Install uv** (if not already installed):
    ```bash
    # On macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # On Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/framunoz/simulated-annealing-for-traveling-salesman-problem.git
    cd simulated-annealing-for-traveling-salesman-problem
    ```

3.  **Sync dependencies**:
    This command will create the virtual environment and install all necessary Python and Julia dependencies.
    ```bash
    uv sync
    ```

## Usage

### Running the Solver
To run the main TSP solver via the Python interface:

```bash
uv run python -m tsp
```

### Examples & Visualization
Explore the `notebook_examples` directory for interactive usage and visualizations:

```bash
# Launch Jupyter Lab or Notebook
uv run jupyter lab notebook_examples/examples.ipynb
```

## Project Structure

The codebase is organized into two main source directories:

-   `src/tsp`: **Python Source**. Contains the application logic, CLI entry points, and wrappers for the Julia code.
-   `src/TspJulia`: **Julia Source**. Contains the performant implementations of the Simulated Annealing kernels and TSP data structures.

## Development

To set up the environment for development (including running tests and linting):

1.  **Install development dependencies**:
    ```bash
    uv sync --group dev
    ```

2.  **Run Tests**:
    ```bash
    uv run pytest
    ```

## License

MIT License. See [LICENSE](LICENSE) for more details.
