# Simulated Annealing for Traveling Salesman Problem

This is a library that makes use of the Simulated Annealing 
algorithm to obtain an approximate solution to the 
Traveling Salesman Problem. To see an example of the use of 
this library, you can look at the notebook 
[examples.ipynb](./notebook_examples/examples.ipynb)

The kernels implemented in this library were obtained from the reference 
[A Hybrid Simulated Annealing Algorithm for Travelling
Salesman Problem with Three Neighbor Generation
Structures](https://hal.science/hal-01962049/document)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. To get started:

1. Install uv if you haven't already:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone the repository and sync dependencies:
   ```bash
   git clone <repository-url>
   cd simulated-annealing-for-traveling-salesman-problem
   uv sync
   ```

3. Run the project:
   ```bash
   uv run python -m tsp
   ```

## Development

To install development dependencies:
```bash
uv sync --group dev
```

To run tests:
```bash
uv run pytest
```

