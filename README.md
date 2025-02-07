# Projektpraktikum I (WiSe2024)

## Overview

This repository contains resources and tools for completing the Projektpraktikum
I assignments. It includes code modules, experiment scripts, and LaTeX documents
for generating reports and presentations.

## Getting Started

1. **Clone the Repository**

   Begin by cloning the repository:

   ```sh
   git clone https://github.com/michaelvanstraten/projektpraktikum-i
   cd projektpraktikum-i
   ```

2. **Dependency Management Options**

   Choose one of the following methods for managing dependencies:

   - [uv (Python-based)](https://docs.astral.sh/uv/)
   - [Nix (System-wide reproducible builds)](https://nixos.org/)

   Follow the setup instructions for your chosen method below.

## Setup with uv

[uv](https://docs.astral.sh/uv/) is a Python dependency management tool. To use
`uv`, ensure that it is installed and available in your system's PATH. If
needed, you can install it without root access:

```sh
# Install uv with restricted permissions (no root access required)
pip install uv --break-system-packages

# Update PATH to include local binaries
export PATH="$HOME/.local/bin:$PATH"
```

### Running Scripts

To run scripts in this repository, first activate a virtual environment with all
dependencies:

```sh
uv venv && source .venv/bin/activate
```

Most scripts in this repository are designed to be executed as command-line
interfaces. To view the available options for a script, append `--help` to the
command. For example, to display CLI options for the derivative approximation
experiments, run:

[!NOTE]

> The following examples assume that you are in the root directory of the
> repository and have activated the virtual environment.

```sh
python src/projektpraktikum_i/derivative_approximation/experiments.py --help
```

Or use the module invocation syntax:

```sh
python -m projektpraktikum_i.derivative_approximation.experiments --help
```

### Discretization Experiments

The discretization experiments module contains several scripts for running
experiments and generating plots. To run an experiment script, use the following
syntax:

```sh
python src/projektpraktikum_i/discretization/<script_name>.py <command> [options]
```

Or use the module invocation syntax:

```sh
python -m projektpraktikum_i.discretization.<script_name> <command> [options]
```

For example, to generate an error plot for the discretization experiments, run:

```sh
python src/projektpraktikum_i/discretization/experiments.py plot-error \
    --n-start 1 \
    --n-stop 7 \
    --n-num 20 \
    --eps 10e-6 \
    --omega 1.5 \
    --save-to error_plot.pdf
```

## Setup with Nix

[Nix](https://nixos.org/) enables reproducible builds and consistent development
environments. To get started,
[install Nix](https://nixos.org/download/#download-nix).

1. **Enter a Nix Shell**: Use the Nix flake configuration in this repository to
   enter a development shell:

   ```sh
   nix develop
   ```

2. **Run Scripts**: Once in the Nix environment, follow the same steps outlined
   in the uv setup section to execute scripts directly.

## Building Documents with Nix

This repository includes LaTeX documents for reports and presentations. To
compile a document with Nix:

1. Ensure Nix is installed as described above.
2. Run the following command to build a specific document (e.g., the derivative
   approximation handout):

   ```sh
   nix build github:michaelvanstraten/projektpraktikum-i#derivative_approximation/handout
   ```

   This command compiles the LaTeX file located at
   [`tex/derivative_approximation/Handout.tex`](tex/derivative_approximation/Handout.tex)
   using the configurations specified in the Nix setup.
