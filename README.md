# Projektpraktikum I (WiSe2024)

## Overview

This repository provides resources and tools for working on Projektpraktikum I
assignments. It includes code modules, experiment scripts, and LaTeX documents
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

   Follow the setup instructions for your preferred method below.

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

To run scripts in this repository, activate a virtual environment with all
dependencies:

```sh
uv venv && source .venv/bin/activate
```

To view the available command-line interface (CLI) options for each module,
append `--help` to the command:

```sh
# Display CLI options for the experiments module
derivative-approximation --help
```

The above command runs the
[`experiments.py`](./src/projektpraktikum_i/derivative_approximation/experiments.py)
file located in this repository.

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
   in the `uv` setup section to execute scripts.

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
