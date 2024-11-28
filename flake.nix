{
  description = "Hello world flake using uv2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixpkgs-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    latix = {
      url = "github:michaelvanstraten/latix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      flake-utils,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      latix,
      git-hooks,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;
        pkgs = import nixpkgs { inherit system; };
        inherit (latix.lib.${system}) buildLatexmkProject;

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        python = pkgs.python312;

        hacks = pkgs.callPackage pyproject-nix.build.hacks { };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                (import ./overrides/pyproject.nix { inherit pkgs hacks; })
              ]
            );
      in
      {
        packages = {
          default = pythonSet.mkVirtualEnv "projektpraktikum-i-env" workspace.deps.default;

          "derivative_approximation/handout" = buildLatexmkProject {
            name = "derivative_approximation-handout";
            filename = "Handout.tex";
            SOURCE_DATE_EPOCH = "${toString self.lastModified}";
            XDG_CACHE_HOME = "$(mktemp -d)";
            XDG_CONFIG_HOME = "$(mktemp -d)";
            XDG_DATA_HOME = "$(mktemp -d)";
            extraOptions = [ "--shell-escape" ];
            src = ./tex/derivative_approximation;
            buildInputs = [
              pkgs.texliveFull
              pkgs.inkscape
            ];
          };
        };

        # Formatter for this flake
        formatter = pkgs.nixfmt-rfc-style;

        checks = {
          pre-commit-hooks =
            let
              devVirtualEnv = pythonSet.mkVirtualEnv "projektpraktikum-i-dev-env" workspace.deps.all;
            in
            git-hooks.lib.${system}.run {
              src = ./.;
              hooks = {
                # Nix
                nixfmt-rfc-style.enable = true;
                # LaTeX
                latexindent = {
                  enable = true;
                  settings = {
                    flags = "--local --silent --modifylinebreak --overwriteIfDifferent";
                  };
                };
                chktex.enable = true;
                # Python
                pylint = {
                  enable = false;
                  settings = {
                    binPath = "${devVirtualEnv}/bin/python -m pylint";
                  };
                };
                # Markdown and YAML
                prettier = {
                  enable = true;
                  settings = {
                    prose-wrap = "always";
                  };
                };
              };
            };
        };

        devShells = {
          default =
            let
              editableOverlay = workspace.mkEditablePyprojectOverlay {
                root = "$REPO_ROOT";
              };

              editablePythonSet = pythonSet.overrideScope editableOverlay;

              virtualEnv = editablePythonSet.mkVirtualEnv "projektpraktikum-i-venv-dev" workspace.deps.all;

              preCommitHooks = self.checks.${system}.pre-commit-hooks;
            in
            pkgs.mkShell {
              packages = [
                virtualEnv
                pkgs.uv
              ] ++ preCommitHooks.enabledPackages;
              shellHook =
                preCommitHooks.shellHook
                +
                  # bash
                  ''
                    # Undo dependency propagation by nixpkgs.
                    unset PYTHONPATH
                    # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                    export REPO_ROOT=$(git rev-parse --show-toplevel)
                  '';
            };
        };
      }
    );
}
