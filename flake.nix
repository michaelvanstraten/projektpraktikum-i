{
  description = "Hello world flake using uv2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";

    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:adisbladis/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
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
      latix,
      git-hooks,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        inherit (latix.lib.${system}) buildLatexmkProject;

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        hacks = pkgs.callPackage pyproject-nix.build.hacks { };

        pyprojectOverrides = final: prev: {
          pycairo = prev.pycairo.overrideAttrs (old: {
            buildInputs = (old.buildInputs or [ ]) ++ [
              pkgs.pkg-config
              pkgs.cairo
            ];
            nativeBuildInputs =
              (old.nativeBuildInputs or [ ])
              ++ final.resolveBuildSystem {
                meson-python = [ ];
              };
          });
          srt = prev.srt.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [ ])
              ++ final.resolveBuildSystem {
                setuptools = [ ];
              };
          });
          manimpango = hacks.nixpkgsPrebuilt {
            from = pkgs.python312Packages.manimpango;
            prev = prev.manimpango;
          };
          scipy = hacks.nixpkgsPrebuilt {
            from = pkgs.python312Packages.scipy;
            prev = prev.scipy;
          };
        };

        python = pkgs.python312;

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (pkgs.lib.composeExtensions overlay pyprojectOverrides);

        virtualenv = pythonSet.mkVirtualEnv "projektpraktikum-i-dev-env" workspace.deps.all;
      in
      {
        packages = {
          "derivative_approximation/handout" = buildLatexmkProject {
            name = "derivative_approximation-handout";
            filename = "Handout.tex";
            SOURCE_DATE_EPOCH = "${toString self.lastModified}";
            src =
              with pkgs.lib.fileset;
              toSource {
                root = ./tex/derivative_approximation;
                fileset = unions [
                  ./tex/derivative_approximation/Handout.tex
                  ./tex/derivative_approximation/handout.bib
                ];
              };
            buildInputs = [ pkgs.texliveFull ];
          };
        };

        # Formatter for this flake
        formatter = pkgs.nixfmt-rfc-style;

        checks = {
          pre-commit-check = git-hooks.lib.${system}.run {
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
                enable = true;
                settings = {
                  binPath = "${virtualenv}/bin/python -m pylint";
                };
              };
            };
          };
        };

        # Development shell with necessary tools
        devShells =
          let
            pre-commit-check = self.checks.${system}.pre-commit-check;
            pre-commit-check-shell-hook = pre-commit-check.shellHook;

            editableOverlay = workspace.mkEditablePyprojectOverlay {
              root = "$REPO_ROOT";
            };

            editablePythonSet = pythonSet.overrideScope editableOverlay;

            virtualenv = editablePythonSet.mkVirtualEnv "projektpraktikum-i-env" {
              projektpraktikum-i = [ ];
            };
          in
          {
            default = pkgs.mkShell {
              packages = pre-commit-check.enabledPackages ++ [
                self.formatter.${system}
                virtualenv
              ];

              shellHook =
                pre-commit-check-shell-hook
                # bash
                + ''
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
