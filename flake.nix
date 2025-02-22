{
  description = "Projektpraktikum I (WiSe2024)";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";

    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-unstable";

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

        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = builtins.toString (
            with lib.fileset;
            toSource {
              root = ./.;
              fileset = unions [
                ./src
                ./tests
                ./pyproject.toml
                ./uv.lock
                ./README.md
              ];
            }
          );
        };

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
            filename = "handout.tex";
            SOURCE_DATE_EPOCH = toString self.lastModified;
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

          "discretization/figures" =
            pkgs.runCommand "discretization-figures"
              {
                buildInputs = [ self.packages.${system}.default ];
              }
              ''
                mkdir -p $out/figures
                export MPLBACKEND=PDF
                discretization-experiements plot-solutions -n 4 --save-to $out/figures/solutions-for-n-equal-4.pdf
                discretization-experiements plot-solutions -n 11 --save-to $out/figures/solutions-for-n-equal-11.pdf
                discretization-experiements plot-solutions -n 128 --save-to $out/figures/solutions-for-n-equal-128.pdf
                discretization-experiements plot-difference -n 4 --save-to $out/figures/difference-for-n-equal-4.pdf
                discretization-experiements plot-difference -n 11 --save-to $out/figures/difference-for-n-equal-11.pdf
                discretization-experiements plot-difference -n 128 --save-to $out/figures/difference-for-n-equal-128.pdf
                discretization-experiements plot-error --end 6 --num-points 20 --save-to $out/figures/error.pdf
                discretization-experiements plot-sparsity --end 6 --num-points 20 --save-to $out/figures/sparsity.pdf
                discretization-experiements plot-sparsity-lu --end 6 --num-points 20 --save-to $out/figures/sparsity-lu.pdf
                discretization-experiements plot-theoretical-memory-usage --end 6 --num-points 20 --save-to $out/figures/theoretical-memory-usage.pdf
              '';

          "discretization/handout" = buildLatexmkProject {
            name = "discretization-handout";
            filename = "handout.tex";
            SOURCE_DATE_EPOCH = toString self.lastModified;
            extraOptions = [ "--shell-escape" ];
            src = pkgs.buildEnv {
              name = "discretization-handout-source";
              paths = [
                ./tex/discretization
                self.packages.${system}."discretization/figures"
              ];
            };
            buildInputs = [
              pkgs.biber
              (pkgs.texlive.combine {
                inherit (pkgs.texlive)
                  scheme-basic

                  csquotes
                  babel-german
                  biblatex
                  float
                  koma-script
                  mathtools
                  ;
              })
            ];
          };

          "discretization/presentation" = buildLatexmkProject {
            name = "discretization-presentation";
            filename = "presentation.tex";
            SOURCE_DATE_EPOCH = toString self.lastModified;
            extraOptions = [ "--shell-escape" ];
            src = pkgs.buildEnv {
              name = "discretization-handout-source";
              paths = [
                ./tex/discretization
                self.packages.${system}."discretization/figures"
              ];
            };
            buildInputs = [
              pkgs.biber
              (pkgs.texlive.combine {
                inherit (pkgs.texlive)
                  scheme-basic

                  csquotes
                  babel-german
                  biblatex
                  float
                  koma-script
                  mathtools
                  beamer
                  extsizes
                  caption
                  ;
              })
            ];
          };

          "discretization/figures-report" =
            pkgs.runCommand "discretization-figures-report"
              {
                buildInputs = [ self.packages.${system}.default ];
              }
              ''
                mkdir -p $out/figures
                export MPLBACKEND=PDF
                cd $out/figures
                ${./tex/discretization/figures/generate.sh}
              '';

          "discretization/report" = buildLatexmkProject {
            name = "discretization-report";
            filename = "report.tex";
            SOURCE_DATE_EPOCH = toString self.lastModified;
            extraOptions = [ "--shell-escape" ];
            src = pkgs.buildEnv {
              name = "discretization-report-source";
              paths = [
                ./tex/discretization
                self.packages.${system}."discretization/figures-report"
              ];
            };
            rcFile = pkgs.writeText "latexmkrc" ''
              add_cus_dep('pytxcode', 'tex', 0, 'pythontex');
              sub pythontex { return system("pythontex \"$_[0]\""); }
            '';
            buildInputs = [
              self.packages.${system}.default
              pkgs.biber
              (pkgs.texlive.combine {
                inherit (pkgs.texlive)
                  scheme-basic

                  babel-german
                  beamer
                  biblatex
                  caption
                  csquotes
                  extsizes
                  float
                  ifoddpage
                  koma-script
                  luacode
                  luatexbase
                  luatodonotes
                  mathtools
                  soul
                  soulpos
                  xkeyval
                  xstring
                  zref

                  minted
                  upquote
                  lineno
                  ;
              })
            ];
          };

          docs = pkgs.stdenvNoCC.mkDerivation {
            name = "projektpraktikum-i-docs";
            src =
              with lib.fileset;
              toSource {
                root = ./.;
                fileset = unions [
                  ./src
                  ./docs
                  ./README.md
                ];
              };
            SOURCE_DATE_EPOCH = toString self.lastModified;
            buildInputs = [
              (pythonSet.mkVirtualEnv "projektpraktikum-i-env" { projektpraktikum-i = [ "docs" ]; })
              (pkgs.texlive.combine {
                inherit (pkgs.texlive)
                  scheme-basic

                  booktabs
                  capt-of
                  cm-super
                  cmap
                  etoolbox
                  fancyvrb
                  float
                  fncychap
                  framed
                  latexmk
                  needspace
                  parskip
                  tabulary
                  tex-gyre
                  titlesec
                  upquote
                  varwidth
                  wrapfig
                  xcolor
                  ;
              })
            ];
            buildPhase = ''
              cd docs && make latexpdf
            '';
            installPhase = ''
              mkdir -p $out/docs && install -m644 -D _build/latex/*.pdf $out/docs
            '';
          };
        };

        # Formatter for this flake
        formatter = pkgs.nixfmt-rfc-style;

        checks = {
          inherit (pythonSet.projektpraktikum-i.passthru.tests) pytest;
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
                  enable = true;
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

              editablePythonSet = pythonSet.overrideScope (
                lib.composeManyExtensions [
                  editableOverlay

                  (final: prev: {
                    projektpraktikum-i = prev.projektpraktikum-i.overrideAttrs (old: {
                      nativeBuildInputs =
                        old.nativeBuildInputs
                        ++ final.resolveBuildSystem {
                          editables = [ ];
                        };
                    });
                  })
                ]
              );

              virtualenv = editablePythonSet.mkVirtualEnv "projektpraktikum-i-dev-env" workspace.deps.all;

              preCommitHooks = self.checks.${system}.pre-commit-hooks;

            in
            pkgs.mkShell {
              inputsFrom = [ self.packages.${system}.docs ];
              packages = [
                virtualenv
                pkgs.uv
              ] ++ preCommitHooks.enabledPackages;

              env = {
                # Don't create venv using uv
                UV_NO_SYNC = "1";

                # Force uv to use Python interpreter from venv
                UV_PYTHON = "${virtualenv}/bin/python";

                # Prevent uv from downloading managed Python's
                UV_PYTHON_DOWNLOADS = "never";
              };

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
