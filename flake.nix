{
  description = "Hello world flake using uv2nix";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";

    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

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
      latix,
      git-hooks,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        inherit (latix.lib.${system}) buildLatexmkProject;
      in
      {
        packages = {
          "derivative_approximation/handout" = buildLatexmkProject {
            name = "derivative_approximation-handout";
            filename = "Handout.tex";
            extraOptions = [ "" ];
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
              nixfmt-rfc-style.enable = true;
              latexindent = {
                enable = true;
                settings = {
                  flags = "--local --silent --modifylinebreak --overwriteIfDifferent";
                };
              };
              chktex.enable = true;
            };
          };
        };

        # Development shell with necessary tools
        devShells =
          let
            pre-commit-check = self.checks.${system}.pre-commit-check;
          in
          {
            default = pkgs.mkShell {
              packages = pre-commit-check.enabledPackages ++ [
                self.formatter.${system}
                # Add other dependencies here
              ];
              inherit (pre-commit-check) shellHook;
            };
          };
      }
    );
