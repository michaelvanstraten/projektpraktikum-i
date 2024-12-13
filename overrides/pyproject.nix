{
  hacks,
  pkgs,
  pythonPackages ? pkgs.python312Packages,
}:
final: prev: {
  pycairo = prev.pycairo.overrideAttrs (old: {
    buildInputs = (old.buildInputs or [ ]) ++ [
      pkgs.pkg-config
      pkgs.cairo
      pkgs.ninja
    ];

    nativeBuildInputs =
      (old.nativeBuildInputs or [ ])
      ++ final.resolveBuildSystem {
        meson-python = [ ];
        ninja = [ ];
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
    from = pythonPackages.manimpango;
    prev = prev.scipy;
  };

  scipy = hacks.nixpkgsPrebuilt {
    from = pythonPackages.scipy;
    prev = prev.scipy;
  };

  projektpraktikum-i = prev.projektpraktikum-i.overrideAttrs (old: {
    passthru = old.passthru // {
      tests =
        let
          virtualenv = final.mkVirtualEnv "projektpraktikum-i-pytest-env" {
            projektpraktikum-i = [ "test" ];
          };
        in
        (old.tests or { })
        // {
          pytest = pkgs.stdenv.mkDerivation {
            name = "${final.projektpraktikum-i.name}-pytest";
            inherit (final.projektpraktikum-i) src;

            nativeBuildInputs = [
              virtualenv
            ];

            dontConfigure = true;

            buildPhase = ''
              runHook preBuild
              pytest --cov tests --cov-report html
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mv htmlcov $out
              runHook postInstall
            '';
          };
        };
    };
  });
}
