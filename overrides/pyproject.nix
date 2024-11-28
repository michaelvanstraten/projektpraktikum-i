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

  scipy = hacks.nixpkgsPrebuilt {
    from = pythonPackages.scipy;
    prev = prev.scipy;
  };
}
