{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    forAllSystems = function:
      nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed (
        system: function nixpkgs.legacyPackages.${system}
      );
  in {
    packages = forAllSystems (pkgs: {
      furnace = pkgs.callPackage ./default.nix {};
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.furnace;
    });

    devShells = forAllSystems (pkgs: {
      default = pkgs.mkShell {
        inputsFrom = [self.packages.${pkgs.stdenv.hostPlatform.system}.furnace];

        shellHook = ''
          export RUST_SRC_PATH=${pkgs.rustPlatform.rustLibSrc}
        '';

        buildInputs = with pkgs; [
          clippy
          rustfmt
          rust-analyzer
        ];
      };
    });

    overlays.default = final: _: {example = final.callPackage ./default.nix {};};
  };
}
