{
  lib,
  rustPlatform,
  openssl,
  pkg-config,
}: let
  toml = (lib.importTOML ./Cargo.toml).package;
in
  rustPlatform.buildRustPackage {
    pname = "furnace";
    inherit (toml) version;

    nativeBuildInputs = [
      openssl
      pkg-config
    ];

    PKG_CONFIG_PATH = "${openssl.dev}/lib/pkgconfig";

    src = lib.fileset.toSource {
      root = ./.;
      fileset = lib.fileset.intersection (lib.fileset.fromSource (lib.sources.cleanSource ./.)) (
        lib.fileset.unions [
          ./Cargo.toml
          ./Cargo.lock
          ./src
          ./tests
          ./benches

          # For tests and benchmark
          ./test_model.burn
        ]
      );
    };

    cargoLock.lockFile = ./Cargo.lock;

    meta = {
      inherit (toml) homepage description;
      license = lib.licenses.mit;
      mainProgram = "furnace";
    };
  }
