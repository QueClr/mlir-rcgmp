{
  description = "Reference-counted GMP integers for MLIR";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      environment =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          llvm = pkgs.llvmPackages_23;
          # FuzzTest and the dependencies it builds, at the versions it pins.
          fuzztestSources = {
            FUZZTEST = pkgs.fetchFromGitHub {
              owner = "google";
              repo = "fuzztest";
              rev = "39e5d587af63ba6209e0958a3141460ce003409c";
              hash = "sha256-KJcX4kckRRRo0pVff9UXqBK8Tobj7m0K+KQUtp+/tVE=";
            };
            ABSEIL-CPP = pkgs.fetchFromGitHub {
              owner = "abseil";
              repo = "abseil-cpp";
              tag = "20260817.0";
              hash = "sha256-CEtLO4il9/jk+bFDDV0rXeX1OkirA0u6nxrSiWq0NPM=";
            };
            RE2 = pkgs.fetchFromGitHub {
              owner = "google";
              repo = "re2";
              tag = "2025-11-05";
              hash = "sha256-0J1HVk+eR7VN0ymucW9dNlT36j16XIfCzcs1EVyEIEU=";
            };
            GOOGLETEST = pkgs.fetchFromGitHub {
              owner = "google";
              repo = "googletest";
              tag = "v1.17.0";
              hash = "sha256-HIHMxAUR4bjmFLoltJeIAVSulVQ6kVuIT2Ku+lwAx/4=";
            };
            ANTLR_CPP = pkgs.fetchzip {
              url = "https://www.antlr.org/download/antlr4-cpp-runtime-4.13.2-source.zip";
              stripRoot = false;
              hash = "sha256-kmABwvJ5c3Nss3tpk96yMdKhtYJ2Ep+Y2hVTbuJ1ES8=";
            };
          };
        in
        {
          inherit pkgs llvm fuzztestSources;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            llvm.llvm
            llvm.tblgen
            # clang-scan-deps finds the module dependencies of C++ sources.
            llvm.clang-tools
            (pkgs.python3.withPackages (ps: [ ps.lit ]))
          ];
          buildInputs = [
            llvm.llvm.dev
            llvm.mlir.dev
            llvm.mlir
            pkgs.gmp
          ];
          LLVM_DIR = "${llvm.llvm.dev}/lib/cmake/llvm";
          MLIR_DIR = "${llvm.mlir.dev}/lib/cmake/mlir";
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          env = environment system;
        in
        {
          default = env.llvm.stdenv.mkDerivation {
            pname = "mlir-rcgmp";
            version = "0.1.0";
            src = self;
            inherit (env)
              nativeBuildInputs
              buildInputs
              LLVM_DIR
              MLIR_DIR
              ;
            cmakeFlags = [
              "-DRCGMP_ENABLE_TESTS=ON"
            ]
            ++ nixpkgs.lib.mapAttrsToList (
              name: source: "-DFETCHCONTENT_SOURCE_DIR_${name}=${source}"
            ) env.fuzztestSources;
            doCheck = true;
            checkPhase = ''
              runHook preCheck
              ctest --output-on-failure
              runHook postCheck
            '';
          };
        }
      );

      devShells = forAllSystems (
        system:
        let
          env = environment system;
        in
        {
          default = (env.pkgs.mkShell.override { stdenv = env.llvm.stdenv; }) {
            inherit (env)
              nativeBuildInputs
              buildInputs
              LLVM_DIR
              MLIR_DIR
              ;
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          env = environment system;
          inherit (nixpkgs) lib;
        in
        {
          build = self.packages.${system}.default;
          format =
            env.pkgs.runCommand "mlir-rcgmp-format"
              {
                nativeBuildInputs = [
                  env.llvm.clang-tools
                  env.pkgs.nixfmt
                ];
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./.clang-format
                    ./flake.nix
                    (lib.fileset.fileFilter (
                      file:
                      lib.any file.hasExt [
                        "h"
                        "cpp"
                        "cppm"
                      ]
                    ) ./.)
                  ];
                };
              }
              ''
                cd "$src"
                find . -type f \( -name '*.h' -o -name '*.cpp' -o -name '*.cppm' \) -print0 \
                  | xargs -0 clang-format --dry-run --Werror
                nixfmt --check flake.nix
                touch "$out"
              '';
        }
      );
      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
