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
        in
        {
          inherit pkgs llvm;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.pkg-config
            llvm.llvm
            llvm.tblgen
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
            cmakeFlags = [ "-DRCGMP_ENABLE_TESTS=ON" ];
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
            packages = [ env.llvm.clang-tools ];
          };
        }
      );

      checks = forAllSystems (
        system:
        let
          env = environment system;
          inherit (nixpkgs) lib;
          sanitizers = "-fsanitize=address,undefined";
          sanitizerFlags = "${sanitizers} -fno-sanitize-recover=all -fno-omit-frame-pointer";
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
                        "c"
                        "h"
                        "cpp"
                      ]
                    ) ./.)
                  ];
                };
              }
              ''
                cd "$src"
                find . -type f \( -name '*.c' -o -name '*.h' -o -name '*.cpp' \) -print0 \
                  | xargs -0 clang-format --dry-run --Werror
                nixfmt --check flake.nix
                touch "$out"
              '';
        }
        # LLVM/MLIR stay uninstrumented, which ASan and UBSan tolerate.
        // lib.optionalAttrs (lib.hasSuffix "-linux" system) {
          asan-ubsan = self.packages.${system}.default.overrideAttrs {
            pname = "mlir-rcgmp-asan-ubsan";
            cmakeBuildType = "Debug";
            # CMake initializes its compiler and linker flags from these.
            CFLAGS = sanitizerFlags;
            CXXFLAGS = sanitizerFlags;
            LDFLAGS = sanitizers;
            # fortify conflicts with ASan; strictoverflow hides signed overflow.
            hardeningDisable = [
              "fortify"
              "strictoverflow"
            ];
            ASAN_OPTIONS = "detect_leaks=1";
            UBSAN_OPTIONS = "print_stacktrace=1";
            installPhase = "touch $out";
            dontFixup = true;
          };
        }
      );
      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
