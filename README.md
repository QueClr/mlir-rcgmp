# Reference-counted GMP integers for MLIR

`mlir-rcgmp` provides the `rcgmp` dialect for reference-counted,
arbitrary-precision arithmetic backed by GMP.

The initial scaffold declares two integer types:

- `!rcgmp.mpz`: nonatomic reference counting.
- `!rcgmp.mpz<atomic>`: atomic reference counting.

Arithmetic operations, ownership operations, and lowering are not implemented
yet. The planned runtime uses explicit ownership, inline small integers, and
GMP's low-level `mpn` API over limbs stored alongside the reference-count header.

## Build

The Nix flake supplies LLVM/MLIR 23, a C/C++ toolchain, GMP, CMake, Ninja,
and the test tools:

```sh
nix develop
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
cmake --build build --target check-rcgmp
```

`nix build` builds, tests, and installs the dialect library, headers, and
`rcgmp-opt`. `nix flake check` runs the package check for the current system.

Outside Nix, install LLVM/MLIR 23 development packages, GMP 6.2 or newer,
CMake 3.28 or newer, Ninja, Python with `lit`, and LLVM's `FileCheck`. Set
`LLVM_DIR` and `MLIR_DIR` to their CMake package directories if needed.
Use `-DGMP_ROOT=/path/to/gmp` for a custom GMP prefix, or set
`GMP_INCLUDE_DIR` and `GMP_LIBRARY` separately for split installations.

## Structure

- `include/RCGMP/IR`, `lib/IR`: TableGen declarations and dialect implementation.
- `tool/rcgmp-opt`: MLIR parser and optimization driver.
- `runtime`: C/C++ runtime dependency target, `RCGMP::Runtime`.
- `cmake/FindGMP.cmake`: GMP discovery and the imported `GMP::GMP` target.
- `tests`: type syntax diagnostics and executable GMP linkage checks.

The runtime dependency target links `libgmp` through `GMP::GMP`. It does not
require the separate `gmpxx` C++ wrapper library. The dialect library itself
does not depend on GMP.

## License

Licensed under either Apache-2.0 with LLVM Exceptions or MIT, at your option.
GMP is a separately linked dependency distributed under its own license.
