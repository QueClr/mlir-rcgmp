# Reference-counted GMP integers for MLIR

`mlir-rcgmp` provides the `rcgmp` dialect for reference-counted,
arbitrary-precision arithmetic backed by GMP.

The dialect provides two integer types:

- `!rcgmp.mpz`: nonatomic reference counting.
- `!rcgmp.mpz<atomic>`: atomic reference counting.

Both types lower to `!llvm.ptr` in address space zero. Their size, ABI alignment,
and preferred alignment follow the pointer data layout. Reference-counting mode
is a distinction in the source type and does not change the handle's ABI.

Ownership is explicit: `rcgmp.from_int` creates an owned reference,
`rcgmp.retain` borrows a value and returns an additional owned reference, and
`rcgmp.release` consumes one reference. Retains and releases of small integers
are no-ops. Arithmetic operations are added separately.

```mlir
%integer = rcgmp.from_int %input : i64 to !rcgmp.mpz
%alias = rcgmp.retain %integer : !rcgmp.mpz
rcgmp.release %integer : !rcgmp.mpz
rcgmp.release %alias : !rcgmp.mpz
```

`from_int` accepts any signless integer width. It interprets the bits as signed
two's complement by default; `rcgmp.from_int %x unsigned : i64 to !rcgmp.mpz`
uses unsigned interpretation. Atomicity must remain consistent for all aliases.
Ownership is a caller contract; the dialect does not yet verify linear use.

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

- `include/RCGMP/IR/RCGMPOps.td`: public operations and ownership contracts.
- `include/RCGMP/IR/RCGMPBridgeOps.td`: low-level `rcgmp.bridge.mpz.*` operations,
  with `Mpz*Op` C++ classes in `mlir::rcgmp::bridge`.
- `lib/Conversion/ConvertRCGMPToSCF.cpp`: public operations to SCF and bridges.
- `lib/Conversion/ConvertRCGMPBridgeToLLVM.cpp`: bridges to LLVM instructions
  and runtime calls.
- `lib/Conversion/ConvertRCGMPToLLVM.cpp`: type conversion and registration.
- `runtime`: C runtime, linked as `RCGMP::Runtime`.
- `tests`: IR checks and native tests compiled through the complete lowering.

## Lowering and representation

```sh
rcgmp-opt input.mlir -convert-rcgmp-to-scf -convert-scf-to-cf \
  -convert-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir
```

The SCF stage keeps `mpz` types and makes tag checks, promotion, and reference
count decisions explicit. Bridge operations have preconditions documented in
TableGen; they are implementation primitives, not ownership-safe public APIs.
The LLVM stage converts handles to pointers, inlines tagging and reference
counts, and calls the runtime for allocation and deallocation. Embedders can
register `registerConvertRCGMPToLLVMInterface` with the generic LLVM conversion
pass or populate type and bridge conversions separately.

A handle encodes small integers as `4*n + 1`. With a W-bit pointer, the immediate
range is `[-2^(W-3), 2^(W-3)-1]`. Tag arithmetic uses the pointer width from the
data layout, independently of `index`. Heap handles point directly to one
allocation containing a reference count, signed limb count, capacity, and
trailing GMP limbs. There is no separate `mpz_t` or limb allocation. The runtime
requires GMP without nail bits and limbs matching the target pointer width.
Generated code and the runtime must use matching target ABIs.

Nonatomic reference counts use ordinary loads/stores. Atomic retains use relaxed
ordering, releases use release ordering, and final destruction performs an
acquire fence. Reference-count overflow aborts. Atomic counting does not permit
retaining a borrowed value after its last owning reference has been released.

## Custom allocation

Every runtime allocation and deallocation uses this C ABI, declared in
`RCGMP/Runtime/Allocation.h`:

```c
void *rcgmp_allocate(size_t size, size_t alignment);
void rcgmp_dealloc(void *pointer, size_t size, size_t alignment);
```

The default implementation uses `malloc` and `free`. To supply an allocator,
build with `-DRCGMP_RUNTIME_DEFAULT_ALLOCATOR=OFF` and link definitions of both
functions. Deallocation receives the original allocation size and alignment.
Allocation must return suitably aligned storage or null; the runtime aborts on
allocation failure. Hooks must support concurrent calls when atomic integers
are used across threads. No allocator pointer is stored in each integer.

`RCGMP::Runtime` links GMP through `GMP::GMP`; the dialect library does not
link GMP. The hooks manage rcgmp objects only and do not change GMP's global
allocation functions.

## License

Licensed under either Apache-2.0 with LLVM Exceptions or MIT, at your option.
GMP is a separately linked dependency distributed under its own license.
