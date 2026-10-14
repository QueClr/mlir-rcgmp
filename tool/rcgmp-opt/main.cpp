#include "RCGMP/IR/RCGMPDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::rcgmp::RCGMPDialect, mlir::func::FuncDialect>();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "rcgmp optimization driver\n", registry));
}
