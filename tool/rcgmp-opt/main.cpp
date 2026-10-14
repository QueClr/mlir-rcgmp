#include "RCGMP/Conversion/ConvertRCGMPToLLVM.h"
#include "RCGMP/IR/RCGMPDialect.h"

#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::rcgmp::RCGMPDialect, mlir::func::FuncDialect,
                  mlir::cf::ControlFlowDialect, mlir::DLTIDialect>();
  mlir::rcgmp::registerConvertRCGMPToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerPass([] { return mlir::createConvertToLLVMPass(); });
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "rcgmp optimization driver\n", registry));
}
