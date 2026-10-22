#include "RCGMP/Conversion/ConvertRCGMPToLLVM.h"
#include "RCGMP/Conversion/ConvertRCGMPToSCF.h"
#include "RCGMP/IR/RCGMPDialect.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#ifdef RCGMP_INCLUDE_TESTS
namespace mlir::rcgmp::test {
void registerTestPasses();
} // namespace mlir::rcgmp::test
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::rcgmp::RCGMPDialect, mlir::func::FuncDialect,
                  mlir::cf::ControlFlowDialect, mlir::DLTIDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::rcgmp::registerConvertRCGMPToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::registerPass([] { return mlir::rcgmp::createConvertRCGMPToSCFPass(); });
  mlir::registerPass([] { return mlir::createSCFToControlFlowPass(); });
  mlir::registerPass([] { return mlir::createReconcileUnrealizedCastsPass(); });
  mlir::registerPass([] { return mlir::createCanonicalizerPass(); });
  mlir::registerPass([] { return mlir::createCSEPass(); });
  mlir::registerPass([] { return mlir::createConvertToLLVMPass(); });
#ifdef RCGMP_INCLUDE_TESTS
  mlir::rcgmp::test::registerTestPasses();
#endif
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "rcgmp optimization driver\n", registry));
}
