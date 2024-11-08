#ifndef TRITON_CONVERSION_PASSES
#define TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemrefToLLVMPtr: Pass<"convert-memref-to-llvmptr", "mlir::ModuleOp"> {
    let summary = "Convert memref to llvmptr";
    let description = [{

    }];
    let constructor = "createConvertMemrefToLLVMPtrPass()";

    let dependentDialects = ["mlir::memref::MemrefDialect",
                             "mlir::LLVM::LLVMDialect",
                             ];

   let options = [ ];
}

#endif
