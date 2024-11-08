set(LLVM_TARGET_DEFINITIONS MemrefToLLVMPtrPass.td)
mlir_tablegen(MemrefToLLVMPtrPass.h.inc -gen-pass-decls --name MemrefToLLVMPtr)
add_public_tablegen_target(MemrefToLLVMPtrPassIncGen)
