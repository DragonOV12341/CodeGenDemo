
add_mlir_conversion_library(MemrefToLLVMPtr
    LoweringPasses.cc

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Conversion
    ${PROJECT_BINARY_DIR}/include/Conversion

    DEPENDS
    MemrefToLLVMPtrPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass

)
