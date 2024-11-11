#include "Conversion/LoweringPasses.h"

using namespace mlir;

namespace KernelCodeGen {

// 将scf的parallelOp 转成Gpu的block/threadIdOp表示，func添加grid/block size作为属性
struct SCFParallelToGPULowering : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp outerParallelOp, PatternRewriter &rewriter) const final {
    constexpr gpu::Dimension dims[] = {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    auto &ops = outerParallelOp.getBody()->getOperations();
    if (ops.empty())
      return failure();

    scf::ParallelOp innerParallelOp = nullptr;
    for (Operation &op : ops) {
      innerParallelOp = dyn_cast<scf::ParallelOp>(&op);
      if (innerParallelOp)
        break;
    }
    if (!innerParallelOp)
      return failure();

    auto outerUpperBounds = outerParallelOp.getUpperBound();
    auto innerUpperBounds = innerParallelOp.getUpperBound();

    std::vector<int32_t> blockUpperBounds;
    std::vector<int32_t> threadUpperBounds;

    // 替换外层 parallelOp 为 gpu::BlockIdOp
    Location loc = outerParallelOp.getLoc();
    SmallVector<Value, 3> blockIds;
    for (unsigned i = 0; i < outerParallelOp.getNumLoops(); ++i) {
      auto blockId = rewriter.create<gpu::BlockIdOp>(loc, dims[i]);
      blockIds.push_back(blockId);

      auto constOp = outerUpperBounds[i].getDefiningOp<arith::ConstantOp>();
      auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>();
      blockUpperBounds.push_back(intAttr.getInt());
    }

    // 替换内层 parallelOp 为 gpu::ThreadIdOp
    rewriter.setInsertionPoint(innerParallelOp);
    SmallVector<Value, 3> threadIds;
    for (unsigned i = 0; i < innerParallelOp.getNumLoops(); ++i) {
      auto threadId = rewriter.create<gpu::ThreadIdOp>(loc, dims[i]);
      threadIds.push_back(threadId);

      auto constOp = innerUpperBounds[i].getDefiningOp<arith::ConstantOp>();
      auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>();
      threadUpperBounds.push_back(intAttr.getInt());
    }

    // 将func设置block和thread的上界属性
    auto parentOp = outerParallelOp->getParentOp();
    auto funcOp = llvm::dyn_cast<func::FuncOp>(parentOp);
    if (funcOp == nullptr) {
      llvm::errs() << "The ParentOp of scf::ParallelOp must is FuncOp!\n";
      assert(false);
    }
    funcOp->setAttr("func.grid.dim", rewriter.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(blockUpperBounds)));
    funcOp->setAttr("func.block.dim", rewriter.getDenseI32ArrayAttr(llvm::ArrayRef<int32_t>(threadUpperBounds)));

    // 替换使用外层和内层循环变量的操作
    auto outerInductionVars = outerParallelOp.getInductionVars();
    for (unsigned i = 0; i < outerInductionVars.size(); ++i) {
      outerInductionVars[i].replaceAllUsesWith(blockIds[i]);
    }

    auto innerInductionVars = innerParallelOp.getInductionVars();
    for (unsigned i = 0; i < innerInductionVars.size(); ++i) {
      innerInductionVars[i].replaceAllUsesWith(threadIds[i]);
    }

    // 内层操作移出内层 p  collect op
    SmallVector<Operation *, 4> innerOpsToMove;
    for (Operation &op : innerParallelOp.getBody()->getOperations()) {
      if (!dyn_cast<scf::YieldOp>(op)) {
        innerOpsToMove.push_back(&op);
      }
    }
    // 内层操作移出内层 p 
    Operation *innerTempOp = threadIds.back().getDefiningOp();
    for (Operation *op : innerOpsToMove) {
      op->moveAfter(innerTempOp);
      innerTempOp = op;
    }
    rewriter.eraseOp(innerParallelOp);

    // 外 collect op
    SmallVector<Operation *, 4> outerOpsToMove;
    for (Operation &op : outerParallelOp.getBody()->getOperations()) {
      if (!dyn_cast<scf::YieldOp>(op)) {
        outerOpsToMove.push_back(&op);
      }
    }
    // move
    Operation *outerTempOp = blockIds.back().getDefiningOp();
    for (Operation *op : outerOpsToMove) {
      op->moveAfter(outerTempOp);
      outerTempOp = op;
    }
    rewriter.eraseOp(outerParallelOp);

    return success();
  }
};

struct RemoveScfReduceOpPass : public OpRewritePattern<scf::ReduceOp> {
  using OpRewritePattern<scf::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ReduceOp reduceOp, PatternRewriter &rewriter) const final {
    rewriter.eraseOp(reduceOp);
    return success();
  }
};



// 将 GUP 的IdOp转成 rocdl的IdOp，读取func的attr加到新的IdOp上
template <typename IdOp, typename XOp, typename YOp, typename ZOp>
struct IdOpGPUToROCDLLowering : public OpRewritePattern<IdOp> {
  using OpRewritePattern<IdOp>::OpRewritePattern;

  private:
    StringRef boundsAttrName;

  public:
    explicit IdOpGPUToROCDLLowering(MLIRContext *context) 
            : OpRewritePattern<IdOp>(context), boundsAttrName("") {}

    explicit IdOpGPUToROCDLLowering(MLIRContext *context, StringRef boundsAttrName) 
            : OpRewritePattern<IdOp>(context), boundsAttrName(boundsAttrName) {}

  LogicalResult matchAndRewrite(IdOp idOp, PatternRewriter &rewriter) const final {
    auto loc = idOp->getLoc();
    MLIRContext *context = rewriter.getContext();
    Value newOp;
    switch (idOp.getDimension()) {
    case gpu::Dimension::x:
      newOp = rewriter.create<XOp>(loc, IntegerType::get(context, 64));
      break;
    case gpu::Dimension::y:
      newOp = rewriter.create<YOp>(loc, IntegerType::get(context, 64));
      break;
    case gpu::Dimension::z:
      newOp = rewriter.create<ZOp>(loc, IntegerType::get(context, 64));
      break;
    }

    auto parentOp = idOp->getParentOp();
    auto funcOp = llvm::dyn_cast<func::FuncOp>(parentOp);
    if (!boundsAttrName.empty() && funcOp) {
      if (auto attr = llvm::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr(boundsAttrName))) {
        int32_t maximum = attr[static_cast<uint32_t>(idOp.getDimension())];
        newOp.getDefiningOp()->setAttr("range", rewriter.getDenseI32ArrayAttr({0, maximum}));
      }
    }
    Value indexVal = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), newOp);
    // auto castOp = rewriter.create<arith::BitcastOp>(loc, IndexType::get(context), newOp);
    rewriter.replaceOp(idOp, indexVal);
    return success();
  }
};

// 将gpu barrier转成rocdl的barrier
struct GPUBarrierToROCDLLowering : public OpRewritePattern<gpu::BarrierOp> {
  using OpRewritePattern<gpu::BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::BarrierOp brOp, PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ROCDL::BarrierOp>(brOp);
    return success();
  }
};

// 将上述三个重写加到这个pass中
struct ParallelToROCDLPass : public PassWrapper<ParallelToROCDLPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelToROCDLPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ROCDL::ROCDLDialect>();
  }
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    LLVMTypeConverter typeConverter(&getContext());
    target.addIllegalOp<gpu::BlockIdOp, gpu::ThreadIdOp>();
    target.addLegalDialect<ROCDL::ROCDLDialect, arith::ArithDialect>();

    patterns.add<SCFParallelToGPULowering>(&getContext());
    patterns.add<RemoveScfReduceOpPass>(&getContext());
    // mlir::populateGpuToROCDLConversionPatterns(typeConverter, patterns, gpu::amd::HIP);
    patterns.add<IdOpGPUToROCDLLowering<gpu::BlockIdOp, ROCDL::BlockIdXOp, 
                                       ROCDL::BlockIdYOp, ROCDL::BlockIdZOp>>(&getContext(), StringRef{"func.grid.dim"});
    patterns.add<IdOpGPUToROCDLLowering<gpu::ThreadIdOp, ROCDL::ThreadIdXOp,
                                       ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>>(&getContext(), StringRef{"func.block.dim"});

    patterns.add<GPUBarrierToROCDLLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      return signalPassFailure();
  }
};


// 弃用，自己写了一个从gpu到rocdl的pass，转了idop和BarrierOp
struct ROCDLIdOpModifyPass : public PassWrapper<ROCDLIdOpModifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ROCDLIdOpModifyPass)
  
  void runOnOperation() override {
    auto module = llvm::dyn_cast<ModuleOp>(getOperation());
    for (Operation &op : module.getBody()->getOperations()) {
      auto funcOp = llvm::dyn_cast<func::FuncOp>(&op);
      if (funcOp == nullptr) {
        llvm::errs() << "there is other operations which is not funcOp in the module!\n";
        assert(false);
      }
      auto blockDims = llvm::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr("func.grid.dim"));
      auto threadDims = llvm::dyn_cast<DenseI32ArrayAttr>(funcOp->getAttr("func.block.dim"));

      funcOp.walk([&](Operation *op) {
        if (auto blockIdXOp = llvm::dyn_cast<ROCDL::BlockIdXOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[0]}));
        } else if (auto blockIdYOp = llvm::dyn_cast<ROCDL::BlockIdYOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[1]}));
        } else if (auto blockIdZOp = llvm::dyn_cast<ROCDL::BlockIdZOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, blockDims[2]}));
        } else if (auto threadIdXOp = llvm::dyn_cast<ROCDL::ThreadIdXOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[0]}));
        } else if (auto threadIdYOp = llvm::dyn_cast<ROCDL::ThreadIdYOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[1]}));
        } else if (auto threadIdZOp = llvm::dyn_cast<ROCDL::ThreadIdZOp>(op)) {
          op->setAttr("range", DenseI32ArrayAttr::get(module.getContext(), {0, threadDims[2]}));
        }
      });
    }
  }
};

// 去除多余的unrealized_conversion_cast操作
struct EraseRedundantUnCCastPass : public PassWrapper<EraseRedundantUnCCastPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseRedundantUnCCastPass)
  
  void runOnOperation() override {
    auto module = llvm::dyn_cast<ModuleOp>(getOperation());
    SmallVector<std::pair<Operation*, Operation*>> pairOps;
    SmallVector<Operation*> noChOps;
    module.walk([&](Operation *op){
      if (auto uccOp = llvm::dyn_cast<UnrealizedConversionCastOp>(op)) {
        for (auto &use: uccOp.getResult(0).getUses()) {
          Operation *nextOp = use.getOwner();
          if (isa<UnrealizedConversionCastOp>(nextOp))
            pairOps.push_back(std::make_pair(op, nextOp));
          break;
        }
        if (uccOp.use_empty()) {
          noChOps.push_back(op);
        }
      }
    });
    for (auto pairOp: pairOps) {
      auto firstOp = llvm::dyn_cast<UnrealizedConversionCastOp>(pairOp.first);
      auto secondOp = llvm::dyn_cast<UnrealizedConversionCastOp>(pairOp.second);
      if (firstOp.getOperand(0).getType() == secondOp.getResult(0).getType()) {
        secondOp.getResult(0).replaceAllUsesWith(firstOp.getOperand(0));
      }
      pairOp.second->erase();
      pairOp.first->erase();
    }
    for (auto noChOp: noChOps) {
      // llvm::outs() << *noChOp << "\n";
      noChOp->erase();
    }
  }
};

// 将arith的constantOp（index）转成i64
struct ConvertArithIndexToI64Pass : public PassWrapper<ConvertArithIndexToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertArithIndexToI64Pass)

  void runOnOperation() override {
    auto module = llvm::dyn_cast<ModuleOp>(getOperation());
    module.walk([&](Operation *op) {
      if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        Type constantType = constantOp.getValue().getType();
        if (constantType.isIndex()) {
          auto indexValue = constantOp.getValue().cast<IntegerAttr>().getInt();
          OpBuilder builder(op);
          auto i64Op = builder.create<arith::ConstantOp>(constantOp.getLoc(), builder.getI64IntegerAttr(indexValue));
          auto indexVal = builder.create<arith::IndexCastOp>(i64Op.getLoc(), builder.getIndexType(), i64Op);
          constantOp.getResult().replaceAllUsesWith(indexVal.getResult());
          constantOp.erase();
        }
      }
    });
  }
};

// 弃用
struct ConvertArithConstantIndexToI64 : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constOp, PatternRewriter &rewriter) const final {
    Type constType = constOp.getValue().getType();
    if (constType.isIndex()) {
      auto indexValue = constOp.getValue().cast<IntegerAttr>().getInt();
      auto i64Op = rewriter.create<arith::ConstantOp>(constOp.getLoc(), rewriter.getI64IntegerAttr(indexValue));
      auto indexVal = rewriter.create<arith::IndexCastOp>(i64Op.getLoc(), rewriter.getIndexType(), i64Op);
      rewriter.replaceOp(constOp, indexVal);
      return success();
    } else {
      return failure();
    }
  }
};
// 弃用
struct ConvertIndexToI64Pass : public PassWrapper<ConvertIndexToI64Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertIndexToI64Pass)
  
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    patterns.add<ConvertArithConstantIndexToI64>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      return signalPassFailure();
  }
};

class MyRewriter : public RewriterBase{

};

// amend func args
struct ConvertFuncArgPass : public PassWrapper<ConvertFuncArgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertFuncArgPass)

  void runOnOperation() override {
    auto module = llvm::dyn_cast<ModuleOp>(getOperation());
    
    module.walk([&](Operation *op) {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        auto oldFuncType = funcOp.getFunctionType();
        llvm::SmallVector<mlir::Type, 4> newArgTypes;
        mlir::TypeConverter typeConverter;
        typeConverter.addConversion([&](mlir::MemRefType type) -> mlir::Type {
            return mlir::LLVM::LLVMPointerType::get(type.getContext(),type.getElementType(),type.getMemorySpaceAsInt());
        });
        // 遍历参数类型，将MemRef转换为llvm.ptr类型
        for (mlir::Type argType : oldFuncType.getInputs()) {
            if (auto memrefType = argType.dyn_cast<mlir::MemRefType>()) {
                // 将MemRef类型转换为LLVM指针类型
                newArgTypes.push_back(typeConverter.convertType(memrefType));
            } else {
                // 保留非MemRef类型
                newArgTypes.push_back(argType);
            }
        }
        mlir::OpBuilder builder(funcOp);
        auto newFuncType = mlir::FunctionType::get(op->getContext(), newArgTypes, oldFuncType.getResults());
        auto newFuncOp = builder.create<mlir::func::FuncOp>(funcOp.getLoc(),funcOp.getName(), newFuncType);
        
        funcOp.getBody().cloneInto(&newFuncOp.getFunctionBody(),);
        funcOp.getBody().cloneInto(&newFuncOp.getBody(),newFuncOp.getBody().begin());
        // newFuncOp.getBody().takeBody(funcOp.getBody());
        
        // 更新参数引用
        // mlir::Block &entryBlock = newFuncOp.getBody().front();
        // auto e = entryBlock.getNumArguments();
        // for (size_t i = 0 ; i < e; ++i) {
        //     auto oldArg = funcOp.getArgument(i);
        //     auto newArg = entryBlock.getArgument(i);

        //     if (oldArg.getType().isa<mlir::MemRefType>()) {
        //         // 对于原先的MemRef参数，替换为对应的llvm.ptr类型的参数
        //         oldArg.replaceAllUsesWith(newArg);
        //     }
        // }
        funcOp.erase();
      }
    });

  }
};



// ConvertMemrefToLLVMPtrBase

class MemrefToLLVMPtrTypeConverter 
    : public TypeConverter
{
public:
    MemrefToLLVMPtrTypeConverter(MLIRContext *context){
        addConversion([context](mlir::MemRefType memrefType) -> mlir::LLVM::LLVMPointerType {
            // types with encoding are already in the right format
            // TODO: check for layout encodings more specifically
            auto eleType = memrefType.getElementType();
            unsigned int addrSpace = memrefType.getMemorySpaceAsInt();
            return LLVM::LLVMPointerType::get(eleType,addrSpace);
        });
    }

private:
    MLIRContext *context;
};

class MemrefToLLVMPtrConversionTarget : public ConversionTarget
{

public:
    explicit MemrefToLLVMPtrConversionTarget(
        MLIRContext &ctx,
        MemrefToLLVMPtrTypeConverter &typeConverter) : ConversionTarget(ctx)
    {
        addLegalDialect<mlir::LLVM::LLVMDialect>();
        // addDynamicallyLegalDialect<mlir::memref::MemRefDialect>();
        addLegalOp<mlir::UnrealizedConversionCastOp>();
    }
};

class ConvertMemrefToLLVMPtr : public ::impl::ConvertMemrefToLLVMPtrBase<ConvertMemrefToLLVMPtr> {
public:
  using ConvertMemrefToLLVMPtrBase<ConvertMemrefToLLVMPtr>::ConvertMemrefToLLVMPtrBase;
  
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter
    MemrefToLLVMPtrTypeConverter typeConverter(context);
    MemrefToLLVMPtrConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    mod.walk([](mlir::func::FuncOp funcOp){
        ;
    });

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};


// ===================== create pass interfaces ======================

std::unique_ptr<OperationPass<ModuleOp>> createParallelToROCDLPass() {
  return std::make_unique<ParallelToROCDLPass>();
}

//  弃用，自己写了一个从gpu到rocdl的pass，转了idop和BarrierOp
std::unique_ptr<OperationPass<ModuleOp>> createROCDLIdOpModifyPass() {
  return std::make_unique<ROCDLIdOpModifyPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createEraseRedundantUnCCastPass() {
  return std::make_unique<EraseRedundantUnCCastPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertArithIndexToI64Pass() {
  return std::make_unique<ConvertArithIndexToI64Pass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertMemrefToLLVMPtrPass() {
    return std::make_unique<ConvertMemrefToLLVMPtr>();
}

std::unique_ptr<OperationPass<ModuleOp>> createAmendFuncArgPass() {
    return std::make_unique<ConvertFuncArgPass>();
}

}