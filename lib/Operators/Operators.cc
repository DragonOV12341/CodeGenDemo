#include "Operators/Operators.h"

namespace KernelCodeGen {

std::string randName(int length) {
  std::string characters{"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(0, characters.size() - 1);
  std::string randomString;
  for (int i = 0; i < length; ++i) {
      randomString += characters[distribution(generator)];
  }
  return randomString;
}

mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype) {
  if(dtype == "float32") return builder.getF32Type();
  if(dtype == "float64") return builder.getF64Type();
  if(dtype == "float16") return builder.getF16Type();
  if(dtype == "int64") return builder.getIntegerType(64);
  if(dtype == "int32") return builder.getIntegerType(32);
  if(dtype == "int16") return builder.getIntegerType(16);
  if(dtype == "index") return builder.getIndexType();
  if(dtype == "bool") return builder.getIntegerType(1);
  return nullptr;
}

mlir::func::FuncOp buildFunction(mlir::ModuleOp module, mlir::OpBuilder& builder, const std::string& funcName, 
                                  const std::string& OpName, const std::vector<mlir::Type>& inputsTypes) {
  bool break_ = false;
  module.walk<mlir::WalkOrder::PreOrder>([&](mlir::func::FuncOp func) {
    auto otherName = func.getSymName();
    if (otherName == funcName) {
      break_ = true;
    }
  });
  if (break_) return nullptr;

  builder.setInsertionPointToStart(module.getBody());
  llvm::ArrayRef<mlir::Type> inputsTypesArray(inputsTypes);
  auto functionType = builder.getFunctionType(mlir::TypeRange(inputsTypesArray), mlir::TypeRange({}));
  auto funcOp = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), llvm::StringRef(funcName), functionType);
  
  auto& region = funcOp->getRegion(0);
  if (!region.hasOneBlock()) {
    region.emplaceBlock();
  }
  auto& body =  funcOp.front(); //? region.front()  : ;
  int nums = static_cast<int>(inputsTypes.size());
  for (int i = 0; i < nums; i++ ) {
    body.addArguments(inputsTypes[i], builder.getUnknownLoc());
  }
  
  funcOp->setAttr(std::string("func.state"), builder.getStringAttr("cpu"));
  funcOp->setAttr(std::string("func.op.name"), builder.getStringAttr(OpName));
  funcOp->setAttr(std::string("llvm.bareptr"), builder.getStringAttr("true"));
  auto& entryBlock = funcOp.front();
  builder.setInsertionPointToStart(&entryBlock);
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  return funcOp;
}

void Matmul::build(mlir::ModuleOp module, mlir::OpBuilder& builder, std::vector<int64_t> shape, 
                    const std::string& dtype, std::vector<MemorySpace> mss) {
  
  auto ver = verify(builder, shape, dtype);
  if (!ver.first) {
    llvm::errs() << ver.second << "\n";
  } else {
    int64_t m = shape[0];
    int64_t n = shape[1];
    int64_t k = shape[2];
    auto emType = getDType(builder, dtype);
    auto ip = builder.saveInsertionPoint();
    
    auto funcOp = createFunc(module, builder, shape, emType, mss);

    auto& bodyBlock = funcOp.front();
    builder.setInsertionPointToStart(&bodyBlock);
    mlir::ValueRange operands = bodyBlock.getArguments();

    mlir::SmallVector<int64_t, 3> lowerBounds(2, /*Value=*/0);
    mlir::SmallVector<int64_t, 3> steps(2, /*Value=*/1);
    mlir::SmallVector<int64_t, 3> upperBounds({m, n});
    mlir::affine::buildAffineLoopNest(builder, builder.getUnknownLoc(), lowerBounds, upperBounds, steps,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
        auto i = ivs[0];
        auto j = ivs[1];

        auto zero = nestedBuilder.create<mlir::arith::ConstantOp>(nestedBuilder.getUnknownLoc(), nestedBuilder.getFloatAttr(emType, 0));

        auto kLoopBody = [&](mlir::OpBuilder &builder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange iterArgs) {
          mlir::OpBuilder::InsertionGuard nestedGuard(builder);
          auto k = iv;
          auto ld_a = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), /*A*/operands[0], mlir::ValueRange({i, k}));
          auto ld_b = builder.create<mlir::affine::AffineLoadOp>(builder.getUnknownLoc(), /*B*/operands[1], mlir::ValueRange({k, j}));
          auto mul = builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), ld_a, ld_b);
          auto add = builder.create<mlir::arith::AddFOp>(builder.getUnknownLoc(), mul, iterArgs[0]);
          builder.create<mlir::affine::AffineYieldOp>(builder.getUnknownLoc(), add.getResult());
        };
        auto Cij = nestedBuilder.create<mlir::affine::AffineForOp>(nestedBuilder.getUnknownLoc(), 0, k, 1, mlir::ValueRange({zero.getResult()}), kLoopBody);

        nestedBuilder.create<mlir::affine::AffineStoreOp>(nestedBuilder.getUnknownLoc(), Cij.getResult(0), /*C*/operands[2], mlir::ValueRange({i, j}));
      }
    );
    builder.restoreInsertionPoint(ip);
  }
}


std::pair<bool, std::string> Matmul::verify(mlir::OpBuilder& builder, std::vector<int64_t> shape, const std::string& dtype) {
  std::string err{""};
  if (shape.size() > 3) {
    std::string err{"Shape size must is 3."};
    return std::make_pair(false, err);
  }

  auto type = getDType(builder, dtype);
  if (type == nullptr) {
    std::string err{"No exist this data type."};
    return std::make_pair(false, err);
  }

  return std::make_pair(true, err);
}

mlir::func::FuncOp Matmul::createFunc(mlir::ModuleOp module, mlir::OpBuilder& builder, std::vector<int64_t> shape, 
                                      mlir::Type dtype, std::vector<MemorySpace> mss) {
  int64_t m = shape[0];
  int64_t n = shape[1];
  int64_t k = shape[2];
  auto shape_a = std::vector<int64_t>{m, k};
  auto shape_b = std::vector<int64_t>{k, n};
  auto shape_c = std::vector<int64_t>{m, n};
  auto typeA = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_a), dtype, {}, static_cast<int>(mss[0]));
  auto typeB = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_b), dtype, {}, static_cast<int>(mss[1]));
  auto typeC = mlir::MemRefType::get(llvm::ArrayRef<int64_t>(shape_c), dtype, {}, static_cast<int>(mss[2]));
  auto funcName = "Matmul_m" + std::to_string(m) + "n" + std::to_string(n) +  "k" + std::to_string(k) + "_" + randName();

  return buildFunction(module, builder, funcName, "Matmul", {typeA, typeB, typeC});
}

}