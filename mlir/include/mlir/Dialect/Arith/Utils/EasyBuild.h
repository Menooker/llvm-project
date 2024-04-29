//===- EasyBuild.h - Easy Arith IR Builder utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the Arith dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_UTILS_EASYBUILD_H
#define MLIR_DIALECT_ARITH_UTILS_EASYBUILD_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include <cstdint>
#include <memory>
#include <stddef.h>

namespace mlir {
namespace easybuild {

enum class ComputeType {
  Signed,
  Unsigned,
  Float,
};

namespace impl {
struct EasyBuildState {
  OpBuilder &builder;
  Location loc;
  bool u64AsIndex;
  EasyBuildState(OpBuilder &builder, Location loc, bool u64AsIndex)
      : builder{builder}, loc{loc}, u64AsIndex{u64AsIndex} {}
};
struct Unreachable {};

template <std::size_t size>
struct ToFloatType {};

template <>
struct ToFloatType<4> {
  using type = Float32Type;
};
template <>
struct ToFloatType<8> {
  using type = Float64Type;
};

} // namespace impl

struct WrappedValue {
  std::shared_ptr<impl::EasyBuildState> builder;
  Value v;
  ComputeType computeType;

  WrappedValue() = default;
  WrappedValue(const std::shared_ptr<impl::EasyBuildState> &builder, Value v,
               ComputeType computeType)
      : builder{builder}, v{v}, computeType{computeType} {}
  Value get() const { return v; }
  operator Value() const { return v; }
};

struct EasyBuilder {
  std::shared_ptr<impl::EasyBuildState> builder;
  EasyBuilder(OpBuilder &builder, Location loc, bool u64AsIndex = false)
      : builder{
            std::make_shared<impl::EasyBuildState>(builder, loc, u64AsIndex)} {}
  EasyBuilder(const std::shared_ptr<impl::EasyBuildState> &builder)
      : builder{builder} {}
  void setLoc(const Location &l) { builder->loc = l; }
  WrappedValue operator()(Value v, ComputeType ct) const {
    return WrappedValue{builder, v, ct};
  }
  WrappedValue toIndex(uint64_t v) const {
    return WrappedValue{
        builder,
        builder->builder.create<arith::ConstantIndexOp>(builder->loc, v),
        ComputeType::Unsigned};
  }

  template <typename T>
  WrappedValue operator()(T v) const {
    using DT = std::decay_t<T>;
    static_assert(std::is_arithmetic_v<DT>, "Expecting arithmetic types");
    if constexpr (std::is_same_v<DT, uint64_t>) {
      if (builder->u64AsIndex) {
        return toIndex(v);
      }
    }

    if constexpr (std::is_same_v<DT, bool>) {
      return WrappedValue{builder,
                          builder->builder.create<arith::ConstantIntOp>(
                              builder->loc, static_cast<int64_t>(v), 1),
                          ComputeType::Unsigned};
    } else if constexpr (std::is_integral_v<DT>) {
      return WrappedValue{
          builder,
          builder->builder.create<arith::ConstantIntOp>(
              builder->loc, static_cast<int64_t>(v), sizeof(T) * 8),
          std::is_signed_v<DT> ? ComputeType::Signed : ComputeType::Unsigned};
    } else {
      using DType = typename impl::ToFloatType<sizeof(DT)>::type;
      return WrappedValue{builder,
                          builder->builder.create<arith::ConstantFloatOp>(
                              builder->loc, APFloat{v},
                              DType::get(builder->builder.getContext())),
                          ComputeType::Float};
    }
  }
  WrappedValue operator()(const OpFoldResult &v, ComputeType ct) const {
    if (v.is<Value>()) {
      return WrappedValue{builder, v.get<Value>(), ct};
    }
    return operator()(
        static_cast<uint64_t>(v.get<Attribute>().cast<IntegerAttr>().getInt()));
  }
};

template <typename OpSigned, typename OpUnsigned, typename OpFloat>
WrappedValue handleBinary(const WrappedValue &a, const WrappedValue &b) {
  assert(a.builder == b.builder);
  assert(a.computeType == b.computeType);
  switch (a.computeType) {
  case ComputeType::Signed:
    return {a.builder,
            a.builder->builder.create<OpSigned>(a.builder->loc, a.v, b.v),
            a.computeType};
  case ComputeType::Unsigned:
    return {a.builder,
            a.builder->builder.create<OpUnsigned>(a.builder->loc, a.v, b.v),
            a.computeType};
  case ComputeType::Float:
    if constexpr (std::is_same_v<impl::Unreachable, OpFloat>) {
      llvm_unreachable("unreachable handleBinary");
    } else {
      return {a.builder,
              a.builder->builder.create<OpFloat>(a.builder->loc, a.v, b.v),
              a.computeType};
    }
  }
  return WrappedValue{};
}

template <typename OpSigned, typename OpUnsigned, typename OpFloat, typename T2>
WrappedValue handleBinaryConst(const WrappedValue &a, const T2 &b) {
  return handleBinary<OpSigned, OpUnsigned, OpFloat>(a,
                                                     EasyBuilder{a.builder}(b));
}

template <typename OpSigned, typename OpUnsigned, typename OpFloat, typename T2>
WrappedValue handleBinaryConst(const T2 &a, const WrappedValue &b) {
  return handleBinary<OpSigned, OpUnsigned, OpFloat>(EasyBuilder{b.builder}(a),
                                                     b);
}

// predicate should be ult, ule, ugt, uge, eq or ne. We will fix the
// predicate to by the computeType internally
WrappedValue handleCmp(const WrappedValue &a, const WrappedValue &b,
                       arith::CmpIPredicate predicate) {
  assert(a.builder == b.builder);
  assert(a.computeType == b.computeType);
  switch (a.computeType) {
  case ComputeType::Signed:
    switch (predicate) {
    case arith::CmpIPredicate::ult:
      predicate = arith::CmpIPredicate::slt;
      break;
    case arith::CmpIPredicate::ule:
      predicate = arith::CmpIPredicate::sle;
      break;
    case arith::CmpIPredicate::ugt:
      predicate = arith::CmpIPredicate::sgt;
      break;
    case arith::CmpIPredicate::uge:
      predicate = arith::CmpIPredicate::sge;
      break;
    default:
      break;
    }
    [[fallthrough]];
  case ComputeType::Unsigned:
    return {a.builder,
            a.builder->builder.create<arith::CmpIOp>(a.builder->loc, predicate,
                                                     a.v, b.v),
            ComputeType::Unsigned};
  case ComputeType::Float:
    arith::CmpFPredicate fppredi;
    switch (predicate) {
    case arith::CmpIPredicate::ult:
      fppredi = arith::CmpFPredicate::OLT;
      break;
    case arith::CmpIPredicate::ule:
      fppredi = arith::CmpFPredicate::OLE;
      break;
    case arith::CmpIPredicate::ugt:
      fppredi = arith::CmpFPredicate::OGT;
      break;
    case arith::CmpIPredicate::uge:
      fppredi = arith::CmpFPredicate::OGE;
      break;
    case arith::CmpIPredicate::eq:
      fppredi = arith::CmpFPredicate::OEQ;
      break;
    case arith::CmpIPredicate::ne:
      fppredi = arith::CmpFPredicate::ONE;
      break;
    default:
      break;
    }
    return {a.builder,
            a.builder->builder.create<arith::CmpFOp>(a.builder->loc, fppredi,
                                                     a.v, b.v),
            ComputeType::Unsigned};
  }
  return WrappedValue{};
}

template <typename T2>
WrappedValue handleCmpConst(const WrappedValue &a, const T2 &b,
                            arith::CmpIPredicate predicate) {
  return handleCmp(a, EasyBuilder{a.builder}(b), predicate);
}

template <typename T2>
WrappedValue handleCmpConst(const T2 &a, const WrappedValue &b,
                            arith::CmpIPredicate predicate) {
  return handleCmp(EasyBuilder{b.builder}(a), b, predicate);
}

#define DEF_EASYBUILD_BINARY_OPERATOR(OP, SIGNED, UNSIGNED, FLOAT)             \
  WrappedValue operator OP(const WrappedValue &a, const WrappedValue &b) {     \
    return handleBinary<SIGNED, UNSIGNED, FLOAT>(a, b);                        \
  }                                                                            \
  template <typename T>                                                        \
  WrappedValue operator OP(const WrappedValue &a, T b) {                       \
    return handleBinaryConst<SIGNED, UNSIGNED, FLOAT>(a, b);                   \
  }                                                                            \
  template <typename T>                                                        \
  WrappedValue operator OP(T a, const WrappedValue &b) {                       \
    return handleBinaryConst<SIGNED, UNSIGNED, FLOAT>(a, b);                   \
  }

DEF_EASYBUILD_BINARY_OPERATOR(+, arith::AddIOp, arith::AddIOp, arith::AddFOp)
DEF_EASYBUILD_BINARY_OPERATOR(-, arith::SubIOp, arith::SubIOp, arith::SubFOp)
DEF_EASYBUILD_BINARY_OPERATOR(*, arith::MulIOp, arith::MulIOp, arith::MulFOp)
DEF_EASYBUILD_BINARY_OPERATOR(/, arith::DivSIOp, arith::DivUIOp, arith::DivFOp)
DEF_EASYBUILD_BINARY_OPERATOR(%, arith::RemSIOp, arith::RemUIOp, arith::RemFOp)
DEF_EASYBUILD_BINARY_OPERATOR(>>, arith::ShRSIOp, arith::ShRUIOp,
                              impl::Unreachable)
DEF_EASYBUILD_BINARY_OPERATOR(<<, arith::ShLIOp, arith::ShLIOp,
                              impl::Unreachable)
DEF_EASYBUILD_BINARY_OPERATOR(&, arith::AndIOp, arith::AndIOp,
                              impl::Unreachable)
DEF_EASYBUILD_BINARY_OPERATOR(|, arith::OrIOp, arith::OrIOp, impl::Unreachable)
DEF_EASYBUILD_BINARY_OPERATOR(^, arith::XOrIOp, arith::XOrIOp,
                              impl::Unreachable)
#undef DEF_EASYBUILD_BINARY_OPERATOR

WrappedValue operator-(const WrappedValue &a) {
  if (a.computeType != ComputeType::Float) {
    llvm_unreachable("operator- can only be applied on float values");
  }
  return {a.builder,
          a.builder->builder.create<arith::NegFOp>(a.builder->loc, a.v),
          a.computeType};
}

#define DEF_EASYBUILD_CMP_OPERATOR(OP, PRED)                                   \
  WrappedValue operator OP(const WrappedValue &a, const WrappedValue &b) {     \
    return handleCmp(a, b, arith::CmpIPredicate::PRED);                        \
  }                                                                            \
  template <typename T>                                                        \
  WrappedValue operator OP(const WrappedValue &a, T b) {                       \
    return handleCmpConst(a, b, arith::CmpIPredicate::PRED);                   \
  }                                                                            \
  template <typename T>                                                        \
  WrappedValue operator OP(T a, const WrappedValue &b) {                       \
    return handleCmpConst(a, b, arith::CmpIPredicate::PRED);                   \
  }

DEF_EASYBUILD_CMP_OPERATOR(<, ult)
DEF_EASYBUILD_CMP_OPERATOR(<=, ule)
DEF_EASYBUILD_CMP_OPERATOR(>, ugt)
DEF_EASYBUILD_CMP_OPERATOR(>=, uge)
DEF_EASYBUILD_CMP_OPERATOR(==, eq)
DEF_EASYBUILD_CMP_OPERATOR(!=, ne)

#undef DEF_EASYBUILD_CMP_OPERATOR

} // namespace easybuild
} // namespace mlir
#endif
