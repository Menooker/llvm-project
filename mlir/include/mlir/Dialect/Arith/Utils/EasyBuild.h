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
#include "mlir/IR/EasyBuild.h"
#include <cstdint>
#include <memory>
#include <stddef.h>

namespace mlir {
namespace easybuild {

namespace impl {

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

inline Type getElementType(Value v) {
  auto type = v.getType();
  if (type.isa<TensorType>() || type.isa<VectorType>()) {
    type = type.cast<ShapedType>().getElementType();
  }
  return type;
}

} // namespace impl

struct EBArithValue : public EBValue {
protected:
  using EBValue::EBValue;
};

struct EBUnsigned : public EBArithValue {
  static FailureOr<EBUnsigned> wrapOrFail(const impl::StatePtr &state,
                                          Value v) {
    auto type = impl::getElementType(v);
    if (type.isUnsignedInteger() || type.isSignlessInteger() ||
        type.isIndex()) {
      return EBUnsigned{state, v};
    }
    return failure();
  }
  static FailureOr<EBUnsigned> wrapOrFail(const impl::StatePtr &state,
                                          const OpFoldResult &v) {
    if (v.is<Value>()) {
      return wrapOrFail(state, v.get<Value>());
    }
    auto attr = v.get<Attribute>();
    if (auto val = attr.dyn_cast<IntegerAttr>()) {
      if (val.getType().isIndex())
        return EBUnsigned{state, state->builder.create<arith::ConstantIndexOp>(
                                     state->loc, val.getInt())};
      else
        return EBUnsigned{state, state->builder.create<arith::ConstantIntOp>(
                                     state->loc, val.getInt(), val.getType())};
    }
    return failure();
  }
  friend struct DefaultArithWrapper;
  friend struct OperatorHandlers;

protected:
  using EBArithValue::EBArithValue;
};

struct EBSigned : EBArithValue {
  static FailureOr<EBSigned> wrapOrFail(const impl::StatePtr &state, Value v) {
    auto type = impl::getElementType(v);
    if (type.isSignedInteger() || type.isSignlessInteger()) {
      return EBSigned{state, v};
    }
    return failure();
  }
  static FailureOr<EBSigned> wrapOrFail(const impl::StatePtr &state,
                                        const OpFoldResult &v) {
    if (v.is<Value>()) {
      return wrapOrFail(state, v.get<Value>());
    }
    auto attr = v.get<Attribute>();
    if (auto val = attr.dyn_cast<IntegerAttr>())
      return EBSigned{state, state->builder.create<arith::ConstantIntOp>(
                                 state->loc, val.getInt(), val.getType())};
    return failure();
  }
  friend struct DefaultArithWrapper;
  friend struct OperatorHandlers;

protected:
  using EBArithValue::EBArithValue;
};

struct EBFloatPoint : EBArithValue {
  static FailureOr<EBFloatPoint> wrapOrFail(const impl::StatePtr &state,
                                            Value v) {
    auto type = impl::getElementType(v);
    if (type.isa<FloatType>()) {
      return EBFloatPoint{state, v};
    }
    return failure();
  }
  static FailureOr<EBFloatPoint> wrapOrFail(const impl::StatePtr &state,
                                            const OpFoldResult &v) {
    if (v.is<Value>()) {
      return wrapOrFail(state, v.get<Value>());
    }
    auto attr = v.get<Attribute>();
    if (auto val = attr.dyn_cast<FloatAttr>())
      return EBFloatPoint{state, state->builder.create<arith::ConstantFloatOp>(
                                     state->loc, val.getValue(),
                                     val.getType().cast<FloatType>())};
    return failure();
  }
  friend struct DefaultArithWrapper;
  friend struct OperatorHandlers;

protected:
  using EBArithValue::EBArithValue;
};

struct DefaultArithWrapper {
  static EBUnsigned toIndex(const impl::StatePtr &state, uint64_t v) {
    return EBUnsigned{
        state, state->builder.create<arith::ConstantIndexOp>(state->loc, v)};
  }

  template <typename T>
  static auto wrapOrFail(const impl::StatePtr &state, T &&v) {
    using DT = std::decay_t<T>;
    if constexpr (std::is_convertible_v<DT, Value>) {
      return FailureOr<EBValue>{EBValue{state, std::forward<T>(v)}};
    } else {
      static_assert(std::is_arithmetic_v<DT>, "Expecting arithmetic types");
      if constexpr (std::is_same_v<DT, uint64_t>) {
        if (state->u64AsIndex) {
          return FailureOr<EBUnsigned>{toIndex(state, v)};
        }
      }

      if constexpr (std::is_same_v<DT, bool>) {
        return FailureOr<EBUnsigned>{
            EBUnsigned{state, state->builder.create<arith::ConstantIntOp>(
                                  state->loc, static_cast<int64_t>(v), 1)}};
      } else if constexpr (std::is_integral_v<DT>) {
        if constexpr (!std::is_signed_v<DT>) {
          return FailureOr<EBUnsigned>{EBUnsigned{
              state, state->builder.create<arith::ConstantIntOp>(
                         state->loc, static_cast<int64_t>(v), sizeof(T) * 8)}};
        } else {
          return FailureOr<EBSigned>{EBSigned{
              state, state->builder.create<arith::ConstantIntOp>(
                         state->loc, static_cast<int64_t>(v), sizeof(T) * 8)}};
        }
      } else {
        using DType = typename impl::ToFloatType<sizeof(DT)>::type;
        return FailureOr<EBFloatPoint>{
            EBFloatPoint{state, state->builder.create<arith::ConstantFloatOp>(
                                    state->loc, APFloat{v},
                                    DType::get(state->builder.getContext()))}};
      }
    }
  }
};

struct OperatorHandlers {
  template <typename OP, typename V>
  static V handleBinary(const V &a, const V &b) {
    assert(a.builder == b.builder);
    return {a.builder,
            a.builder->builder.template create<OP>(a.builder->loc, a.v, b.v)};
  }

  template <typename OP, typename V, typename T2>
  static V handleBinaryConst(const V &a, const T2 &b) {
    return handleBinary<OP>(a, EasyBuilder{a.builder}(b));
  }

  template <typename OP, typename V, typename T2>
  static V handleBinaryConst(const T2 &a, const V &b) {
    return handleBinary<OP>(EasyBuilder{b.builder}(a), b);
  }

  template <typename OP, typename V, typename Pred>
  static EBUnsigned handleCmp(const V &a, const V &b, Pred predicate) {
    assert(a.builder == b.builder);
    return {a.builder, a.builder->builder.template create<OP>(
                           a.builder->loc, predicate, a.v, b.v)};
  }

  template <typename OP, typename V, typename T2, typename Pred>
  static EBUnsigned handleCmpConst(const V &a, const T2 &b, Pred predicate) {
    return handleCmp<OP>(a, EasyBuilder{a.builder}(b), predicate);
  }

  template <typename OP, typename V, typename T2, typename Pred>
  static EBUnsigned handleCmpConst(const T2 &a, const V &b, Pred predicate) {
    return handleCmp<OP>(EasyBuilder{b.builder}(a), b, predicate);
  }

  static EBFloatPoint handleNeg(const EBFloatPoint &a) {
    return {a.builder, a.builder->builder.template create<arith::NegFOp>(
                           a.builder->loc, a.v)};
  }
};

#define DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, OPCLASS, TYPE)              \
  inline TYPE operator OP(const TYPE &a, const TYPE &b) {                      \
    return OperatorHandlers::handleBinary<OPCLASS>(a, b);                      \
  }                                                                            \
  template <typename T>                                                        \
  inline TYPE operator OP(const TYPE &a, T b) {                                \
    return OperatorHandlers::handleBinaryConst<OPCLASS, TYPE>(a, b);           \
  }                                                                            \
  template <typename T>                                                        \
  inline TYPE operator OP(T a, const TYPE &b) {                                \
    return OperatorHandlers::handleBinaryConst<OPCLASS, TYPE>(a, b);           \
  }

#define DEF_EASYBUILD_BINARY_OPERATOR(OP, SIGNED, UNSIGNED, FLOAT)             \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, SIGNED, EBSigned)                 \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, UNSIGNED, EBUnsigned)             \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, FLOAT, EBFloatPoint)

DEF_EASYBUILD_BINARY_OPERATOR(+, arith::AddIOp, arith::AddIOp, arith::AddFOp)
DEF_EASYBUILD_BINARY_OPERATOR(-, arith::SubIOp, arith::SubIOp, arith::SubFOp)
DEF_EASYBUILD_BINARY_OPERATOR(*, arith::MulIOp, arith::MulIOp, arith::MulFOp)
DEF_EASYBUILD_BINARY_OPERATOR(/, arith::DivSIOp, arith::DivUIOp, arith::DivFOp)
DEF_EASYBUILD_BINARY_OPERATOR(%, arith::RemSIOp, arith::RemUIOp, arith::RemFOp)

#undef DEF_EASYBUILD_BINARY_OPERATOR
#define DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(OP, SIGNED, UNSIGNED)            \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, SIGNED, EBSigned)                 \
  DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE(OP, UNSIGNED, EBUnsigned)

DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(>>, arith::ShRSIOp, arith::ShRUIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(<<, arith::ShLIOp, arith::ShLIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(&, arith::AndIOp, arith::AndIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(|, arith::OrIOp, arith::OrIOp)
DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT(^, arith::XOrIOp, arith::XOrIOp)

#undef DEF_EASYBUILD_BINARY_OPERATOR_FOR_INT
#undef DEF_EASYBUILD_BINARY_OPERATOR_FOR_TYPE

inline EBFloatPoint operator-(const EBFloatPoint &a) {
  return OperatorHandlers::handleNeg(a);
}

#define DEF_EASYBUILD_CMP_OPERATOR(OP, OPCLASS, TYPE, PRED)                    \
  EBUnsigned operator OP(const TYPE &a, const TYPE &b) {                       \
    return OperatorHandlers::handleCmp<OPCLASS>(a, b, PRED);                   \
  }                                                                            \
  template <typename T>                                                        \
  EBUnsigned operator OP(const TYPE &a, T b) {                                 \
    return OperatorHandlers::handleCmpConst<OPCLASS, TYPE>(a, b, PRED);        \
  }                                                                            \
  template <typename T>                                                        \
  EBUnsigned operator OP(T a, const TYPE &b) {                                 \
    return OperatorHandlers::handleCmpConst<OPCLASS, TYPE>(a, b, PRED);        \
  }

DEF_EASYBUILD_CMP_OPERATOR(<, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ult)
DEF_EASYBUILD_CMP_OPERATOR(<=, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ule)
DEF_EASYBUILD_CMP_OPERATOR(>, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ugt)
DEF_EASYBUILD_CMP_OPERATOR(>=, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::uge)
DEF_EASYBUILD_CMP_OPERATOR(==, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::eq)
DEF_EASYBUILD_CMP_OPERATOR(!=, arith::CmpIOp, EBUnsigned,
                           arith::CmpIPredicate::ne)

DEF_EASYBUILD_CMP_OPERATOR(<, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::slt)
DEF_EASYBUILD_CMP_OPERATOR(<=, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::sle)
DEF_EASYBUILD_CMP_OPERATOR(>, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::sgt)
DEF_EASYBUILD_CMP_OPERATOR(>=, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::sge)
DEF_EASYBUILD_CMP_OPERATOR(==, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::eq)
DEF_EASYBUILD_CMP_OPERATOR(!=, arith::CmpIOp, EBSigned,
                           arith::CmpIPredicate::ne)

DEF_EASYBUILD_CMP_OPERATOR(<, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OLT)
DEF_EASYBUILD_CMP_OPERATOR(<=, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OLE)
DEF_EASYBUILD_CMP_OPERATOR(>, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OGT)
DEF_EASYBUILD_CMP_OPERATOR(>=, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OGE)
DEF_EASYBUILD_CMP_OPERATOR(==, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::OEQ)
DEF_EASYBUILD_CMP_OPERATOR(!=, arith::CmpFOp, EBFloatPoint,
                           arith::CmpFPredicate::ONE)

#undef DEF_EASYBUILD_CMP_OPERATOR

} // namespace easybuild
} // namespace mlir
#endif
