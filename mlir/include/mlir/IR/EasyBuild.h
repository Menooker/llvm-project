//===- EasyBuild.h - Easy  IR Builder utilities -----------------*- C++ -*-===//
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

#ifndef MLIR_IR_EASYBUILD_H
#define MLIR_IR_EASYBUILD_H
#include "mlir/IR/Builders.h"
#include <cstdint>
#include <memory>
#include <stddef.h>

namespace mlir {
namespace easybuild {

namespace impl {
struct EasyBuildState {
  OpBuilder &builder;
  Location loc;
  bool u64AsIndex;
  EasyBuildState(OpBuilder &builder, Location loc, bool u64AsIndex)
      : builder{builder}, loc{loc}, u64AsIndex{u64AsIndex} {}
};

using StatePtr = std::shared_ptr<impl::EasyBuildState>;

} // namespace impl

struct EBValue {
  std::shared_ptr<impl::EasyBuildState> builder;
  Value v;
  EBValue() = default;
  EBValue(const impl::StatePtr &builder, Value v) : builder{builder}, v{v} {}
  Value get() const { return v; }
  operator Value() const { return v; }
};

struct DefaultArithWrapper;

struct EasyBuilder {
  std::shared_ptr<impl::EasyBuildState> builder;
  EasyBuilder(OpBuilder &builder, Location loc, bool u64AsIndex = false)
      : builder{
            std::make_shared<impl::EasyBuildState>(builder, loc, u64AsIndex)} {}
  EasyBuilder(const std::shared_ptr<impl::EasyBuildState> &builder)
      : builder{builder} {}
  void setLoc(const Location &l) { builder->loc = l; }

  template <typename W, typename V>
  auto wrapOrFail(V &&v) {
    return W::wrapOrFail(builder, std::forward<V>(v));
  }

  template <typename W, typename V>
  auto wrap(V &&v) {
    auto ret = wrapOrFail<W>(std::forward<V>(v));
    if (failed(ret)) {
      llvm_unreachable("wrap failed!");
    }
    return *ret;
  }

  EBValue operator()(Value v) const { return EBValue{builder, v}; }

  template <typename V>
  auto operator()(V &&v) {
    return wrap<DefaultArithWrapper>(std::forward<V>(v));
  }

  template <typename W = DefaultArithWrapper>
  auto toIndex(uint64_t v) const {
    return W::toIndex(builder, v);
  }
};

} // namespace easybuild
} // namespace mlir
#endif
