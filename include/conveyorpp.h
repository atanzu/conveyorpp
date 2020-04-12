/*
 * MIT License
 *
 * Copyright (c) 2020 Mark K
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <type_traits>
#include <utility>

#include "conveyorpp_internal.h"

namespace cnvpp {

template <typename Out, typename... In>
struct LambdaWrapper {
  using LambdaFunctorType = std::function<Out(In...)>;
  LambdaWrapper(LambdaFunctorType f) : m_functor(f) {}
  LambdaFunctorType m_functor;
};

template <typename Out>
struct LambdaWrapper<Out, void> {
  using LambdaFunctorType = std::function<Out(void)>;
  LambdaWrapper(LambdaFunctorType f) : m_functor(f) {}
  LambdaFunctorType m_functor;
};

template <typename... Args>
auto MakeChainer(Args&&... args) {
  auto refs = std::make_tuple(
      internal::MakeFunctionWrapper(std::forward<Args>(args))...);

  using ConveyorInputType = typename std::remove_reference<decltype(
      std::get<0>(refs))>::type::InputTupleType;

  using ConveyorOutputType = typename std::remove_reference<decltype(
      std::get<sizeof...(Args) - 1>(refs))>::type::OutputType;

  auto pipeline = internal::AddExtractor<decltype(refs), ConveyorOutputType>(
      std::move(refs), std::is_same<ConveyorOutputType, void>());

  return internal::Chainer<decltype(pipeline), ConveyorInputType,
                           ConveyorOutputType>{std::move(pipeline)};
}

}  // namespace cnvpp
