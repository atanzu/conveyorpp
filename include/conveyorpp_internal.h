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

#include <bits/c++config.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>

namespace cnvpp {

namespace internal {

template <typename F, typename T, size_t... I>
auto callTuple(F&& f, T&& t, std::index_sequence<I...>) {
  return f(std::forward<decltype(std::get<I>(t))>(std::get<I>(t))...);
}

template <typename F, typename T>
auto callTuple(F&& f, T&& t) {
  return callTuple(std::forward<F>(f), std::forward<T>(t),
                   std::make_index_sequence<std::tuple_size<T>::value>{});
}

class ConveyorStopException : public std::exception {};

template <typename T>
class ConveyorQueue {
 public:
  explicit ConveyorQueue(std::size_t capacity) : m_capacity(capacity) {}

  ConveyorQueue(const ConveyorQueue&) = delete;
  ConveyorQueue& operator=(const ConveyorQueue&) = delete;

  ConveyorQueue(ConveyorQueue&& rhs)
      : m_capacity(rhs.m_capacity),
        m_stop(rhs.m_stop.load()),
        m_queue(std::move(rhs.m_queue)) {
    rhs.Stop();
  }

  ~ConveyorQueue() { Stop(); }

  template <typename... Ts>
  void Push(Ts&&... args) {
    static_assert(std::is_constructible<T, Ts...>::value,
                  "Cannot construct queue value from given arguments");
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_queue.size() > m_capacity && !m_stop) {
      m_cv.wait(lock);
      if (m_stop) {
        return;
      }
    }
    m_queue.emplace(std::forward<Ts>(args)...);
    m_cv.notify_all();
  }

  T Pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    while (m_queue.empty() && !m_stop) {
      m_cv.wait(lock);
      if (m_stop) {
        throw ConveyorStopException{};
      }
    }
    T ret = std::move(m_queue.front());
    m_queue.pop();
    m_cv.notify_one();
    return ret;
  }

  void Stop() {
    m_stop = true;
    m_cv.notify_all();
  }

 private:
  const std::size_t m_capacity;
  std::atomic<bool> m_stop{false};
  std::queue<T> m_queue;
  mutable std::mutex m_mutex;
  std::condition_variable m_cv;
};

template <typename Fout, typename... Fin>
class ConveyorElement {
  static constexpr std::size_t m_defaultQueueSize{5};

 public:
  using FunctorArgsTuple = typename std::tuple<Fin...>;
  using Functor = std::function<Fout(Fin...)>;
  using Callback = std::function<void(Fout)>;

  ConveyorElement(Functor f, std::size_t queueSize = m_defaultQueueSize)
      : m_stop(false), m_functor(f), m_argsQueue(queueSize) {}

  ConveyorElement(const ConveyorElement&) = delete;
  ConveyorElement& operator=(const ConveyorElement&) = delete;

  ConveyorElement(ConveyorElement&& rhs)
      : m_stop(rhs.m_stop.load()),
        m_functor(std::move(rhs.m_functor)),
        m_callback(std::move(rhs.m_callback)),
        m_argsQueue(std::move(rhs.m_argsQueue)),
        m_executionThread(std::move(rhs.m_executionThread)) {
    rhs.m_stop = true;
  }

  ~ConveyorElement() {
    m_stop = true;
    m_argsQueue.Stop();
    if (m_executionThread.joinable()) {
      m_executionThread.join();
    }
  }

  template <typename... Args>
  void Call(Args&&... args) {
    m_argsQueue.Push(std::forward<Args>(args)...);
  }

  void SetCallback(std::function<void(Fout)> cb) { m_callback = cb; }

  void Start() {
    m_executionThread = std::thread{[&] { Worker(); }};
  }

 private:
  template <typename U = FunctorArgsTuple>
  typename std::enable_if<std::tuple_size<U>::value != 0>::type Worker() {
    while (!m_stop) {
      try {
        auto arg = callTuple(m_functor, m_argsQueue.Pop());
        m_callback(std::move(arg));
      } catch (const ConveyorStopException&) {
        return;
      } catch (const std::exception& e) {
        std::cerr << "Worker got exception: " << e.what() << std::endl;
      }
    }
  }

  template <typename U = FunctorArgsTuple>
  typename std::enable_if<std::tuple_size<U>::value == 0>::type Worker() {
    while (!m_stop) {
      try {
        auto arg = m_functor();
        m_callback(std::move(arg));
      } catch (const ConveyorStopException&) {
        return;
      } catch (const std::exception& e) {
        std::cerr << "Worker got exception: " << e.what() << std::endl;
      }
    }
  }

  std::atomic<bool> m_stop;
  Functor m_functor;
  Callback m_callback;
  ConveyorQueue<FunctorArgsTuple> m_argsQueue;
  std::thread m_executionThread;
};

template <typename... Fin>
class ConveyorElement<void, Fin...> {
  static constexpr std::size_t m_defaultQueueSize{5};

 public:
  using FunctorArgsTuple = typename std::tuple<Fin...>;
  using Functor = std::function<void(Fin...)>;

  ConveyorElement(Functor f, std::size_t queueSize = m_defaultQueueSize)
      : m_stop(false), m_functor(f), m_argsQueue(queueSize) {}

  ConveyorElement(const ConveyorElement&) = delete;
  ConveyorElement& operator=(const ConveyorElement&) = delete;

  ConveyorElement(ConveyorElement&& rhs)
      : m_stop(rhs.m_stop.load()),
        m_functor(std::move(rhs.m_functor)),
        m_argsQueue(std::move(rhs.m_argsQueue)),
        m_executionThread(std::move(rhs.m_executionThread)) {
    rhs.m_stop = true;
  }

  ~ConveyorElement() {
    m_stop = true;
    m_argsQueue.Stop();
    if (m_executionThread.joinable()) {
      m_executionThread.join();
    }
  }

  template <typename... Args>
  void Call(Args&&... args) {
    m_argsQueue.Push(std::forward<Args>(args)...);
  }

  void Start() {
    m_executionThread = std::thread{[&] { Worker(); }};
  }

 private:
  void Worker() {
    while (!m_stop) {
      try {
        callTuple(m_functor, m_argsQueue.Pop());
      } catch (const ConveyorStopException&) {
        return;
      } catch (const std::exception& e) {
        std::cerr << "Worker got exception: " << e.what() << std::endl;
      }
    }
  }

  std::atomic<bool> m_stop;
  Functor m_functor;
  ConveyorQueue<FunctorArgsTuple> m_argsQueue;
  std::thread m_executionThread;
};

template <typename... Ts>
class ConveyorExtractor {
  template <typename Arg, typename... Args>
  struct ExtractorStorage {
    using Type = std::tuple<Arg, Args...>;
  };
  template <typename Arg>
  struct ExtractorStorage<Arg> {
    using Type = Arg;
  };

  using QueueElementType = typename ExtractorStorage<Ts...>::Type;
  static constexpr std::size_t m_defaultQueueSize{5};

 public:
  ConveyorExtractor() : m_resultsQueue(m_defaultQueueSize) {}

  ConveyorExtractor(ConveyorExtractor&&) = default;

  ~ConveyorExtractor() { m_resultsQueue.Stop(); }

  void Start() {}

  template <typename... Args>
  void Call(Args&&... args) {
    m_resultsQueue.Push(std::forward<Args>(args)...);
  }

  auto GetResult() { return m_resultsQueue.Pop(); }

 private:
  ConveyorQueue<QueueElementType> m_resultsQueue;
};

template <typename T>
struct FunctionWrapper;

template <typename R, typename... Args>
struct FunctionWrapper<std::function<R(Args...)>> {
  static const std::size_t nargs = sizeof...(Args);

  using OutputType = R;
  using InputTupleType = typename std::tuple<Args...>;

  template <std::size_t I>
  struct arg {
    typedef typename std::tuple_element<I, std::tuple<Args...>>::type type;
  };
  FunctionWrapper(std::function<R(Args...)> f) : element(f) {}

  FunctionWrapper(const FunctionWrapper&) = delete;
  FunctionWrapper& operator=(const FunctionWrapper&) = delete;

  FunctionWrapper(FunctionWrapper&&) = default;
  FunctionWrapper& operator=(FunctionWrapper&&) = default;

  template <typename... Ts>
  void Call(Ts&&... args) {
    element.Call(std::forward<Ts>(args)...);
  }

  void Start() { element.Start(); }

  ConveyorElement<R, Args...> element;
};

template <class R, class T, class... Args>
auto MakeFunctionWrapper(R (T::*f)(Args...) const) {
  return FunctionWrapper<std::function<R(Args...)>>(f);
}

template <class R, class T, class... Args>
auto MakeFunctionWrapper(R (T::*f)(Args...)) {
  return FunctionWrapper<std::function<R(Args...)>>(f);
}

template <class R, class... Args>
auto MakeFunctionWrapper(R (*f)(Args...)) {
  return FunctionWrapper<std::function<R(Args...)>>(f);
}

template <class R, class... Args>
auto MakeFunctionWrapper(std::function<R(Args...)> f) {
  return FunctionWrapper<std::function<R(Args...)>>(f);
}

template <typename W>
auto MakeFunctionWrapper(W w) {
  using F = typename W::LambdaFunctorType;
  return FunctionWrapper<F>(w.m_functor);
}

template <typename C, typename ChainIn, typename ChainOut>
class Chainer final {
 public:
  Chainer(C&& chain) : m_chain(std::move(chain)) { SetCallbacks(); }

  Chainer(Chainer&& rhs) = default;

  Chainer& operator=(Chainer&& rhs) = default;

  Chainer(const Chainer&) = delete;
  Chainer& operator=(const Chainer&) = delete;

  ~Chainer() {}

  template <typename U = ChainIn, typename... Ts>
  std::enable_if_t<std::is_same<void, U>::value == false> Call(Ts&&... args) {
    std::get<0>(m_chain).Call(std::forward<Ts>(args)...);
  }

  template <typename U = ChainOut>
  std::enable_if_t<std::is_same<void, U>::value == false, U> Result() {
    return std::get<std::tuple_size<C>::value - 1>(m_chain).GetResult();
  }

 private:
  static constexpr size_t ChainerLen = std::tuple_size<C>::value;

  /*
  template <std::size_t I = 0>
  typename std::enable_if<I != std::tuple_size<C>::value - 1, void>::type
  SetCallbacks() {
    auto& w = std::get<I>(m_chain);
    auto& wnext = std::get<I + 1>(m_chain);
    auto cb = [&wnext](auto&& p) { return wnext.Call(std::move(p)); };
    w.element.SetCallback(cb);
    SetCallbacks<I + 1>();
  }
  */

  template <std::size_t I = ChainerLen - 1>
  typename std::enable_if<I != 0, void>::type SetCallbacks() {
    auto& w = std::get<I>(m_chain);
    auto& wnext = std::get<I - 1>(m_chain);
    auto cb = [&w](auto&& p) { return w.Call(std::move(p)); };
    wnext.element.SetCallback(cb);
    w.Start();
    SetCallbacks<I - 1>();
  }

  template <std::size_t I = ChainerLen - 1>
  typename std::enable_if<I == 0, void>::type SetCallbacks() {
    auto& w = std::get<I>(m_chain);
    w.Start();
  }

  C m_chain;
};

template <typename T, typename E>
auto AddExtractor(T&& t, std::false_type) {
  return std::tuple_cat(std::move(t), std::make_tuple(ConveyorExtractor<E>{}));
}

template <typename T, typename... E>
auto AddExtractor(T&& t, std::true_type) {
  return std::move(t);
}

}  // namespace internal

}  // namespace cnvpp
