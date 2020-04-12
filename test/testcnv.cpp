#include <bits/c++config.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <thread>
#include <tuple>
#include <vector>

#include "conveyorpp.h"
#include "gtest/gtest.h"

using namespace std::chrono_literals;
using ::testing::Test;

float test_function_int__float(int i) { return i / 2.0f; }

int test_function_double_float__int(double d, float f) { return (d + f) / 2; }

auto test_lambda_float__float = [](float f) { return f * 2; };

double tf1(int i, long l) { return (float)l / i; }

float tf2(double d) { return d * 2; }

long long tf3(float f) { return (long long)f + 1; }

TEST(ConveyorppTest, Queue) {
  cnvpp::internal::ConveyorQueue<int> cnvq{5};
  std::vector<int> samples{10};
  std::iota(begin(samples), end(samples), 1);

  std::thread feedThread{[&] {
    for (const auto& e : samples) {
      cnvq.Push(e);
    }
  }};

  for (auto i = 0; i < samples.size(); ++i) {
    auto val = cnvq.Pop();
    EXPECT_EQ(val, samples[i]);
  }

  feedThread.join();
}

TEST(ConveyorppTest, Element) {
  const size_t kCycleCount = 20;

  std::vector<int> inputs(kCycleCount);
  std::iota(begin(inputs), end(inputs), 0);

  std::vector<float> outputs;
  std::transform(cbegin(inputs), cend(inputs), std::back_inserter(outputs),
                 [&](auto i) {
                   return test_lambda_float__float(test_function_int__float(
                       test_function_double_float__int(i * 30.0, i / 0.1)));
                 });

  cnvpp::internal::ConveyorElement<int, double, float> ce1(
      test_function_double_float__int);
  cnvpp::internal::ConveyorElement<float, int> ce2(test_function_int__float);
  cnvpp::internal::ConveyorElement<float, float> ce3(test_lambda_float__float);
  cnvpp::internal::ConveyorExtractor<float> extractor{};

  ce3.SetCallback([&](float f) { extractor.Call(f); });
  ce3.Start();
  ce2.SetCallback([&](float f) { ce3.Call(f); });
  ce2.Start();
  ce1.SetCallback([&](int i) { ce2.Call(i); });
  ce1.Start();

  std::thread feedThread{[&] {
    for (auto val : inputs) {
      ce1.Call(val * 30.0, val / 0.1);
    }
  }};

  for (auto v : outputs) {
    auto r = extractor.GetResult();
    EXPECT_EQ(v, r);
  }

  feedThread.join();
}

TEST(ConveyorppTest, Concurrency) {
  const size_t kCycleCount = 20;

  auto timestampStart = std::chrono::steady_clock::now();

  size_t f1_increment = 10;
  auto f1 = [=](size_t st) {
    std::this_thread::sleep_for(50ms);
    return st + f1_increment;
  };
  size_t f2_decrement = 5;
  auto f2 = [&](size_t st) {
    std::this_thread::sleep_for(100ms);
    return st - f2_decrement;
  };
  auto f3 = [](size_t st) {
    std::this_thread::sleep_for(75ms);
    return st * 2;
  };

  for (auto i = 0; i < kCycleCount; ++i) {
    f3(f2(f1(i)));
  }

  auto durationUsual = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - timestampStart);

  timestampStart = std::chrono::steady_clock::now();

  cnvpp::internal::ConveyorElement<size_t, size_t> ce1(f1);
  cnvpp::internal::ConveyorElement<size_t, size_t> ce2(f2);
  cnvpp::internal::ConveyorElement<size_t, size_t> ce3(f3);
  cnvpp::internal::ConveyorExtractor<size_t> extractor{};

  ce3.SetCallback([&](auto f) { extractor.Call(f); });
  ce3.Start();
  ce2.SetCallback([&](auto i) { ce3.Call(i); });
  ce2.Start();
  ce1.SetCallback([&](auto i) { ce2.Call(i); });
  ce1.Start();

  std::thread feedThread{[&] {
    for (auto i = 0; i < kCycleCount; ++i) {
      ce1.Call(i);
    }
  }};

  for (auto i = 0; i < kCycleCount; ++i) {
    extractor.GetResult();
  }

  feedThread.join();

  auto durationConveyorized =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - timestampStart);

  EXPECT_LE(durationConveyorized * 2, durationUsual);
}

TEST(ConveyorppTest, FunctionTraits) {
  std::function<long(short, int, long)> stdfunc = [](short a, int b, long c) {
    return c + b + a;
  };
  typedef decltype(cnvpp::internal::MakeFunctionWrapper(stdfunc)) stdfunc_info;
  auto stdfunc_nargs = stdfunc_info::nargs;
  EXPECT_EQ(stdfunc_nargs, 3);

  auto lambda = [](short a, int b) -> long { return (long)(a + b); };
  cnvpp::LambdaWrapper<long, short, int> lw(lambda);
  typedef decltype(cnvpp::internal::MakeFunctionWrapper(lw)) lambda_info;
  auto lambda_nargs = lambda_info::nargs;
  EXPECT_EQ(lambda_nargs, 2);
}

TEST(ConveyorppTest, BuildConveyor) {
  auto chainer1 = cnvpp::MakeChainer(tf1, tf2, tf3);
  chainer1.Call(4, 8l);
  auto res = chainer1.Result();
  EXPECT_EQ(res, 5);  // Pre-calculated value
}

TEST(ConveyorppTest, BuildConveyorVoidReturn) {
  long long result = 0;
  bool hasResult = false;
  auto lastFunction = [&](long long val) {
    result = val;
    hasResult = true;
  };

  auto chainer1 = cnvpp::MakeChainer(
      tf1, tf2, tf3, cnvpp::LambdaWrapper<void, long long>(lastFunction));
  chainer1.Call(4, 8l);
  while (!hasResult)
    ;
  EXPECT_EQ(result, 5);  // Pre-calculated value
}
