# conveyorpp
A simple С++ library for building conveyor-like pipelines

## Description
This library allows users to easily create linear parallel pipelines.
For instance. imagine you have three functions `f1`, `f2` and `f3`, and your result is calculated like
```
output = f3(f2(f1(input)));
```
So, if you continuously feed this pipeline with data, your timing looks like this:

f1: |XXXXXX|              |XXXXXX|             |XXXXXX|

f2:        |XXXXXXXXXX|          |XXXXXXXXXX|         |XXXXXXXXXX|

f3:                   |XX|                  |XX|                 |XX|

An easy way o speed up the process is to execute the function chain in parallel, i.e. create several threads for each `f3(f2(f1(...)))` call. However, sometimes it is not possible, e.g. your functions depend on some hardware calculation modules with exclusive access or aren't reentrant.

Conveyourpp lib solves this problems by simplifying building of a parallel worker chain. For instance, the example above with conveyorpp (with zero-queue settings) looks like this:

f1: |XXXXXX||XXXXXX|...|XXXXXX|....|XXXXXX|....|XXXXXX|....

f2:        |XXXXXXXXXX||XXXXXXXXXX||XXXXXXXXXX||XXXXXXXXXX|

f3:                   |XX|........|XX|........|XX|.........

So if your calculation can be divided in several stages, you can utilize all the computational power you've got.

## Usage
### Usual functions
Imagine you have three functions:
```
int f1(float);
double f2(int);
bool f3(double);
```
You can parallelize their execution by:
```
auto chainer = cnvpp::MakeChainer(f1, f2, f3);
```
This line of code creates execution chain f1->f2->f3, which is equivalent to `f3(f2(f1(...)))`
To feed this conveyor with input data, simply call:
```
int x = 42;
chainer.Call(x);
```
To obtain result of the computation, call
```
auto result = chainer.GetResult();
```
Method `GetResult` blocks until result is available. If the last function's return type is `void`,method `GetResult` is excluded from compilation.

### Lambdas
Lambda-functions need to be processed with some special care. In particular, user has to manually describe their input and output types, e.g.
```
double multiplier = 2.5;
auto l1 = [=] (int x) { return (float)x * multiplier; };
auto l2 = [&] (float y) { return (size_t)(y / multiplier + 1); };

auto chainer = cnvpp::MakeChainer(
  cnvpp::LambdaWrapper<float, int>(l1),
  cnvpp::LambdaWrapper<size_t, float>(l2),
};
```

## Roadmap
* Add more test and examples
* Add more documentation and diagrams
* Find a way to get rid of LambdaWrapper class and handle functions, class methods and lambdas in a unified way
* Make internal queuing mechanism more customizable (at the current moment each conveyor stage has internal queue with size 5)
* Add mechanisms for auto-balancing the load (e.g. user adds a sequence of actions to perform, and conveyorpp's runtime determines optimal amount of threads to use, not just execute each stage in a separate thread)
