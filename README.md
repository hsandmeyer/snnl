# Simple neural network library
A zero dependency, simple library for neural networks using only a few hundred lines of code, written in C++. Currently, I develope this library in order to teach
myself the fundamentals of machine learning.

There is no performance optimization, nor any acceleration methods present. It is not even parallized yet. On the one hand, this makes is quite slow.
On the other it stays readable. So if you are interested in what happens behind the facade of the large machine learning libraries like tensorflow or pytorch, you 
might want too look into the source code presented here. While the library lacks a lot of standard machine learning layers (called connectors here), the automatic
differenciation already works for all kind of complex networks. It supports weight sharing, scip connections and recurrent networks. However there is still a lot to do
until this becomes a usable library.
