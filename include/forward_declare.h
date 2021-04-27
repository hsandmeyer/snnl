#pragma once
#include <memory>

namespace snnl {

template <class TElem>
class Tensor;

template <class TElem>
class Connector;

template <class TElem>
class Node;

template <class TElem>
class Module;

template <class TElem>
using NodeShPtr = std::shared_ptr<Node<TElem>>;

template <class TElem>
using ConnectorShPtr = std::shared_ptr<Connector<TElem>>;

template <class TElem>
using ModuleShPtr = std::shared_ptr<Module<TElem>>;

} // namespace snnl