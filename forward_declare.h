#pragma once
#include <memory>

namespace snnl {

template <class TElem>
class TConnectorBaseImpl;

template <class TElem>
class TConnector;

template <class TElem>
class TNodeBaseImpl;

template <class TElem>
class TNode;

template <class TElem>
using TNodePtr = std::shared_ptr<TNode<TElem>>;

template <class TElem>
using TConnectorPtr = std::shared_ptr<TConnector<TElem>>;

} // namespace snnl