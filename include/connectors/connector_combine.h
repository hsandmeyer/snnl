#pragma once
#include "element_wise_combination.h"
#include "forward_declare.h"
#include <stdexcept>

namespace snnl
{

template<typename TElem>
struct CalcAdd
{

    static TElem forward(TElem& input_1, TElem input_2) { return input_1 + input_2; }

    static std::tuple<TElem, TElem> backward(TElem&, TElem&)
    {
        return std::tuple<TElem, TElem>(1, 1);
    }
};

template<class TElem>
using AddConnector = ElementWiseCombination<TElem, CalcAdd>;

template<class TElem>
NodeShPtr<TElem> Add(const NodeShPtr<TElem>& node_1, const NodeShPtr<TElem>& node_2)
{
    return Connector<TElem>::template apply<AddConnector>(node_1, node_2);
}

template<typename TElem>
struct CalcSubtract
{

    static TElem forward(TElem& input_1, TElem input_2) { return input_1 - input_2; }

    static std::tuple<TElem, TElem> backward(TElem&, TElem&)
    {
        return std::tuple<TElem, TElem>(1, -1);
    }
};

template<class TElem>
using SubtractConnector = ElementWiseCombination<TElem, CalcSubtract>;

template<class TElem>
NodeShPtr<TElem> Subtract(const NodeShPtr<TElem>& node_1, const NodeShPtr<TElem>& node_2)
{
    return Connector<TElem>::template apply<SubtractConnector>(node_1, node_2);
}

template<typename TElem>
struct CalcMult
{

    static TElem forward(TElem& input_1, TElem input_2) { return input_1 * input_2; }

    static std::tuple<TElem, TElem> backward(TElem& input_1, TElem& input_2)
    {
        return std::tuple<TElem, TElem>(input_2, input_1);
    }
};

template<class TElem>
using MultConnector = ElementWiseCombination<TElem, CalcMult>;

template<class TElem>
NodeShPtr<TElem> Mult(const NodeShPtr<TElem>& node_1, const NodeShPtr<TElem>& node_2)
{
    return Connector<TElem>::template apply<MultConnector>(node_1, node_2);
}

template<typename TElem>
struct CalcDivide
{

    static TElem forward(TElem& input_1, TElem input_2) { return input_1 / input_2; }

    static std::tuple<TElem, TElem> backward(TElem& input_1, TElem& input_2)
    {
        return std::tuple<TElem, TElem>(1.0 / input_2, -input_1 / (input_2 * input_2));
    }
};

template<class TElem>
using DivideConnector = ElementWiseCombination<TElem, CalcDivide>;

template<class TElem>
NodeShPtr<TElem> Divide(const NodeShPtr<TElem>& node_1, const NodeShPtr<TElem>& node_2)
{
    return Connector<TElem>::template apply<DivideConnector>(node_1, node_2);
}

} // namespace snnl