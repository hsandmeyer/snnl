
#pragma once
#include "connector.h"
#include "element_wise_connector.h"

namespace snnl
{

template<typename TElem>
struct CalcSin
{

    static TElem forward(TElem& input) { return std::sin(input); }

    static TElem backward(TElem& input) { return std::cos(input); }
};

template<class TElem>
using SinConnector = ElementWiseConnector<TElem, CalcSin>;

template<class TElem>
NodeShPtr<TElem> Sin(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<SinConnector>(node);
}

template<typename TElem>
struct CalcCos
{

    static TElem forward(TElem& input) { return std::cos(input); }

    static TElem backward(TElem& input) { return -std::sin(input); }
};

template<class TElem>
using CosConnector = ElementWiseConnector<TElem, CalcCos>;

template<class TElem>
NodeShPtr<TElem> Cos(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<CosConnector>(node);
}

template<typename TElem>
struct CalcSigmoid
{

    static TElem forward(TElem& input)
    {
        return static_cast<TElem>(1) / (static_cast<TElem>(1) + std::exp(-input));
    }

    static TElem backward(TElem& input)
    {

        TElem tmp = std::exp(-input) + 1;
        return std::exp(-input) / (tmp * tmp);
    }
};

template<class TElem>
using SigmoidConnector = ElementWiseConnector<TElem, CalcSigmoid>;

template<class TElem>
NodeShPtr<TElem> Sigmoid(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<SigmoidConnector>(node);
}

template<typename TElem>
struct CalcReLu
{

    static TElem forward(TElem& input)
    {
        if(input < 0) {
            return 0;
        }
        else {
            return input;
        }
    }

    static TElem backward(TElem& input)
    {
        if(input < 0) {
            return 0;
        }
        else {
            return 1;
        }
    }
};

template<class TElem>
using ReLuConnector = ElementWiseConnector<TElem, CalcReLu>;

template<class TElem>
NodeShPtr<TElem> ReLu(const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<ReLuConnector>(node);
}

} // namespace snnl