
#pragma once
#include "connector.h"

namespace snnl
{

template<class TElem>
class UpSampleConnector : public Connector<TElem>
{

    friend class Connector<TElem>;

    size_t _pool_width;
    size_t _pool_height;

    void dimChecks(const std::vector<NodeShPtr<TElem>>& input_nodes) const
    {
        if(input_nodes.size() != 1) {
            throw std::invalid_argument("Need exactly one input for upsampling2d");
        }
    }

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        Index out_shape = input_nodes.at(0)->shape();
        out_shape[-3] *= _pool_width;
        out_shape[-2] *= _pool_height;

        return out_shape;
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {
        Tensor<TElem> out_view = output_node->values().viewWithNDimsOnTheRight(4);

        Tensor<TElem> input = input_nodes.at(0)->values().viewWithNDimsOnTheRight(4);

        size_t n_channels = input.shape(-1);

        size_t image_width  = input.shape(-3);
        size_t image_height = input.shape(-2);

        TElem weight = static_cast<TElem>(static_cast<TElem>(1.0) /
                                          static_cast<TElem>(_pool_height * _pool_width));

        for(size_t higherDim = 0; higherDim < input.shape(0); higherDim++) {
            for(size_t i = 0; i < image_width; i++) {
                for(size_t j = 0; j < image_height; j++) {

                    for(size_t i_pool = 0; i_pool < _pool_width; i_pool++) {

                        for(size_t j_pool = 0; j_pool < _pool_height; j_pool++) {

                            for(size_t out_chan = 0; out_chan < n_channels; out_chan++) {
                                out_view(higherDim, i * _pool_width + i_pool,
                                         j * _pool_height + j_pool, out_chan) +=
                                    weight * input(higherDim, i, j, out_chan);
                            }
                        }
                    }
                }
            }
        }
    }

    void backwardHandler(const Node<TElem>*             output_node,
                         std::vector<NodeShPtr<TElem>>& input_nodes) override
    {

        Tensor<TElem> out_grad_view = output_node->gradient().viewWithNDimsOnTheRight(4);

        Tensor<TElem> input_grad = input_nodes.at(0)->gradient().viewWithNDimsOnTheRight(4);

        size_t n_channels = input_grad.shape(-1);

        size_t image_width  = input_grad.shape(-3);
        size_t image_height = input_grad.shape(-2);

        TElem weight = static_cast<TElem>(static_cast<TElem>(1.0) /
                                          static_cast<TElem>(_pool_height * _pool_width));

        for(size_t higherDim = 0; higherDim < input_grad.shape(0); higherDim++) {
            for(size_t i = 0; i < image_width; i++) {
                for(size_t j = 0; j < image_height; j++) {

                    for(size_t i_pool = 0; i_pool < _pool_width; i_pool++) {

                        for(size_t j_pool = 0; j_pool < _pool_height; j_pool++) {

                            for(size_t out_chan = 0; out_chan < n_channels; out_chan++) {
                                input_grad(higherDim, i, j, out_chan) +=
                                    weight * out_grad_view(higherDim, i * _pool_width + i_pool,
                                                           j * _pool_height + j_pool, out_chan);
                            }
                        }
                    }
                }
            }
        }
    }

    UpSampleConnector(size_t pool_width, size_t pool_height)
        : _pool_width(pool_width)
        , _pool_height(pool_height)
    {
    }

public:
    virtual ~UpSampleConnector() {}
};

template<class TElem>
NodeShPtr<TElem> UpSample2D(const NodeShPtr<TElem>& node, size_t pool_height, size_t pool_width)
{
    auto conn = Connector<TElem>::template create<UpSampleConnector>(pool_height, pool_width);
    {
        return conn->call(node);
    }
}

} // namespace snnl