#pragma once
#include "connector.h"

namespace snnl
{

template<class TElem>
class Conv2DConnector : public Connector<TElem>
{

    friend class Connector<TElem>;

    void dimChecks(const std::vector<NodeShPtr<TElem>>& input_nodes) const
    {
        if(input_nodes.size() != 2) {
            throw std::invalid_argument("Need exactly two inputs for conv2d");
        }

        auto& input  = input_nodes.at(1);
        auto& kernel = input_nodes.at(0);

        if(input->NDims() < 3) {
            throw std::invalid_argument(
                "Conv2D: Need at least three dimensions for input in conv2d-layer. Got " +
                input->shape());
        }
        if(kernel->NDims() != 4) {
            throw std::invalid_argument(
                "Conv2D: Need exact four dimensions for kernel in conv2d-layer");
        }
        if(kernel->shape(0) % 2 != 1) {
            throw std::invalid_argument("Conv2D: Only uneven dimensions for kernel allow");
        }
        if(kernel->shape(1) % 2 != 1) {
            throw std::invalid_argument("Conv2D: Only uneven dimensions for kernel allowed");
        }
        if(kernel->shape(-2) != input->shape(-1)) {
            throw std::invalid_argument(
                "Conv2D: Output numer of channels of input does not match number"
                " of output dimensions of kernel (kernel:" +
                std::to_string(kernel->shape(-2)) +
                " vs. input:" + std::to_string(input->shape(-1)) + ")");
        }
    }

    Index outputDims(const std::vector<NodeShPtr<TElem>>& input_nodes) const override
    {
        dimChecks(input_nodes);
        Index input_shape  = input_nodes.at(1)->shape();
        Index kernel_shape = input_nodes.at(0)->shape();
        input_shape[-1]    = kernel_shape[-1];
        return input_shape;
    }

    void forwardHandler(const std::vector<NodeShPtr<TElem>>& input_nodes,
                        Node<TElem>*                         output_node) override
    {
        Tensor<TElem> out_view = output_node->values().viewWithNDimsOnTheRight(4);

        Tensor<TElem>  input  = input_nodes.at(1)->values().viewWithNDimsOnTheRight(4);
        Tensor<TElem>& kernel = input_nodes.at(0)->values();

        size_t n_input_channels  = input.shape(-1);
        size_t n_output_channels = kernel.shape(-1);

        long kernel_width  = kernel.shape(0);
        long kernel_height = kernel.shape(1);

        long image_width  = input.shape(-3);
        long image_height = input.shape(-2);

        long half_width  = kernel_width / 2;
        long half_height = kernel_height / 2;

        for(size_t higherDim = 0; higherDim < input.shape(0); higherDim++) {
            for(long i = 0; i < image_width; i++) {
                for(long j = 0; j < image_height; j++) {

                    long i_kernel_begin = std::max(-i, -half_width);

                    long i_kernel_end = std::min(image_width - i - 1, half_width);

                    for(long i_kernel = i_kernel_begin; i_kernel <= i_kernel_end; i_kernel++) {

                        long j_kernel_begin = std::max(-j, -half_height);

                        long j_kernel_end = std::min(image_height - j - 1, half_height);

                        for(long j_kernel = j_kernel_begin; j_kernel <= j_kernel_end; j_kernel++) {

                            for(size_t out_chan = 0; out_chan < n_output_channels; out_chan++) {

                                for(size_t in_chan = 0; in_chan < n_input_channels; in_chan++) {

                                    out_view(higherDim, i, j, out_chan) +=
                                        kernel(i_kernel + half_width, j_kernel + half_height,
                                               in_chan, out_chan) *
                                        input(higherDim, i + i_kernel, j + j_kernel, in_chan);
                                }
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

        Tensor<TElem>  input       = input_nodes.at(1)->values().viewWithNDimsOnTheRight(4);
        Tensor<TElem>  grad_input  = input_nodes.at(1)->gradient().viewWithNDimsOnTheRight(4);
        Tensor<TElem>& kernel      = input_nodes.at(0)->values();
        Tensor<TElem>& grad_kernel = input_nodes.at(0)->gradient();

        size_t n_input_channels  = input.shape(-1);
        size_t n_output_channels = kernel.shape(-1);

        long kernel_width  = kernel.shape(0);
        long kernel_height = kernel.shape(1);

        long image_width  = input.shape(-3);
        long image_height = input.shape(-2);

        long half_width  = kernel_width / 2;
        long half_height = kernel_height / 2;

        for(size_t higherDim = 0; higherDim < input.shape(0); higherDim++) {

            for(long i = 0; i < image_width; i++) {
                for(long j = 0; j < image_height; j++) {

                    long i_kernel_begin = std::max(-i, -half_width);

                    long i_kernel_end = std::min(image_width - i - 1, half_width);

                    for(long i_kernel = i_kernel_begin; i_kernel <= i_kernel_end; i_kernel++) {

                        long j_kernel_begin = std::max(-j, -half_height);

                        long j_kernel_end = std::min(image_height - j - 1, half_height);

                        for(long j_kernel = j_kernel_begin; j_kernel <= j_kernel_end; j_kernel++) {

                            for(size_t out_chan = 0; out_chan < n_output_channels; out_chan++) {

                                for(size_t in_chan = 0; in_chan < n_input_channels; in_chan++) {

                                    TElem out_grad = out_grad_view(higherDim, i, j, out_chan);

                                    grad_kernel(i_kernel + half_width, j_kernel + half_height,
                                                in_chan, out_chan) +=
                                        input(higherDim, i + i_kernel, j + j_kernel, in_chan) *
                                        out_grad;

                                    grad_input(higherDim, i + i_kernel, j + j_kernel, in_chan) +=
                                        kernel(i_kernel + half_width, j_kernel + half_height,
                                               in_chan, out_chan) *
                                        out_grad;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

public:
    virtual ~Conv2DConnector() {}
};

template<class TElem>
NodeShPtr<TElem> Conv2D(const NodeShPtr<TElem>& kernel, const NodeShPtr<TElem>& node)
{
    return Connector<TElem>::template apply<Conv2DConnector>(std::move(kernel), std::move(node));
}
} // namespace snnl