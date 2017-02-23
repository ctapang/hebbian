#include <algorithm>
#include <vector>

#include "caffe/layers/hebbian_conv_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::compute_output_shape() {
		const int* kernel_shape_data = this->kernel_shape_.cpu_data();
		const int* stride_data = this->stride_.cpu_data();
		const int* pad_data = this->pad_.cpu_data();
		const int* dilation_data = this->dilation_.cpu_data();
		this->output_shape_.clear();
		for (int i = 0; i < this->num_spatial_axes_; ++i) {
			// i + 1 to skip channel axis
			const int input_dim = this->input_shape(i + 1);
			const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
			const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
				/ stride_data[i] + 1;
			this->output_shape_.push_back(output_dim);
		}
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Add_feedback(const int size, const Dtype gain, const Dtype* feedback,
		Dtype* bottom) {
		caffe_axpy<Dtype>(size, gain, feedback, bottom);
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* weight = this->blobs_[0]->cpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* feedback = bottom[i]->cpu_diff();
			Dtype* bottom_data = bottom[i]->mutable_cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				Add_feedback(this->bottom_dim_, this->feedback_gain_, feedback + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_);
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* weight = this->blobs_[0]->cpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
		Blob<Dtype>* feedback = new Blob<Dtype>(bottom[0]->shape());
		Dtype* feedback_data = feedback->mutable_cpu_data();
		for (int i = 0; i < top.size(); ++i) {
			Dtype* top_diff = top[i]->mutable_cpu_diff(); // top_diff is set by layer above, which must be a pooling layer (in which it is "bottom_diff")

			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype* bottom_data = bottom[i]->cpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
				caffe_set(feedback->count(), Dtype(0), feedback_data);
				for (int n = 0; n < this->num_; ++n) {
					// Hebbian learning (Oja's rule)
					// feedback to bottom, temporarily store in bottom_diff (for use in Oja's rule below, and also in forward above).
					// bottom_diff = (top_diff X weight)
					this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
						bottom_diff + n * this->bottom_dim_);
					// each top level winner should be fattened only once, so trim winners
					// scan all winners, remove duplicate winners
					int toparea = top[i]->count(2, 4);
					int topnodes = top[i]->shape(1);
					for (int k = 0; k < topnodes; k++)
					{
						int next = k * toparea;
						bool found = false;
						for (int m = 0; m < toparea; m++)
						{
							if (top_diff[next + m] == 1.0)
							{
								if (found) top_diff[next + m] = 0;
								else found = true;
							}
						}
					}
					// Bias gradient, if necessary.
					if (this->bias_term_ && this->param_propagate_down_[1]) {
						Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
						for (int n = 0; n < this->num_; ++n) {
							this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
						}
					}
					// Oja's rule: weight_diff = top_diff * (bottom_data - bottom_diff) where bottom_diff = (top_diff X weight)  <-- see above statement
					// (it's negative because the last step in Update subtracts the weight_diff from the current weight.)
					caffe_axpy(this->bottom_dim_, Dtype(-1.), bottom_diff + n * this->bottom_dim_, feedback_data + n * this->bottom_dim_);
					caffe_axpy(this->bottom_dim_, Dtype(1.), bottom_data + n * this->bottom_dim_, feedback_data + n * this->bottom_dim_);
					// why use top diff here instead of top data? because only pooling layer winner should "learn"
					this->weight_cpu_gemm(feedback_data + n * this->bottom_dim_,	top_diff + n * this->top_dim_, weight_diff);
				}
			}
		}
	}

	INSTANTIATE_CLASS(HebbianConvLayer);
	REGISTER_LAYER_CLASS(HebbianConv);
}