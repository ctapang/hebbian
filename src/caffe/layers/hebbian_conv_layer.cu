#include <algorithm>
#include <vector>

#include "caffe/layers/hebbian_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Add_gpu_feedback(const int size, const Dtype gain, const Dtype* feedback,
		Dtype* bottom) {
		caffe_gpu_axpy<Dtype>(size, gain, feedback, bottom);
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			Dtype* bottom_data = bottom[i]->mutable_gpu_data();
			const Dtype* feedback = bottom[i]->gpu_diff();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				Add_gpu_feedback(this->bottom_dim_, this->feedback_gain_, feedback + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_);
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}

	template <typename Dtype>
	void HebbianConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		Blob<Dtype>* feedback = new Blob<Dtype>(bottom[0]->shape());
		Dtype* feedback_data = feedback->mutable_gpu_data();
		for (int i = 0; i < top.size(); ++i) {
			Dtype* top_diff = top[i]->mutable_gpu_diff(); // top_diff is set by layer above, which must be a pooling layer (in which it is "bottom_diff")
			const Dtype* top_data = top[i]->gpu_data();

			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				caffe_gpu_set(feedback->count(), Dtype(0), feedback_data);
				for (int n = 0; n < this->num_; ++n) {
					// Hebbian learning (Oja's rule)
					// feedback to bottom, temporarily store in bottom_diff (for use in Oja's rule below, and also in forward above).
					// bottom_diff = (top_diff X weight)
					this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
						bottom_diff + n * this->bottom_dim_);
					// Bias gradient, if necessary.
					if (this->bias_term_ && this->param_propagate_down_[1]) {
						Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
						for (int n = 0; n < this->num_; ++n) {
							this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
						}
					}
					//// each top level winner should be fattened only once, so trim winners
					//// scan all winners, remove duplicate winners
					int toparea = top[i]->count(2, 4);
					int topnodes = top[i]->shape(1);
					for (int k = 0; k < topnodes; k++)
					{
						int next = k * toparea;

						// don't care which intra-channel node wins
						bool found = false;
						for (int m = 0; m < toparea; m++)
						{
							int n = next + m;
							if (top_diff[n] == 1.0)
							{
								if (!found)	found = true;
								else top_diff[n] = 0.0;
							}
						}
					}
					// Oja's rule: weight_diff = top_diff * (bottom_data - bottom_diff) where bottom_diff = (top_diff X weight)  <-- see above statement
					// (it's negative because the last step in Update subtracts the weight_diff from the current weight.)
					caffe_gpu_axpy(this->bottom_dim_, Dtype(-1.), bottom_diff + n * this->bottom_dim_, feedback_data + n * this->bottom_dim_);
					caffe_gpu_axpy(this->bottom_dim_, Dtype(1.), bottom_data + n * this->bottom_dim_, feedback_data + n * this->bottom_dim_);
					// why use top diff here instead of top data? because only pooling layer winner should "learn"
					this->weight_gpu_gemm(feedback_data + n * this->bottom_dim_, top_diff + n * this->top_dim_, weight_diff);
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(HebbianConvLayer);
}