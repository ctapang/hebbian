#include <algorithm>
#include <vector>

#include "caffe/layers/hebbian_relu_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void HebbianReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int count = bottom[0]->count();
		for (int i = 0; i < count; ++i) {
			top_data[i] = std::max(bottom_data[i], Dtype(0));
		}
	}

	template <typename Dtype>
	void HebbianReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[0]) {
			const Dtype* top_data = top[0]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			const int count = bottom[0]->count();
			for (int i = 0; i < count; ++i) {
				bottom_diff[i] = top_data[i];
			}
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(HebbianReLULayer);
#endif

	INSTANTIATE_CLASS(HebbianReLULayer);

}  // namespace caffe
