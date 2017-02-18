#pragma once

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"

namespace caffe {

	// The hebbian checker presents input to a hebbian layer, and checks that
	// weights change in the winning unit according to input.
	template <typename Dtype>
	class HebbianChecker {
	public:
		// kink and kink_range specify an ignored nonsmooth region of the form
		// kink - kink_range <= |feature value| <= kink + kink_range,
		// which accounts for all nonsmoothness in use by caffe
		HebbianChecker(const Dtype stepsize, const Dtype threshold,
			const unsigned int seed = 1701, const Dtype kink = 0.,
			const Dtype kink_range = -1)
			: stepsize_(stepsize), threshold_(threshold), seed_(seed),
			kink_(kink), kink_range_(kink_range) {
					bottomMem_ = nullptr;
		}
		// Checks the hebbian weights of a layer, with provided bottom layers and top
		// layers.
		// Note that after the hebbian weights check, we do not guarantee that the data
		// stored in the layer parameters and the blobs are unchanged.
		void CheckHebbian(Layer<Dtype>* layer, const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top, int check_bottom = -1) {
			layer->SetUp(bottom, top);
			CheckHebbianSingle(layer, bottom, top, check_bottom, -1, -1);
		}
		void CheckHebbianExhaustive(Layer<Dtype>* layer,
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
			int check_bottom = -1);

		// CheckHebbianEltwise can be used to test layers that perform element-wise
		// computation only (e.g., neuron layers) -- where (d y_i) / (d x_j) = 0 when
		// i != j.
		void CheckHebbianEltwise(Layer<Dtype>* layer,
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		// Checks the hebbian weights of a single output with respect to particular input
		// blob(s).  If check_bottom = i >= 0, check only the ith bottom Blob.
		// If check_bottom == -1, check everything -- all bottom Blobs and all
		// param Blobs.  Otherwise (if check_bottom < -1), check only param Blobs.
		void CheckHebbianSingle(Layer<Dtype>* layer,
			const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
			int check_bottom, int top_id, int top_data_id, bool element_wise = false);

		// Checks the hebbian weights of a network. This network should not have any data
		// layers or loss layers, since the function does not explicitly deal with
		// such cases yet. All input blobs and parameter blobs are going to be
		// checked, layer-by-layer to avoid numerical problems to accumulate.
		void CheckHebbianNet(const Net<Dtype>& net,
			const vector<Blob<Dtype>*>& input);

	protected:
		void SetFeedbackWinner(const Layer<Dtype>& layer,
			const vector<Blob<Dtype>*>& top, int top_id = -1, int top_data_id = -1);
		void SetFeedbackBuffer(Layer<Dtype>* layer, const vector<Blob<Dtype>*>& topBVec, const vector<Blob<Dtype>*>& bottomBVec, int top_id, int top_data_id);
		Dtype GetLossUsingOjasRule(Layer<Dtype>* layer, const vector<Blob<Dtype>*>& topBVec, const vector<Blob<Dtype>*>& bottom, const Blob<Dtype>* blobToCheck, int top_id, int top_data_id, int feature);
		Dtype stepsize_;
		Dtype threshold_;
		unsigned int seed_;
		Dtype kink_;
		Dtype kink_range_;
		Dtype* bottomMem_;
	};


	template <typename Dtype>
	void HebbianChecker<Dtype>::CheckHebbianSingle(Layer<Dtype>* layer,
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int check_bottom, int top_id, int top_data_id, bool element_wise) {
		if (element_wise) {
			CHECK_EQ(0, layer->blobs().size());
			CHECK_LE(0, top_id);
			CHECK_LE(0, top_data_id);
			const int top_count = top[top_id]->count();
			for (int blob_id = 0; blob_id < bottom.size(); ++blob_id) {
				CHECK_EQ(top_count, bottom[blob_id]->count());
			}
		}
		// First, figure out what blobs we need to check against, and zero init
		// parameter blobs.
		vector<Blob<Dtype>*> blobs_to_check;
		vector<bool> propagate_down(bottom.size(), check_bottom == -1);
		for (int i = 0; i < layer->blobs().size(); ++i) {
			Blob<Dtype>* blob = layer->blobs()[i].get();
			caffe_set(blob->count(), static_cast<Dtype>(0), blob->mutable_cpu_diff());
			blobs_to_check.push_back(blob);
		}
		if (check_bottom == -1) {
			for (int i = 0; i < bottom.size(); ++i) {
				blobs_to_check.push_back(bottom[i]);
			}
		}
		else if (check_bottom >= 0) {
			CHECK_LT(check_bottom, bottom.size());
			blobs_to_check.push_back(bottom[check_bottom]);
			propagate_down[check_bottom] = true;
		}
		CHECK_GT(blobs_to_check.size(), 0) << "No blobs to check.";
		// Compute the hebbian weights analytically using Backward
		Caffe::set_random_seed(seed_);
		// zero out feedback
		for (int i = 0; i < bottom.size(); ++i) {
			caffe_set(bottom[i]->count(), static_cast<Dtype>(0), bottom[i]->mutable_cpu_diff());
		}
		// Ignore the loss from the layer (it's just the weighted sum of the losses
		// from the top blobs, whose hebbian weights we may want to test individually).
		layer->Forward(bottom, top);
		// Winner is based on forward result.
		SetFeedbackWinner(*layer, top, top_id, top_data_id);
		layer->Backward(top, propagate_down, bottom);
		// Store computed hebbian weights for all checked blobs
		vector<shared_ptr<Blob<Dtype> > >
			computed_hebbian_blobs(blobs_to_check.size());
		for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
			Blob<Dtype>* current_blob = blobs_to_check[blob_id];
			computed_hebbian_blobs[blob_id].reset(new Blob<Dtype>());
			computed_hebbian_blobs[blob_id]->ReshapeLike(*current_blob);
			const int count = blobs_to_check[blob_id]->count();
			const Dtype* diff = blobs_to_check[blob_id]->cpu_diff();
			Dtype* computed_hebbians =
				computed_hebbian_blobs[blob_id]->mutable_cpu_data();
			caffe_copy(count, diff, computed_hebbians);
		}
		// zero out feedback (feedback values or "computed hebbian" have been saved at this point)
		//for (int i = 0; i < bottom.size(); ++i) {
		//	caffe_set(bottom[i]->count(), static_cast<Dtype>(0), bottom[i]->mutable_cpu_diff());
		//}

		SetFeedbackBuffer(layer, top, bottom, top_id, top_data_id);

		// Most feedback to input should be zero because only the winning conv node is allowed
		// to feedback its weights. We enumerate the input features, checking that each value
		// is either zerro (for most of them) or equal to the weight of a node that lands on that
		// feature.
		// LOG(ERROR) << "Checking " << blobs_to_check.size() << " blobs.";
		for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
			Blob<Dtype>* current_blob = blobs_to_check[blob_id];
			const Dtype* computed_hebbians =
				computed_hebbian_blobs[blob_id]->cpu_data();
			// LOG(ERROR) << "Blob " << blob_id << ": checking "
			//     << current_blob->count() << " parameters.";
			for (int feat_id = 0; feat_id < current_blob->count(); ++feat_id) {
				Dtype estimated_hebbian = GetLossUsingOjasRule(layer, top, bottom, current_blob, top_id, top_data_id, feat_id);
				Dtype computed_hebbian = computed_hebbians[feat_id];
				Dtype feature = current_blob->cpu_data()[feat_id];
				// LOG(ERROR) << "debug: " << current_blob->cpu_data()[feat_id] << " "
				//     << current_blob->cpu_diff()[feat_id];
				if (kink_ - kink_range_ > fabs(feature)
					|| fabs(feature) > kink_ + kink_range_) {
					// We check relative accuracy, but for too small values, we threshold
					// the scale factor by 1.
					Dtype scale = std::max<Dtype>(
						std::max(fabs(computed_hebbian), fabs(estimated_hebbian)),
						Dtype(1.));
					EXPECT_NEAR(computed_hebbian, estimated_hebbian, threshold_ * scale)
						<< "debug: (top_id, top_data_id, blob_id, feat_id)="
						<< top_id << "," << top_data_id << "," << blob_id << "," << feat_id
						<< "; feat = " << feature;
				}
				// LOG(ERROR) << "Feature: " << current_blob->cpu_data()[feat_id];
				// LOG(ERROR) << "computed hebbian: " << computed_hebbian
				//    << " estimated_hebbian: " << estimated_hebbian;
			}
		}
		if (bottomMem_ != nullptr)
		{
			free(bottomMem_);
			bottomMem_ = nullptr;
		}
	}

	template <typename Dtype>
	void HebbianChecker<Dtype>::CheckHebbianExhaustive(Layer<Dtype>* layer,
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int check_bottom) {
		layer->SetUp(bottom, top);
		CHECK_GT(top.size(), 0) << "Exhaustive mode requires at least one top blob.";
		// LOG(ERROR) << "Exhaustive Mode.";
		for (int i = 0; i < top.size(); ++i) {
			// LOG(ERROR) << "Exhaustive: blob " << i << " size " << top[i]->count();
			for (int j = 0; j < top[i]->count(); ++j) {
				// LOG(ERROR) << "Exhaustive: blob " << i << " data " << j;
				CheckHebbianSingle(layer, bottom, top, check_bottom, i, j);
			}
		}
	}

	template <typename Dtype>
	void HebbianChecker<Dtype>::CheckHebbianEltwise(Layer<Dtype>* layer,
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		layer->SetUp(bottom, top);
		CHECK_GT(top.size(), 0) << "Eltwise mode requires at least one top blob.";
		const int check_bottom = -1;
		const bool element_wise = true;
		for (int i = 0; i < top.size(); ++i) {
			for (int j = 0; j < top[i]->count(); ++j) {
				CheckHebbianSingle(layer, bottom, top, check_bottom, i, j, element_wise);
			}
		}
	}

	template <typename Dtype>
	void HebbianChecker<Dtype>::CheckHebbianNet(
		const Net<Dtype>& net, const vector<Blob<Dtype>*>& input) {
		const vector<shared_ptr<Layer<Dtype> > >& layers = net.layers();
		vector<vector<Blob<Dtype>*> >& bottom_vecs = net.bottom_vecs();
		vector<vector<Blob<Dtype>*> >& top_vecs = net.top_vecs();
		for (int i = 0; i < layers.size(); ++i) {
			net.Forward(input);
			LOG(ERROR) << "Checking hebbian for " << layers[i]->layer_param().name();
			CheckHebbianExhaustive(*(layers[i].get()), bottom_vecs[i], top_vecs[i]);
		}
	}

	// There should be 3 height x 3 width x 3 channels = 27 weights
	// and 2 nodes in output, for a total of 54 weights. 
	template <typename Dtype>
	void HebbianChecker<Dtype>::SetFeedbackWinner(const Layer<Dtype>& layer,
		const vector<Blob<Dtype>*>& top, int top_id, int top_data_id) {

		for (int i = 0; i < top.size(); i++)
		{
			Blob<Dtype>* top_blob = top[i];
			// only one top_diff should have a non-zero value
			Dtype* top_blob_diff = top_blob->mutable_cpu_diff();
			caffe_set(top_blob->count(), Dtype(0), top_blob_diff);

			// force top_data_id to win
			top_blob_diff[top_data_id] = 1.0;
		}
	}

	template <typename Dtype>
	void HebbianChecker<Dtype>::SetFeedbackBuffer(
		Layer<Dtype>* layer, const vector<Blob<Dtype>*>& topBVec, const vector<Blob<Dtype>*>& bottomBVec, int top_id, int top_data_id) {
		Blob<Dtype>* weightsBlob = layer->blobs()[0].get();
		const Blob<Dtype>* topBlob = topBVec[top_id];
		const vector<int>& wShape = weightsBlob->shape();
		const vector<int>& topShape = topBlob->shape();
		const Blob<Dtype>* botBlob = bottomBVec[0]; // assume that all bottom blobs are the same
		const vector<int>& botShape = botBlob->shape();
		const ConvolutionParameter& convParam = layer->layer_param().convolution_param();
		int ksize = convParam.kernel_size(0);
		int stride = convParam.stride(0);

		int ecount = botBlob->count(0, 4);
		CHECK_LE(bottomBVec.size(), 5);

			if (bottomMem_ == nullptr)
				bottomMem_ = (Dtype*)malloc(ecount * sizeof(Dtype));

			int loc = top_data_id % (topBlob->count(2, 4));
			int topx = stride * (loc % topShape[3]);
			int topy = stride * (loc / topShape[3]);
			int topChan = (top_data_id / topBlob->count(2, 4)) % topShape[1]; // conv filter index
			int wn = top_data_id / topBlob->count(1, 4);

			const Dtype* weights = weightsBlob->cpu_data();

			for (int n = 0; n < botShape[0]; n++)
			{
				int nloc = n * botBlob->count(1, 4);
				for (int chan = 0; chan < botShape[1]; chan++)
				{
					int cloc = nloc + chan * botBlob->count(2, 4);
					for (int y = 0; y < botShape[2]; y++)
					{
						int yloc = cloc + y * botShape[3];
						for (int x = 0; x < botShape[3]; x++)
						{
							if (wn == n && x >= topx && x < (topx + ksize) && y >= topy && y < (topy + ksize))
							{
								int kx = x - topx;
								int ky = y - topy;
								int wloc = topChan * weightsBlob->count(1, 4) + chan * weightsBlob->count(2, 4) + ky * wShape[3] + kx;
								bottomMem_[yloc + x] = weights[wloc];
							}
							else
								bottomMem_[yloc + x] = 0;
						}
					}
				}
			}
	}


// weightDiff = topDiff * (bottomData - topDiff * weight)
// determine whether blob to check is weightdiff or bottomdiff
template <typename Dtype>
Dtype HebbianChecker<Dtype>::GetLossUsingOjasRule(
		Layer<Dtype>* layer, const vector<Blob<Dtype>*>& topBVec, const vector<Blob<Dtype>*>& bottomBVec, const Blob<Dtype>* blobToCheck, int top_id, int top_data_id, int feature) {
	Blob<Dtype>* weightsBlob = layer->blobs()[0].get();
	Blob<Dtype>* biasBlob = layer->blobs()[1].get();
	Blob<Dtype>* topBlob = topBVec[top_id];
	Dtype topDiff = topBlob->cpu_diff()[top_data_id];
	const vector<int>& topShape = topBlob->shape();
	int topChan = (top_data_id / topBlob->count(2, 4)) % topShape[1]; // winning conv filter index

	// determine if blobToCheck is an input blob; if so, return with current value of loss
	if (blobToCheck == weightsBlob)
	{
		const vector<int>& wShape = weightsBlob->shape();
		int featChan = feature / weightsBlob->count(1, 4);  // feature channel being tested
		if (topChan != featChan)
			return 0.0;

		const ConvolutionParameter& convParam = layer->layer_param().convolution_param();
		int stride = convParam.stride(0);
		int toploc = top_data_id % (topBlob->count(2, 4));
		// determine upper-left corner location of node wrt to bottom dimensions
		int botx = stride * (toploc % topShape[3]);
		int boty = stride * ((toploc / topShape[3]) % topShape[2]);

		// determine location of feature wrt weights location
		int wx = feature % wShape[3];
		int wy = (feature / wShape[3]) % wShape[2];
		int warea = weightsBlob->count(2, 4);
		int ch = (feature / warea) % wShape[1];
		int topN = top_data_id / topBlob->count(1, 4);
		int ecount = bottomBVec[0]->count(0, 4);
		Dtype* inputMem = (Dtype*)malloc(ecount * sizeof(Dtype));

		int bottomArea = 0;
		const vector<int>& botShape = bottomBVec[0]->shape(); // assume that all bottom blobs have same shape

		for (int i = 0; i < ecount; i++)
		{
			inputMem[i] = 0;
		}

		for (int k = 0; k < bottomBVec.size(); k++)
		{
			const Blob<Dtype>* botBlob = bottomBVec[k];
			const Dtype* botData = botBlob->cpu_data();
			ecount = botBlob->count(0, 4);

			bottomArea = botBlob->count(2, 4);

			for (int i = 0; i < ecount; i++)
			{
				inputMem[i] += botData[i] - bottomMem_[i];
			}
		}
		return topDiff * inputMem[topN * bottomArea * botShape[1] + ch * bottomArea + (boty + wy) * botShape[3] + botx + wx];
	}
	else if (blobToCheck == biasBlob)
	{
		return (topChan == feature) ? bottomBVec.size() : 0;
	}
	else {
		// bottom diffs (feedback)
		Dtype singleWeight = bottomMem_[feature];
		Dtype loss = topDiff * singleWeight;
		return loss;
	}
}

}  // namespace caffe
