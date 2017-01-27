#ifndef CAFFE_HEBBIAN_SOLVERS_HPP_
#define CAFFE_HEBBIAN_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/hebbian_net.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class HebbianSolver : public Solver<Dtype> {
 public:
  explicit HebbianSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit HebbianSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "Hebbian"; }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

	void InitTrainNet();
	void InitTestNets();
	void Step(int iters);

	//virtual void ApplyUpdate();

 protected:
	void PreSolve();
	Dtype GetLearningRate();
	virtual void ApplyUpdate();
	virtual void Normalize(int param_id);
	virtual void Regularize(int param_id);
	virtual void ClipGradients();
	virtual void ComputeUpdateValue(int param_id, Dtype rate);
	virtual void SnapshotSolverState(const string& model_filename);
	virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
	virtual void SnapshotSolverStateToHDF5(const string& model_filename);
	virtual void RestoreSolverStateFromHDF5(const string& state_file);
	virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
	// history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(HebbianSolver);
};

}  // namespace caffe

#endif  // CAFFE_HEBBIAN_SOLVERS_HPP_
