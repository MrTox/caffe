#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

// Load a solver.
int solver() {

  std::string solver_filename = "";

  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(solver_filename, &solver_param);

  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

//  solver->Restore(FLAGS_snapshot.c_str());

  return 0;
}

float mean(int N, const float* data) {
  float sum = 0;
  for(int i = 0; i < N; i++) {
    sum += data[i];
  }
  float mean = sum/N;
  return mean;
}

// net: Load a network
int net() {
  Caffe::set_mode(Caffe::CPU);

  string net_filename = "debug.prototxt";

  Net<float> net(net_filename, caffe::TRAIN);
//  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  float loss;
//  const vector<Blob<float>*>& result = net.Forward(&loss);
//  LOG(INFO) << "PAT loss: " << loss;

  net.Forward(&loss);
  LOG(INFO) << "PAT loss: " << loss;

  LOG(INFO) << "Backward...";
  net.Backward();

  std::cout << std::endl;

  for(int i = 0; i < net.blobs().size(); i++) {
    std::cout << "Blob " << net.blob_names()[i] << ": "
//        << "shape = " << net.blobs()[i]->shape_string()
        << "mean = " << mean(net.blobs()[i]->count(), net.blobs()[i]->cpu_data())
        << std::endl;
  }

  std::cout << std::endl;

  for(int i = 0; i < net.blobs().size(); i++) {
    std::cout << "Blob " << net.blob_names()[i] << ": "
        << "diff mean = " << mean(net.blobs()[i]->count(), net.blobs()[i]->cpu_diff())
        << std::endl;
  }

  std::cout << std::endl;

  LOG(INFO) << "Finished.";

  return 0;
}


int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: virtue <command> <args>\n\n"
      "commands:\n"
      "  solver          load a solver\n"
      "  net            load a network\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);

  net();
//  solver();

  return 0;
}
