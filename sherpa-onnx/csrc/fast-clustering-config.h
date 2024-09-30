// sherpa-onnx/csrc/fast-clustering-config.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FAST_CLUSTERING_CONFIG_H_
#define SHERPA_ONNX_CSRC_FAST_CLUSTERING_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct FastClusteringConfig {
  // If greater than 0, then threshold is ignored.
  //
  // We strongly recommend that you set it if you know the number of clusters
  // in advance
  int32_t num_clusters = -1;

  // distance threshold.
  //
  // The lower, the more clusters it will generate.
  // The higher, the fewer clusters it will generate.
  float threshold = 0.5;

  FastClusteringConfig() = default;

  FastClusteringConfig(int32_t num_clusters, float threshold)
      : num_clusters(num_clusters), threshold(threshold) {}

  std::string ToString() const;

  void Register(ParseOptions *po);
  bool Validate() const;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_FAST_CLUSTERING_CONFIG_H_
