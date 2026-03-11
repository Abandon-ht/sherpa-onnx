// sherpa-onnx/csrc/fire-red-vad-model.h
//
// Copyright (c)  2026  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_FIRE_RED_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_FIRE_RED_VAD_MODEL_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

class FireRedVadModel : public VadModel {
 public:
  explicit FireRedVadModel(const VadModelConfig &config);

  template <typename Manager>
  FireRedVadModel(Manager *mgr, const VadModelConfig &config);

  ~FireRedVadModel() override;

  // reset the internal model states
  void Reset() override;

  /**
   * @brief Process a single frame of features and return whether speech is
   * detected.
   *
   * @param features Pointer to a 1-D array containing FBank features for a
   * single frame. Shape: (feat_dim,), where feat_dim is from model metadata
   * (typically 80).
   * @param n Number of features. Must equal to FeatDim().
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  bool IsSpeech(const float *features, int32_t n) override;

  /**
   * @brief Compute VAD probabilities for the given feature frames.
   *
   * This is the main inference method that accepts pre-computed FBank features.
   *
   * @param features Pointer to a 2-D array of FBank features.
   *                 Shape: (num_frames, feat_dim), where feat_dim is from model
   * metadata.
   * @param num_frames Number of frames.
   * @param feat_dim Feature dimension (typically 80).
   *
   * @return A vector of probabilities, one per frame.
   */
  std::vector<float> ComputeProbs(const float *features, int32_t num_frames,
                                  int32_t feat_dim);

  /**
   * @brief Get the feature dimension used by the model.
   *
   * @return Feature dimension (typically 80).
   */
  int32_t FeatDim() const;

  float Compute(const float *samples, int32_t n) override;

  // Returns window size in samples
  int32_t WindowSize() const override;

  // Returns window shift in samples
  int32_t WindowShift() const override;

  int32_t MinSilenceDurationSamples() const override;
  int32_t MinSpeechDurationSamples() const override;

  void SetMinSilenceDuration(float s) override;
  void SetThreshold(float threshold) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FIRE_RED_VAD_MODEL_H_
