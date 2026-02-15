#!/usr/bin/env python3
# python-api-examples/qwen3-tts.py
#
# Copyright (c)  2026  Xiaomi Corporation

"""
This file shows how to use sherpa-onnx Python API
for TTS with Qwen3-TTS.

Please download the Qwen3-TTS ONNX models first:

  https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

Usage:

  python3 ./python-api-examples/qwen3-tts.py
"""

import sherpa_onnx
import soundfile as sf

model_dir = "./qwen3-tts-0.6b-int8"

tts_config = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        qwen3=sherpa_onnx.OfflineTtsQwen3ModelConfig(
            text_project=f"{model_dir}/text_project.int8.onnx",
            codec_embed=f"{model_dir}/codec_embed.onnx",
            code_predictor_embed=f"{model_dir}/code_predictor_embed.onnx",
            code_predictor=f"{model_dir}/code_predictor.int8.onnx",
            talker_prefill=f"{model_dir}/talker_prefill.int8.onnx",
            talker_decode=f"{model_dir}/talker_decode.int8.onnx",
            speaker_encoder=f"{model_dir}/speaker_encoder.onnx",
            tokenizer12hz_encode=f"{model_dir}/tokenizer12hz_encode.onnx",
            tokenizer12hz_decode=f"{model_dir}/tokenizer12hz_decode.int8.onnx",
            tokenizer_dir=f"{model_dir}/tokenizer",
        ),
        num_threads=4,
        debug=True,
    ),
)

tts = sherpa_onnx.OfflineTts(tts_config)

text = "Today is a beautiful day. The sun is shining brightly."

audio = tts.generate(text, sid=0, speed=1.0)

filename = "generated-qwen3-tts.wav"
sf.write(filename, audio.samples, samplerate=audio.sample_rate)
print(f"Saved to {filename}")
print(f"Input text: {text}")
