#!/bin/bash
./build-aarch64-linux-gnu/install/bin/sherpa-onnx-offline-tts \
  --debug=true \
  --num-threads=16 \
  --provider=cpu \
  --qwen3-text-project=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/text_project.onnx \
  --qwen3-codec-embed=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/codec_embed.onnx \
  --qwen3-code-predictor-embed=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/code_predictor_embed.onnx \
  --qwen3-code-predictor=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/code_predictor.onnx \
  --qwen3-talker-prefill=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/talker_prefill.onnx \
  --qwen3-talker-decode=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/talker_decode.onnx \
  --qwen3-tokenizer12hz-decode=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/tokenizer12hz_decode.onnx \
  --qwen3-tokenizer-dir=/home/ncy/Workspace/Qwen3-TTS/Qwen3-TTS-ONNX-DLL/models/Qwen3-TTS-12Hz-0.6B-Base \
  --output-filename=./qwen3-out.wav \
  --non-streaming-mode=false \
  "为了守护蒙德城周边的安定，我曾经发动过不少次「远征」，但比起这一次，都算不上什么…比如清剿达达乌帕谷、联合千岩军扫荡石门、从鹰翔海滩出发迎击外海魔物…嗯？你说难怪在这些地方都遇不到什么强敌…我应该还是留了些下来给人练手的吧？"
  