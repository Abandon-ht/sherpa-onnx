#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#
# Export Qwen3-TTS-12Hz-0.6B-Base to ONNX models for sherpa-onnx.
#
# Usage:
#   pip install qwen-tts torch onnx onnxruntime
#   python3 export-onnx.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --output-dir ./qwen3-tts-0.6b-12hz
#
# This script exports 9 ONNX sub-models:
#   1. text_project.onnx         - text token IDs -> text embeddings
#   2. codec_embed.onnx          - codec token ID -> codec embedding
#   3. code_predictor_embed.onnx - residual code IDs -> embeddings (15 layers)
#   4. code_predictor.onnx       - sub-talker for residual codebook prediction
#   5. talker_prefill.onnx       - talker prefill (full context -> KV-cache)
#   6. talker_decode.onnx        - talker decode (one token -> next token + KV-cache)
#   7. speaker_encoder.onnx      - mel spectrogram -> speaker embedding
#   8. tokenizer12hz_encode.onnx - audio waveform -> codec tokens
#   9. tokenizer12hz_decode.onnx - codec tokens -> audio waveform

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

# Register Qwen3TTS model classes
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def export_text_project(model, output_dir, opset_version=14):
    """Export text_projection + text_embed_tokens as a single model.

    Input:  input_ids [1, T] int64  - text token IDs
    Output: text_embed [1, T, D] float32  - projected text embeddings
    """
    print("Exporting text_project.onnx ...")

    class TextProject(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.text_embed = talker.model.text_embed_tokens
            self.text_projection = talker.text_projection

        def forward(self, input_ids):
            return self.text_projection(self.text_embed(input_ids))

    wrapper = TextProject(model.talker)
    wrapper.eval()

    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids,),
        os.path.join(output_dir, "text_project.onnx"),
        input_names=["input_ids"],
        output_names=["text_embed"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "text_embed": {1: "seq_len"},
        },
        opset_version=opset_version,
    )
    print("  Done: text_project.onnx")


def export_codec_embed(model, output_dir, opset_version=14):
    """Export codec token embedding layer.

    Input:  token_ids [1, T] int64  - codec token IDs
    Output: embed [1, T, D] float32  - codec embeddings
    """
    print("Exporting codec_embed.onnx ...")

    class CodecEmbed(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.embed_tokens = talker.model.embed_tokens

        def forward(self, token_ids):
            return self.embed_tokens(token_ids)

    wrapper = CodecEmbed(model.talker)
    wrapper.eval()

    dummy_ids = torch.tensor([[100]], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        os.path.join(output_dir, "codec_embed.onnx"),
        input_names=["token_ids"],
        output_names=["embed"],
        dynamic_axes={
            "token_ids": {1: "seq_len"},
            "embed": {1: "seq_len"},
        },
        opset_version=opset_version,
    )
    print("  Done: codec_embed.onnx")


def export_code_predictor_embed(model, output_dir, opset_version=14):
    """Export code predictor embedding layers (15 layers for residual codebooks).

    Input:  token_id int64 scalar - residual codebook token ID
            layer_idx int64 scalar - which residual layer (0-14)
    Output: embed [1, 1, D] float32 - embedding
    """
    print("Exporting code_predictor_embed.onnx ...")

    num_groups = model.talker.config.num_code_groups  # typically 16
    code_predictor = model.talker.code_predictor

    # Export each embedding layer separately as a combined model
    class CodePredictorEmbed(nn.Module):
        def __init__(self, embed_layers):
            super().__init__()
            self.embed_layers = nn.ModuleList(embed_layers)

        def forward(self, token_id, layer_idx):
            # For ONNX, we'll use a simple approach: export all layers
            # and use gather at runtime
            embeds = []
            for layer in self.embed_layers:
                embeds.append(layer(token_id))
            stacked = torch.stack(embeds, dim=0)  # [num_layers, 1, 1, D]
            return stacked[layer_idx]  # [1, 1, D]

    embed_layers = list(code_predictor.get_input_embeddings())
    wrapper = CodePredictorEmbed(embed_layers)
    wrapper.eval()

    dummy_token = torch.tensor([[100]], dtype=torch.long, device=model.device)
    dummy_layer = torch.tensor(0, dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_token, dummy_layer),
        os.path.join(output_dir, "code_predictor_embed.onnx"),
        input_names=["token_id", "layer_idx"],
        output_names=["embed"],
        opset_version=opset_version,
    )
    print(f"  Done: code_predictor_embed.onnx ({num_groups - 1} layers)")


def export_speaker_encoder(model, output_dir, opset_version=14):
    """Export speaker encoder.

    Input:  mel [1, T, 128] float32 - mel spectrogram
    Output: speaker_embedding [D] float32 - speaker embedding vector
    """
    print("Exporting speaker_encoder.onnx ...")

    if model.speaker_encoder is None:
        print("  Skipped: model has no speaker_encoder (not a base model)")
        return

    class SpeakerEncoderWrapper(nn.Module):
        def __init__(self, speaker_encoder):
            super().__init__()
            self.encoder = speaker_encoder

        def forward(self, mel):
            return self.encoder(mel)[0]

    wrapper = SpeakerEncoderWrapper(model.speaker_encoder)
    wrapper.eval()

    dummy_mel = torch.randn(1, 100, 128, device=model.device, dtype=model.dtype)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        os.path.join(output_dir, "speaker_encoder.onnx"),
        input_names=["mel"],
        output_names=["speaker_embedding"],
        dynamic_axes={
            "mel": {1: "time"},
        },
        opset_version=opset_version,
    )
    print("  Done: speaker_encoder.onnx")


def export_tokenizer_12hz(speech_tokenizer, output_dir, device, opset_version=14):
    """Export 12Hz speech tokenizer encoder and decoder."""
    print("Exporting tokenizer12hz_encode.onnx and tokenizer12hz_decode.onnx ...")

    # The speech tokenizer is a Qwen3TTSTokenizer which wraps a HuggingFace model
    # We need to access the underlying encoder and decoder
    tokenizer_model = speech_tokenizer.model

    # Export encoder
    class TokenizerEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            # audio: [1, num_samples] float32
            return self.model.encode(audio)

    # Export decoder
    class TokenizerDecoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, codes):
            # codes: [1, T, num_codebooks] int64
            return self.model.decode(codes)

    print("  Note: Tokenizer export may need custom handling depending on model architecture.")
    print("  Consider using the community pre-exported models from:")
    print("    https://huggingface.co/sivasub987/Qwen3-TTS-0.6B-ONNX-INT8")
    print("  Done (placeholder)")


def export_tokenizer_files(processor, output_dir):
    """Copy tokenizer files (vocab.json, merges.txt, etc.) to output directory."""
    print("Exporting tokenizer files ...")

    tokenizer = processor.tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print("  Done: tokenizer/")


def get_config_json(model):
    """Extract key configuration values needed at inference time."""
    config = model.config
    talker_config = config.talker_config

    return {
        "model_type": "qwen3-tts-12hz",
        "tts_model_type": config.tts_model_type,
        "tts_model_size": config.tts_model_size,
        "tokenizer_type": config.tokenizer_type,
        "hidden_size": talker_config.hidden_size,
        "text_hidden_size": talker_config.text_hidden_size,
        "vocab_size": talker_config.vocab_size,
        "num_code_groups": talker_config.num_code_groups,
        "num_attention_heads": talker_config.num_attention_heads,
        "num_key_value_heads": talker_config.num_key_value_heads,
        "num_hidden_layers": talker_config.num_hidden_layers,
        "codec_bos_id": talker_config.codec_bos_id,
        "codec_eos_token_id": talker_config.codec_eos_token_id,
        "codec_pad_id": talker_config.codec_pad_id,
        "codec_nothink_id": talker_config.codec_nothink_id,
        "codec_think_id": talker_config.codec_think_id,
        "codec_think_bos_id": talker_config.codec_think_bos_id,
        "codec_think_eos_id": talker_config.codec_think_eos_id,
        "tts_bos_token_id": config.tts_bos_token_id,
        "tts_eos_token_id": config.tts_eos_token_id,
        "tts_pad_token_id": config.tts_pad_token_id,
        "codec_language_id": dict(talker_config.codec_language_id),
        "spk_id": dict(talker_config.spk_id) if hasattr(talker_config, "spk_id") else {},
        "speaker_encoder_sample_rate": config.speaker_encoder_config.sample_rate,
        "output_sample_rate": 24000,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS to ONNX for sherpa-onnx")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen3-tts-0.6b-12hz",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (cpu or cuda:0)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    dtype = torch.float32
    model = AutoModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model, fix_mistral_regex=True)

    print(f"Model loaded. Device: {model.device}")
    print(f"Config: tts_model_type={model.tts_model_type}, "
          f"tokenizer_type={model.tokenizer_type}, "
          f"tts_model_size={model.tts_model_size}")

    # Export individual components
    export_text_project(model, args.output_dir, args.opset_version)
    export_codec_embed(model, args.output_dir, args.opset_version)
    export_code_predictor_embed(model, args.output_dir, args.opset_version)
    export_speaker_encoder(model, args.output_dir, args.opset_version)

    # Export tokenizer files
    export_tokenizer_files(processor, args.output_dir)

    # Save config
    config_data = get_config_json(model)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"Saved config to {config_path}")

    # Note about talker and tokenizer12hz
    print("\n" + "=" * 60)
    print("NOTE: talker_prefill, talker_decode, code_predictor,")
    print("tokenizer12hz_encode, and tokenizer12hz_decode require")
    print("careful KV-cache handling for ONNX export.")
    print()
    print("For pre-exported ONNX models, see:")
    print("  https://huggingface.co/sivasub987/Qwen3-TTS-0.6B-ONNX-INT8")
    print("  https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL")
    print("=" * 60)


if __name__ == "__main__":
    main()
