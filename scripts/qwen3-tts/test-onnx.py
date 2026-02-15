#!/usr/bin/env python3
"""
Verify Qwen3-TTS ONNX inference pipeline step by step.
Compare with C++ implementation to find discrepancies.
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../qwen3-tts-0.6b-int8")


def load_session(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"  SKIP: {name} not found")
        return None
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    print(f"  Loaded {name}: inputs={[i.name for i in sess.get_inputs()]}, "
          f"outputs={[o.name for o in sess.get_outputs()]}")
    return sess


def tokenize_text(text, tokenizer_dir):
    """Return hardcoded token IDs from C++ debug output for test text."""
    # For text = "Today is a beautiful day. The sun is shining brightly."
    # With format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    # Verified from C++ FunASRNanoTokenizer output:
    return [151644, 77091, 198, 15364, 374, 264, 6233, 1899, 13, 576,
            7015, 374, 47925, 75289, 13, 151645, 198, 151644, 77091, 198]


def run_text_project(sess, input_ids):
    """Run text_project: input_ids [1, T] -> embeds [1, T, D]"""
    ids = np.array([input_ids], dtype=np.int64)
    result = sess.run(None, {"input_ids": ids})
    return result[0]  # [1, T, D]


def run_codec_embed(sess, input_ids):
    """Run codec_embed: input_ids [1, T] -> embeds [1, T, D]"""
    ids = np.array([input_ids], dtype=np.int64)
    result = sess.run(None, {"input_ids": ids})
    return result[0]  # [1, T, D]


def run_talker_prefill(sess, inputs_embeds, attention_mask):
    """Run talker_prefill."""
    result = sess.run(None, {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask.astype(np.int64),
    })
    # outputs: logits, last_hidden, kv_cache...
    output_names = [o.name for o in sess.get_outputs()]
    return {name: val for name, val in zip(output_names, result)}


def run_talker_decode(sess, inputs_embeds, attention_mask, kv_cache):
    """Run talker_decode with KV cache."""
    feeds = {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "attention_mask": attention_mask.astype(np.int64),
    }
    # Add KV cache inputs
    input_names = [i.name for i in sess.get_inputs()]
    kv_names = [n for n in input_names if n.startswith("past_key_") or n.startswith("past_value_")]
    for name in kv_names:
        # Map past_key_0 -> present_key_0 from prefill output
        present_name = name.replace("past_", "present_")
        if present_name in kv_cache:
            feeds[name] = kv_cache[present_name]

    result = sess.run(None, feeds)
    output_names = [o.name for o in sess.get_outputs()]
    return {name: val for name, val in zip(output_names, result)}


def run_code_predictor(sess, inputs_embeds, generation_step):
    """Run code_predictor."""
    result = sess.run(None, {
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "generation_step": np.array([generation_step], dtype=np.int64),
    })
    return result[0]  # logits


def run_code_predictor_embed(sess, input_ids, generation_step):
    """Run code_predictor_embed."""
    result = sess.run(None, {
        "input_ids": np.array([[input_ids]], dtype=np.int64),
        "generation_step": np.array([generation_step], dtype=np.int64),
    })
    return result[0]  # embeds [1, 1, D]


def sample_from_logits(logits, vocab_size, temperature=0.9, top_k=50, top_p=1.0,
                       suppress_start=-1, suppress_end=-1, suppress_exception=-1,
                       suppress_eos=False):
    """Sample a token from logits."""
    # Take last position if needed
    if logits.ndim == 3:
        logits = logits[0, -1, :]  # [V]
    elif logits.ndim == 2:
        logits = logits[0, :]  # [V]

    logits = logits[:vocab_size].copy()

    # Suppress tokens
    if suppress_start >= 0 and suppress_end > suppress_start:
        for i in range(suppress_start, min(suppress_end, len(logits))):
            if i != suppress_exception:
                logits[i] = -1e9

    if suppress_eos and 0 <= suppress_exception < len(logits):
        logits[suppress_exception] = -1e9

    # Temperature
    logits = logits / max(temperature, 1e-6)

    # Top-k
    if 0 < top_k < len(logits):
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -1e9

    # Softmax
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()

    # Top-p
    if 0 < top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, top_p) + 1
        mask = np.zeros_like(probs)
        mask[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]
        probs = mask / mask.sum()

    return np.random.choice(len(probs), p=probs)


def run_tokenizer12hz_decode(sess, audio_codes):
    """Run tokenizer12hz_decode: audio_codes [1, T, 16] -> audio [1, N], lengths [1]"""
    result = sess.run(None, {"audio_codes": audio_codes.astype(np.int64)})
    output_names = [o.name for o in sess.get_outputs()]
    print(f"  Decoder outputs: {[(n, v.shape) for n, v in zip(output_names, result)]}")
    return result


def main():
    print("=" * 60)
    print("Qwen3-TTS ONNX Verification")
    print("=" * 60)

    # Load config
    config_path = os.path.join(MODEL_DIR, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    talker_cfg = config["talker_config"]
    D = talker_cfg["hidden_size"]  # 1024
    num_code_groups = talker_cfg["num_code_groups"]  # 16
    talker_vocab_size = talker_cfg["vocab_size"]  # 3072
    cp_vocab_size = talker_cfg["code_predictor_config"]["vocab_size"]  # 2048

    tts_bos = config["tts_bos_token_id"]
    tts_eos = config["tts_eos_token_id"]
    tts_pad = config["tts_pad_token_id"]
    codec_bos = talker_cfg["codec_bos_id"]
    codec_eos = talker_cfg["codec_eos_token_id"]
    codec_pad = talker_cfg["codec_pad_id"]
    codec_nothink = talker_cfg["codec_nothink_id"]
    codec_think_bos = talker_cfg["codec_think_bos_id"]
    codec_think_eos = talker_cfg["codec_think_eos_id"]

    print(f"\nConfig: D={D}, num_code_groups={num_code_groups}")
    print(f"  talker_vocab_size={talker_vocab_size}, cp_vocab_size={cp_vocab_size}")
    print(f"  tts_bos={tts_bos}, tts_eos={tts_eos}, tts_pad={tts_pad}")
    print(f"  codec_bos={codec_bos}, codec_eos={codec_eos}, codec_pad={codec_pad}")

    # Load sessions
    print("\nLoading ONNX models...")
    text_project_sess = load_session("text_project_q.onnx")
    codec_embed_sess = load_session("codec_embed_q.onnx")
    cp_embed_sess = load_session("code_predictor_embed_q.onnx")
    cp_sess = load_session("code_predictor_q.onnx")
    talker_prefill_sess = load_session("talker_prefill_q.onnx")
    talker_decode_sess = load_session("talker_decode_q.onnx")
    decoder_sess = load_session("tokenizer12hz_decode_q.onnx")

    # Step 1: Tokenize
    text = "Today is a beautiful day. The sun is shining brightly."
    tokenizer_dir = os.path.join(MODEL_DIR, "tokenizer")
    token_ids = tokenize_text(text, tokenizer_dir)
    print(f"\nToken IDs ({len(token_ids)}): {token_ids}")

    # Step 2: Build prefill embeddings
    # Role prefix (first 3 tokens)
    role_ids = token_ids[:3]
    role_embed = run_text_project(text_project_sess, role_ids)  # [1, 3, D]
    print(f"\nRole embed shape: {role_embed.shape}, mean={role_embed.mean():.6f}")

    # Special tokens: tts_bos, tts_eos, tts_pad
    special_embed = run_text_project(text_project_sess, [tts_bos, tts_eos, tts_pad])
    tts_bos_embed = special_embed[:, 0:1, :]
    tts_eos_embed = special_embed[:, 1:2, :]
    tts_pad_embed = special_embed[:, 2:3, :]
    print(f"Special embeds: bos_mean={tts_bos_embed.mean():.6f}, "
          f"eos_mean={tts_eos_embed.mean():.6f}, pad_mean={tts_pad_embed.mean():.6f}")

    # Codec prefix: [nothink, think_bos, think_eos, pad, bos]
    codec_prefix_embed = run_codec_embed(codec_embed_sess,
                                          [codec_nothink, codec_think_bos, codec_think_eos,
                                           codec_pad, codec_bos])
    print(f"Codec prefix embed shape: {codec_prefix_embed.shape}")

    # Body text (tokens 3 to -5)
    body_ids = token_ids[3:-5]
    print(f"Body text IDs ({len(body_ids)}): {body_ids}")
    body_embed = run_text_project(text_project_sess, body_ids)  # [1, N, D]

    codec_pad_embed = run_codec_embed(codec_embed_sess, [codec_pad])  # [1, 1, D]

    # Streaming mode (reference default, non_streaming_mode=False):
    # Prefill: [role(3)] + [tts_pad+nothink, tts_pad+think_bos,
    #           tts_pad+think_eos, tts_bos+pad] + [text[3]+codec_bos]
    # = 3 + 4 + 1 = 8 positions
    # trailing_text_hidden: [text[4], ..., text[-6], tts_eos]

    n_body = len(body_ids)
    prefill_len = 3 + 3 + 1 + 1  # role + codec_prefix + tts_bos+pad + text[3]+codec_bos
    prefill = np.zeros((1, prefill_len, D), dtype=np.float32)

    pos = 0

    # Role prefix (3)
    prefill[0, pos:pos + 3, :] = role_embed[0, :3, :]
    pos += 3

    # tts_pad + codec_{nothink, think_bos, think_eos}
    for i in range(3):
        prefill[0, pos, :] = tts_pad_embed[0, 0, :] + codec_prefix_embed[0, i, :]
        pos += 1

    # tts_bos + codec_pad
    prefill[0, pos, :] = tts_bos_embed[0, 0, :] + codec_prefix_embed[0, 3, :]
    pos += 1

    # text[3] + codec_bos
    prefill[0, pos, :] = body_embed[0, 0, :] + codec_prefix_embed[0, 4, :]
    pos += 1

    # Build trailing_text_hidden: [text[4], ..., text[-6], tts_eos]
    trailing_text_hidden_list = []
    for i in range(1, n_body):
        trailing_text_hidden_list.append(body_embed[0, i, :])
    trailing_text_hidden_list.append(tts_eos_embed[0, 0, :])

    print(f"\nStreaming prefill: {prefill_len} positions, trailing_text: {len(trailing_text_hidden_list)} steps")
    print(f"Prefill embed stats: mean={prefill.mean():.6f}, std={prefill.std():.6f}")

    # Step 3: Run talker prefill
    attention_mask = np.ones((1, prefill_len), dtype=np.int64)
    print("\nRunning talker prefill...")
    prefill_result = run_talker_prefill(talker_prefill_sess, prefill, attention_mask)

    logits = prefill_result["logits"]
    last_hidden = prefill_result["last_hidden"]
    print(f"Prefill logits shape: {logits.shape}")
    print(f"Prefill last_hidden shape: {last_hidden.shape}")

    # Build KV cache dict
    kv_cache = {k: v for k, v in prefill_result.items()
                if k.startswith("present_")}
    print(f"KV cache entries: {len(kv_cache)}")

    # Step 4: Autoregressive generation
    max_new_tokens = 200
    suppress_start = talker_vocab_size - 1024  # 2048
    suppress_end = talker_vocab_size  # 3072
    temperature = 0.9  # default for Qwen3-TTS

    all_codes = []
    total_seq_len = prefill_len

    print(f"\nStarting generation (max {max_new_tokens} tokens, temp={temperature})...")
    for step in range(max_new_tokens):
        # Sample primary code
        primary_code = sample_from_logits(
            logits, talker_vocab_size, temperature=temperature, top_k=50, top_p=1.0,
            suppress_start=suppress_start, suppress_end=suppress_end,
            suppress_exception=codec_eos, suppress_eos=(step < 2))

        # Check EOS
        if primary_code == codec_eos:
            print(f"  EOS at step {step}")
            break

        if step < 5 or step % 50 == 0:
            # Debug: log eos logit
            if logits.ndim == 3:
                l = logits[0, -1, :talker_vocab_size]
            else:
                l = logits[0, :talker_vocab_size]
            eos_logit = l[codec_eos] if codec_eos < len(l) else -999
            top1 = np.argmax(l)
            print(f"  Step {step}: code={primary_code}, eos_logit={eos_logit:.3f}, "
                  f"top1={top1}({l[top1]:.3f})")

        # Get primary code embedding
        primary_embed = run_codec_embed(codec_embed_sess, [primary_code])  # [1, 1, D]

        # Run code predictor for residual codes
        frame_codes = [primary_code]

        # Build code predictor context: [last_hidden_last_pos, primary_embed]
        lh_last = last_hidden[:, -1:, :]  # [1, 1, D]
        cp_context = np.concatenate([lh_last, primary_embed], axis=1)  # [1, 2, D]

        codec_sum = primary_embed[0, 0, :].copy()  # [D]

        for j in range(num_code_groups - 1):
            # Run code predictor
            cp_logits = run_code_predictor(cp_sess, cp_context, j)

            # Sample residual code
            residual_code = sample_from_logits(
                cp_logits, cp_vocab_size, temperature=temperature, top_k=50, top_p=1.0)

            frame_codes.append(int(residual_code))

            # Get residual embedding
            res_embed = run_code_predictor_embed(cp_embed_sess, int(residual_code), j)

            # Append to context
            cp_context = np.concatenate([cp_context, res_embed], axis=1)

            # Add to codec sum
            codec_sum += res_embed[0, 0, :]

        all_codes.append(frame_codes)

        # Build next talker input: codec_sum + trailing_text_hidden[step]
        if step < len(trailing_text_hidden_list):
            text_hidden = trailing_text_hidden_list[step]
        else:
            text_hidden = tts_pad_embed[0, 0, :]
        next_input = codec_sum + text_hidden
        next_embeds = next_input.reshape(1, 1, D)

        # Growing attention mask
        total_seq_len += 1
        new_mask = np.ones((1, total_seq_len), dtype=np.int64)

        # Run talker decode
        decode_result = run_talker_decode(talker_decode_sess, next_embeds, new_mask, kv_cache)

        logits = decode_result["logits"]
        last_hidden = decode_result["last_hidden"]
        kv_cache = {k: v for k, v in decode_result.items()
                    if k.startswith("present_")}

        if (step + 1) % 50 == 0:
            print(f"  Generated {step + 1} frames...")

    print(f"\nGeneration complete: {len(all_codes)} frames")

    if not all_codes:
        print("No frames generated!")
        return

    # Step 5: Decode to audio
    audio_codes = np.array([all_codes], dtype=np.int64)  # [1, T, 16]
    print(f"Audio codes shape: {audio_codes.shape}")
    print(f"First frame codes: {all_codes[0]}")
    if len(all_codes) > 1:
        print(f"Second frame codes: {all_codes[1]}")

    results = run_tokenizer12hz_decode(decoder_sess, audio_codes)
    audio_values = results[0]
    print(f"Audio values shape: {audio_values.shape}")

    if len(results) > 1:
        lengths = results[1]
        print(f"Lengths: {lengths}")
        num_samples = int(lengths.flatten()[0])
    else:
        num_samples = audio_values.size

    audio = audio_values.flatten()[:num_samples]
    print(f"Output: {num_samples} samples ({num_samples / 24000:.2f} seconds)")

    # Save
    output_path = os.path.join(os.path.dirname(__file__), "../../generated-qwen3-tts-python.wav")
    sf.write(output_path, audio, 24000)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
