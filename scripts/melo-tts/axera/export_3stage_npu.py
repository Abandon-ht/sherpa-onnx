#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_3stage_npu.py - MeloTTS 三阶段切分导出脚本（最大化 NPU 利用率）

切分方案：
  1. encoder.onnx    - 文本 → z_p  (CPU 推理，动态 shape)
  2. flow.onnx       - z_p → z     (NPU 推理，固定 shape, dec_len=128)
  3. generator.onnx  - z  → audio  (NPU 推理，固定 shape, dec_len=128)

相比原始两阶段方案（encoder + flow_dec），三阶段方案将 decoder 的
flow 网络和 HiFiGAN generator 分别导出，以便：
  - 对两个子模块分别进行精度验证
  - 支持 flow 和 generator 使用不同量化策略
  - 理论上可以在支持并行的 NPU 上并行执行（如 AX650N 双核）

使用方法：
  conda activate cosyvoice
  cd /home/m5stack/Workspace/AXERA/melotts.axera/model_convert
  python export_3stage_npu.py -l ZH -d 128

输入参数：
  -l/--language    目标语言 (ZH, EN, JP, KR, FR, ES)
  -d/--dec_len     decoder 固定输入长度，默认 128

依赖：
  - melotts (本目录下 melotts/ 包)
  - torch, onnx, onnxsim
  - conda activate cosyvoice
"""

import argparse
import os
import sys
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import onnx
    import onnxsim
except ImportError:
    print("请先安装: pip install onnx onnxsim")
    sys.exit(1)

# ─── 添加 melotts 模块路径 ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from melotts.download_utils import load_or_download_config, load_or_download_model
from melotts.tts import TTS
from melotts.models import SynthesizerTrn

# ─── 语言文本示例（用于校准数据生成） ────────────────────────────────────────
TEXT = {
    "ZH": "爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。",
    "JP": "海の向こうには何があるの？",
    "EN": "Did you ever hear a folk tale about a giant turtle?",
    "KR": "한국 음식을 먹어보고 싶어요. 불고기랑 김치찌개가 제가 좋아하는 음식이에요.",
    "FR": "Les cafés animés résonnent de conversations passionnées.",
    "ES": "El susurro suave del viento atraviesa los campos de lavanda.",
}


# ─── 1. Encoder Wrapper ───────────────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    """
    封装 enc_forward，与 AXERA convert.py 接口保持一致。

    输入:
        phone    int32[phone_len]   - 音素 ID（无 batch 维度）
        tone     int32[phone_len]   - 声调
        language int32[phone_len]   - 语言 ID
        g        float32[1,256,1]   - 说话人嵌入

        noise_scale   float32[1]
        noise_scale_w float32[1]
        length_scale  float32[1]
        sdp_ratio     float32[1]

    输出:
        z_p          float32[1, 192, T]  - 潜在表示（动态长度）
        pronoun_lens int32[phone_len]    - 每个音素的发音帧数
        audio_len    int32[1]            - 总音频采样数
    """
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, phone, tone, language, g,
                noise_scale, noise_scale_w, length_scale, sdp_ratio):
        return self.model.enc_forward(
            phone, tone, language, g,
            noise_scale=noise_scale[0],
            noise_scale_w=noise_scale_w[0],
            length_scale=length_scale[0],
            sdp_ratio=sdp_ratio[0],
        )


# ─── 2. Flow Wrapper（独立 flow 网络） ────────────────────────────────────────

class FlowWrapper(nn.Module):
    """
    封装 flow 网络（TransformerCouplingBlock），固定输入形状。

    输入:
        z_p   float32[1, 192, dec_len]  - 固定形状（NPU 友好）
        g     float32[1, 256, 1]         - 说话人嵌入

    输出:
        z     float32[1, 192, dec_len]  - 变换后的潜在变量
    """
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, z_p, g):
        # y_mask 全 1（因为 z_p 是完整的，padding 部分已在 Python 端处理）
        y_mask = torch.ones(1, 1, z_p.size(-1), dtype=torch.float32)
        z = self.model.flow(z_p, y_mask, g, reverse=True)
        return z * y_mask


# ─── 3. Generator Wrapper（HiFiGAN 声码器） ───────────────────────────────────

class GeneratorWrapper(nn.Module):
    """
    封装 HiFiGAN Generator，固定输入形状。

    输入:
        z     float32[1, 192, dec_len]   - 固定形状（NPU 友好）
        g     float32[1, 256, 1]          - 说话人嵌入

    输出:
        audio float32[1, 1, dec_len*512] - 生成的音频（65536 when dec_len=128）

    注: dec_len * upsample_factor = dec_len * 512 (8*8*4*2=512)
    """
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(self, z, g):
        return self.model.dec(z, g=g)


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def simplify_and_save(model_path: str) -> None:
    """ONNX 简化并原地保存。"""
    print(f"  Simplifying {os.path.basename(model_path)} ...")
    model = onnx.load(model_path)
    sim_model, check = onnxsim.simplify(model)
    if not check:
        print(f"  Warning: simplification check failed for {model_path}")
    onnx.save(sim_model, model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"  Saved {model_path} ({size_mb:.1f} MB)")


def get_upsample_factor(model: SynthesizerTrn) -> int:
    """计算 HiFiGAN 的总上采样倍率。"""
    factor = 1
    for r in model.dec.ups:
        # ConvTranspose1d stride == upsample_rate
        factor *= r.stride[0]
    return factor


# ─── 导出函数 ─────────────────────────────────────────────────────────────────

def export_encoder(tts: TTS, out_path: str, language: str) -> None:
    """
    导出 encoder.onnx（与 AXERA convert.py 兼容，动态 phone_len）。
    接口与 AXERA encoder 完全一致，可直接替换。
    """
    model = tts.model
    wrapper = EncoderWrapper(model)

    phone_len = 128  # 导出时的示例长度
    phones   = torch.zeros(phone_len, dtype=torch.int32)
    tones    = torch.randint(0, 5, (phone_len,), dtype=torch.int32)
    language_ids = torch.zeros(phone_len, dtype=torch.int32) + 3
    g        = torch.rand(1, 256, 1)

    noise_scale   = torch.FloatTensor([0.667])
    noise_scale_w = torch.FloatTensor([0.8])
    length_scale  = torch.FloatTensor([1.0])
    sdp_ratio     = torch.FloatTensor([0.2])

    inputs = (phones, tones, language_ids, g,
              noise_scale, noise_scale_w, length_scale, sdp_ratio)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            out_path,
            opset_version=16,
            export_params=True,
            do_constant_folding=True,
            input_names=["phone", "tone", "language", "g",
                         "noise_scale", "noise_scale_w", "length_scale", "sdp_ratio"],
            output_names=["z_p", "pronoun_lens", "audio_len"],
            dynamic_axes={
                "phone":    {0: "phone_len"},
                "tone":     {0: "phone_len"},
                "language": {0: "phone_len"},
                # z_p 的 T 维是动态的（由 duration predictor 决定）
                "z_p":      {2: "audio_len"},
                "pronoun_lens": {0: "phone_len"},
            },
        )

    simplify_and_save(out_path)
    print(f"  [Encoder] Input:  phone/tone/language int32[phone_len], g float32[1,256,1], 4x scales")
    print(f"  [Encoder] Output: z_p float32[1,192,T], pronoun_lens int32[phone_len], audio_len int32[1]")


def export_flow(tts: TTS, out_path: str, dec_len: int) -> None:
    """
    导出 flow.onnx（固定形状，NPU 友好）。
    独立导出 flow 网络（TransformerCouplingBlock），与 generator 分开。
    """
    model = tts.model
    wrapper = FlowWrapper(model)

    z_p = torch.rand(1, 192, dec_len)
    g   = torch.rand(1, 256, 1)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (z_p, g),
            out_path,
            opset_version=16,
            export_params=True,
            do_constant_folding=True,
            input_names=["z_p", "g"],
            output_names=["z"],
            # 固定形状，不设置 dynamic_axes
        )

    simplify_and_save(out_path)
    print(f"  [Flow]    Input:  z_p float32[1,192,{dec_len}], g float32[1,256,1]")
    print(f"  [Flow]    Output: z   float32[1,192,{dec_len}]")


def export_generator(tts: TTS, out_path: str, dec_len: int) -> None:
    """
    导出 generator.onnx（固定形状，NPU 友好）。
    独立导出 HiFiGAN generator。
    """
    model = tts.model
    wrapper = GeneratorWrapper(model)

    upsample = get_upsample_factor(model)
    audio_len = dec_len * upsample

    z = torch.rand(1, 192, dec_len)
    g = torch.rand(1, 256, 1)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (z, g),
            out_path,
            opset_version=16,
            export_params=True,
            do_constant_folding=True,
            input_names=["z", "g"],
            output_names=["audio"],
            # 固定形状，不设置 dynamic_axes
        )

    simplify_and_save(out_path)
    print(f"  [Generator] Input:  z float32[1,192,{dec_len}], g float32[1,256,1]")
    print(f"  [Generator] Output: audio float32[1,1,{audio_len}]")


def export_g_bin(tts: TTS, speaker_id: int, out_path: str) -> None:
    """
    导出说话人嵌入向量为二进制文件（与 AXERA 兼容）。
    """
    model = tts.model
    with torch.no_grad():
        g = model.emb_g(torch.IntTensor([speaker_id])).unsqueeze(-1)
        g.numpy().astype(np.float32).tofile(out_path)
    print(f"  [g.bin]   Saved speaker {speaker_id} embedding → {out_path}")
    print(f"             Shape: {g.shape}, dtype: float32")


def export_flow_dec_combined(tts: TTS, out_path: str, dec_len: int) -> None:
    """
    导出 combined flow+decoder.onnx（与 AXERA convert.py 原始 decoder 完全兼容）。
    此为参考实现，与三阶段方案中的 flow.onnx + generator.onnx 功能等价。
    """
    model = tts.model

    dec_len_val = dec_len
    z_p = torch.rand(1, 192, dec_len_val)
    g   = torch.rand(1, 256, 1)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (z_p, g),
            out_path,
            opset_version=16,
            export_params=True,
            do_constant_folding=True,
            input_names=["z_p", "g"],
            output_names=["audio"],
        )
        # 覆写 forward 为 flow_dec_forward
        # 注意：export 前需要临时设置 forward

    simplify_and_save(out_path)


def generate_calibration_data(tts: TTS, speaker_id: int, text: str,
                               language: str, out_dir: str, dec_len: int) -> None:
    """
    生成三阶段模型各自需要的校准数据：
    - encoder_calib/: phone, tone, lang, g, scale 样本
    - flow_calib/: z_p 样本（来自 encoder 输出）
    - generator_calib/: z 样本（来自 flow 输出）
    """
    import tarfile
    from datetime import datetime

    print(f"  Generating calibration data for {language} ...")

    os.makedirs(f"{out_dir}/encoder_calib/phone", exist_ok=True)
    os.makedirs(f"{out_dir}/encoder_calib/g", exist_ok=True)
    os.makedirs(f"{out_dir}/flow_calib/z_p", exist_ok=True)
    os.makedirs(f"{out_dir}/flow_calib/g", exist_ok=True)
    os.makedirs(f"{out_dir}/generator_calib/z", exist_ok=True)
    os.makedirs(f"{out_dir}/generator_calib/g", exist_ok=True)

    model = tts.model
    from melotts.tts import get_text_for_tts_infer

    # 获取文本的 phone/tone/lang
    _, _, phones, tones, lang_ids = get_text_for_tts_infer(
        text, tts.language, tts.hps, "cpu", tts.symbol_to_id
    )

    with torch.no_grad():
        g = model.emb_g(torch.IntTensor([speaker_id])).unsqueeze(-1)  # [1,256,1]

        # Encoder 推理得到 z_p
        z_p, pronoun_lens, audio_len = model.enc_forward(
            phones.int(), tones.int(), lang_ids.int(), g,
            noise_scale=0.667, noise_scale_w=0.8,
            length_scale=1.0, sdp_ratio=0.2
        )

        # 保存 encoder 校准样本
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(f"{out_dir}/encoder_calib/phone/{ts}.npy", phones.int().numpy())
        np.save(f"{out_dir}/encoder_calib/g/{ts}.npy", g.numpy())

        # z_p 切片 → flow 校准样本
        n_slices = int(np.ceil(z_p.size(-1) / dec_len))
        for i in range(n_slices):
            z_p_slice = z_p[..., i*dec_len:(i+1)*dec_len]
            if z_p_slice.size(-1) < dec_len:
                z_p_slice = F.pad(z_p_slice, (0, dec_len - z_p_slice.size(-1)))

            np.save(f"{out_dir}/flow_calib/z_p/{ts}_{i:04d}.npy", z_p_slice.numpy())
            np.save(f"{out_dir}/flow_calib/g/{ts}_{i:04d}.npy", g.numpy())

            # flow → z → generator 校准样本
            y_mask = torch.ones(1, 1, dec_len, dtype=torch.float32)
            z = model.flow(z_p_slice, y_mask, g, reverse=True)
            z = z * y_mask

            np.save(f"{out_dir}/generator_calib/z/{ts}_{i:04d}.npy", z.numpy())
            np.save(f"{out_dir}/generator_calib/g/{ts}_{i:04d}.npy", g.numpy())

    # 打包为 tar.gz
    for sub, names in [
        ("flow_calib", ["z_p", "g"]),
        ("generator_calib", ["z", "g"]),
    ]:
        for name in names:
            folder = f"{out_dir}/{sub}/{name}"
            tar_path = f"{out_dir}/{sub}/{name}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                for f in sorted(os.listdir(folder)):
                    if f.endswith(".npy"):
                        tar.add(os.path.join(folder, f), arcname=f)
            print(f"  Packed calibration: {tar_path}")


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="MeloTTS 三阶段切分导出（最大化 NPU 利用率）"
    )
    parser.add_argument(
        "-l", "--language", type=str, default="ZH",
        choices=["ZH", "EN", "JP", "KR", "FR", "ES"],
        help="目标语言"
    )
    parser.add_argument(
        "-d", "--dec_len", type=int, default=128,
        help="Decoder (flow & generator) 固定输入长度"
    )
    parser.add_argument(
        "--out-dir", type=str, default=".",
        help="输出目录"
    )
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="模型配置文件路径"
    )
    parser.add_argument(
        "--ckpt", type=str, default="checkpoint.pth",
        help="模型权重文件路径"
    )
    parser.add_argument(
        "--calib", action="store_true",
        help="同时生成校准数据集"
    )
    parser.add_argument(
        "--skip-simplify", action="store_true",
        help="跳过 onnxsim 简化步骤（加快导出速度）"
    )
    parser.add_argument(
        "--export-combined-decoder", action="store_true",
        help="同时导出原始合并版 decoder.onnx（兼容 AXERA convert.py）"
    )
    return parser.parse_args()


def main():
    args = get_args()
    language = args.language
    lower_lang = language.lower()
    dec_len = args.dec_len
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ─── 加载模型 ─────────────────────────────────────────────────────────────
    config_path = args.config
    ckpt_path   = args.ckpt

    if not os.path.exists(config_path):
        print(f"Downloading config for {language}...")
        load_or_download_config(locale=language)

    with open(config_path, "r") as f:
        config = json.load(f)

    if language == "EN":
        speaker_id = config["data"]["spk2id"]["EN-US"]
    else:
        speaker_id = config["data"]["spk2id"][language]
    print(f"Language: {language}, Speaker ID: {speaker_id}, dec_len: {dec_len}")

    if not os.path.exists(ckpt_path):
        print(f"Downloading checkpoint for {language}...")
        load_or_download_model(locale=language, device="cpu")

    print("\nLoading model ...")
    tts = TTS(
        language=language,
        dec_len=dec_len,
        config_path=config_path,
        ckpt_path=ckpt_path,
        device="cpu"
    )

    upsample = get_upsample_factor(tts.model)
    audio_out_len = dec_len * upsample
    print(f"  Upsample factor: {upsample}x → decoder output: {dec_len}×{upsample} = {audio_out_len} samples")

    # ─── 临时覆写 forward ─────────────────────────────────────────────────────
    # 导出各子图时需要切换 forward 函数
    original_forward = tts.model.forward

    # ─── 导出 encoder.onnx ───────────────────────────────────────────────────
    encoder_path = os.path.join(out_dir, f"encoder-{lower_lang}.onnx")
    print(f"\n[1/4] Exporting encoder → {encoder_path}")
    with torch.no_grad():
        export_encoder(tts, encoder_path, language)

    # ─── 导出 flow.onnx ──────────────────────────────────────────────────────
    flow_path = os.path.join(out_dir, f"flow-{lower_lang}-d{dec_len}.onnx")
    print(f"\n[2/4] Exporting flow → {flow_path}")
    with torch.no_grad():
        export_flow(tts, flow_path, dec_len)

    # ─── 导出 generator.onnx ─────────────────────────────────────────────────
    gen_path = os.path.join(out_dir, f"generator-{lower_lang}-d{dec_len}.onnx")
    print(f"\n[3/4] Exporting generator → {gen_path}")
    with torch.no_grad():
        export_generator(tts, gen_path, dec_len)

    # ─── 导出 g.bin ──────────────────────────────────────────────────────────
    g_bin_path = os.path.join(out_dir, f"g-{lower_lang}.bin")
    print(f"\n[4/4] Exporting speaker embedding → {g_bin_path}")
    export_g_bin(tts, speaker_id, g_bin_path)

    # ─── 可选: 导出兼容版 decoder.onnx ───────────────────────────────────────
    if args.export_combined_decoder:
        print(f"\n[+] Exporting combined decoder (AXERA-compatible) ...")
        # 需要临时设置 forward 为 flow_dec_forward
        tts.model.forward = tts.model.flow_dec_forward
        dec_path = os.path.join(out_dir, f"decoder-{lower_lang}.onnx")
        z_p = torch.rand(1, 192, dec_len)
        g   = torch.rand(1, 256, 1)
        torch.onnx.export(
            tts.model, (z_p, g), dec_path,
            opset_version=16, export_params=True,
            input_names=["z_p", "g"], output_names=["audio"],
        )
        tts.model.forward = original_forward
        simplify_and_save(dec_path)
        print(f"  [Decoder] Input:  z_p float32[1,192,{dec_len}], g float32[1,256,1]")
        print(f"  [Decoder] Output: audio float32[1,1,{audio_out_len}]")

    # ─── 可选: 生成校准数据 ───────────────────────────────────────────────────
    if args.calib:
        print(f"\n[+] Generating calibration data ...")
        calib_dir = os.path.join(out_dir, "calib_3stage")
        generate_calibration_data(
            tts, speaker_id, TEXT[language], language, calib_dir, dec_len
        )

    # ─── 导出摘要 ─────────────────────────────────────────────────────────────
    tts.model.forward = original_forward
    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print(f"Language:   {language}")
    print(f"Speaker ID: {speaker_id}")
    print(f"dec_len:    {dec_len}")
    print(f"Output dir: {out_dir}")
    print()
    print("Files generated:")
    for fname in [
        f"encoder-{lower_lang}.onnx",
        f"flow-{lower_lang}-d{dec_len}.onnx",
        f"generator-{lower_lang}-d{dec_len}.onnx",
        f"g-{lower_lang}.bin",
    ]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024 / 1024
            role = {
                "encoder": "CPU (动态 shape)",
                "flow":    "NPU (固定 shape)",
                "generator": "NPU (固定 shape)",
                "g-":      "预计算嵌入",
            }
            tag = next((v for k, v in role.items() if k in fname), "")
            print(f"  {fname:45s} {size:7.1f} MB  [{tag}]")

    print()
    print("Inference pipeline:")
    print("  1. [CPU] encoder.onnx: phone/tone/lang → z_p, pronoun_lens, audio_len")
    print(f"  2. [NPU] flow.onnx:    z_p[1,192,{dec_len}] + g → z[1,192,{dec_len}]")
    print(f"  3. [NPU] generator.onnx: z[1,192,{dec_len}] + g → audio[1,1,{audio_out_len}]")
    print("     (步骤2+3 循环运行，z_p 每次取 dec_len 长的切片)")
    print()
    print("Note: flow + generator 可以在 AXERA pulsar2 中分别转换为 axmodel，")
    print("      或合并为一个 flow_dec.axmodel（等同于原始 AXERA convert.py decoder）")
    print("="*60)


if __name__ == "__main__":
    main()
