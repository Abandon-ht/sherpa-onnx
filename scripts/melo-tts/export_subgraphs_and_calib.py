#!/usr/bin/env python3

import argparse
import copy
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort

LANG_FOLDERS = ["zh_en", "en", "jp", "kr", "es", "fr"]

SUBGRAPH_A_INPUTS = [
    "x",
    "x_lengths",
    "tones",
    "sid",
    "noise_scale",
    "length_scale",
    "noise_scale_w",
]
SUBGRAPH_A_OUTPUTS = ["/Mul_10_output_0", "/Unsqueeze_6_output_0"]
SUBGRAPH_B_INPUTS = ["/Mul_10_output_0", "/Unsqueeze_6_output_0"]
SUBGRAPH_B_OUTPUTS = ["y"]


def sanitize_tensor_name(name: str) -> str:
    # Keep folder names flat and filesystem friendly.
    out = name.strip().lstrip("/")
    out = out.replace("/", "__")
    out = out.replace(":", "_")
    if not out:
        out = "root"
    return out


def load_vocab_and_sid(model_path: Path) -> Tuple[int, int]:
    m = onnx.load(str(model_path))

    vocab_size = None
    for init in m.graph.initializer:
        if init.name == "model.model.enc_p.emb.weight":
            vocab_size = int(init.dims[0])
            break
    if vocab_size is None:
        raise RuntimeError(f"Cannot find vocab initializer in {model_path}")

    meta = {x.key: x.value for x in m.metadata_props}
    sid = int(meta.get("speaker_id", "0"))
    return vocab_size, sid


def export_subgraph(full_model: Path, out_model: Path, inputs: List[str], outputs: List[str]) -> None:
    if out_model.exists():
        out_model.unlink()

    try:
        onnx.utils.extract_model(
            str(full_model),
            str(out_model),
            inputs,
            outputs,
            check_model=False,
        )
    except Exception:
        # Some models miss explicit value_info shapes. Try shape inference first.
        inferred = onnx.shape_inference.infer_shapes(onnx.load(str(full_model)))
        inferred_model = out_model.with_suffix(".inferred_tmp.onnx")
        onnx.save(inferred, str(inferred_model))
        try:
            onnx.utils.extract_model(
                str(inferred_model),
                str(out_model),
                inputs,
                outputs,
                check_model=False,
            )
        finally:
            if inferred_model.exists():
                inferred_model.unlink()


def build_quant_input_configs(tensor_names: List[str]) -> List[Dict[str, object]]:
    items = []
    for t in tensor_names:
        folder_name = sanitize_tensor_name(t)
        items.append(
            {
                "tensor_name": t,
                "calibration_dataset": f"./{folder_name}.tar.gz",
                "calibration_size": -1,
                "calibration_format": "Numpy",
            }
        )
    return items


def write_config(template: Dict[str, object], out_path: Path, tensor_names: List[str]) -> None:
    cfg = copy.deepcopy(template)
    cfg.setdefault("quant", {})
    cfg["quant"]["input_configs"] = build_quant_input_configs(tensor_names)

    out_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_subgraph_a_inputs(vocab_size: int, sid_value: int, length: int) -> Dict[str, np.ndarray]:
    x = np.random.randint(0, vocab_size, size=(1, length), dtype=np.int64)
    tones = np.zeros((1, length), dtype=np.int64)

    return {
        "x": x,
        "x_lengths": np.array([length], dtype=np.int64),
        "tones": tones,
        "sid": np.array([sid_value], dtype=np.int64),
        "noise_scale": np.array([0.667], dtype=np.float32),
        "length_scale": np.array([1.0], dtype=np.float32),
        "noise_scale_w": np.array([0.8], dtype=np.float32),
    }


def save_sample(base_dir: Path, tensor_name: str, array: np.ndarray, sample_id: int) -> None:
    folder = base_dir / sanitize_tensor_name(tensor_name)
    folder.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = folder / f"{ts}_{sample_id:04d}.npy"
    np.save(str(filename), array)


def package_folder(folder: Path, tar_path: Path) -> None:
    # Write files at archive root to avoid nested directory structure.
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(folder.glob("*.npy")):
            tar.add(str(f), arcname=f.name)


def package_all_tensors(sample_root: Path, tensor_names: List[str]) -> None:
    for t in tensor_names:
        folder = sample_root / sanitize_tensor_name(t)
        tar_path = sample_root / f"{sanitize_tensor_name(t)}.tar.gz"
        package_folder(folder, tar_path)


def collect_calibration_data(
    subgraph_a_model: Path,
    sample_root_a: Path,
    sample_root_b: Path,
    vocab_size: int,
    sid_value: int,
    num_samples: int,
    length_buckets: List[int],
) -> None:
    sess_a = ort.InferenceSession(str(subgraph_a_model), providers=["CPUExecutionProvider"])

    np.random.seed(20260421)

    for i in range(num_samples):
        length = int(np.random.choice(length_buckets))
        a_inputs = make_subgraph_a_inputs(vocab_size=vocab_size, sid_value=sid_value, length=length)

        for name, value in a_inputs.items():
            save_sample(sample_root_a, name, value, i)

        a_outs = sess_a.run(SUBGRAPH_A_OUTPUTS, a_inputs)
        b_inputs = {
            SUBGRAPH_B_INPUTS[0]: a_outs[0],
            SUBGRAPH_B_INPUTS[1]: a_outs[1],
        }

        for name, value in b_inputs.items():
            save_sample(sample_root_b, name, value, i)


def process_language(
    lang_dir: Path,
    template_cfg: Dict[str, object],
    num_samples: int,
    length_buckets: List[int],
) -> None:
    model_path = lang_dir / "model.onnx"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    subgraph_a = lang_dir / "subgraph_a.onnx"
    subgraph_b = lang_dir / "subgraph_b.onnx"

    export_subgraph(model_path, subgraph_a, SUBGRAPH_A_INPUTS, SUBGRAPH_A_OUTPUTS)
    export_subgraph(model_path, subgraph_b, SUBGRAPH_B_INPUTS, SUBGRAPH_B_OUTPUTS)

    cfg_a = lang_dir / "config_subgraph_a_u16.json"
    cfg_b = lang_dir / "config_subgraph_b_u16.json"
    write_config(template_cfg, cfg_a, SUBGRAPH_A_INPUTS)
    write_config(template_cfg, cfg_b, SUBGRAPH_B_INPUTS)

    calib_root = lang_dir / "calib"
    sample_root_a = calib_root / "subgraph_a"
    sample_root_b = calib_root / "subgraph_b"
    sample_root_a.mkdir(parents=True, exist_ok=True)
    sample_root_b.mkdir(parents=True, exist_ok=True)

    # Remove previous calibration artifacts for deterministic reruns.
    for sample_root in [sample_root_a, sample_root_b]:
        for old_npy in sample_root.rglob("*.npy"):
            old_npy.unlink()
        for old_tar in sample_root.glob("*.tar.gz"):
            old_tar.unlink()
        for old_dir in sorted(sample_root.glob("*")):
            if old_dir.is_dir() and not any(old_dir.iterdir()):
                old_dir.rmdir()

    vocab_size, sid_value = load_vocab_and_sid(model_path)
    collect_calibration_data(
        subgraph_a_model=subgraph_a,
        sample_root_a=sample_root_a,
        sample_root_b=sample_root_b,
        vocab_size=vocab_size,
        sid_value=sid_value,
        num_samples=num_samples,
        length_buckets=length_buckets,
    )

    package_all_tensors(sample_root_a, SUBGRAPH_A_INPUTS)
    package_all_tensors(sample_root_b, SUBGRAPH_B_INPUTS)

    summary = {
        "language_folder": lang_dir.name,
        "subgraph_a_model": str(subgraph_a.name),
        "subgraph_b_model": str(subgraph_b.name),
        "subgraph_a_config": str(cfg_a.name),
        "subgraph_b_config": str(cfg_b.name),
        "subgraph_a_tar": [f"calib/subgraph_a/{sanitize_tensor_name(t)}.tar.gz" for t in SUBGRAPH_A_INPUTS],
        "subgraph_b_tar": [f"calib/subgraph_b/{sanitize_tensor_name(t)}.tar.gz" for t in SUBGRAPH_B_INPUTS],
    }
    (lang_dir / "subgraph_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--template-config", type=Path, default=Path("config_decoder_front_u16.json"))
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--length-buckets", type=str, default="32,48,64,80,96,128,160,192,224,256")
    parser.add_argument("--langs", type=str, default=",".join(LANG_FOLDERS))
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    template_path = (base_dir / args.template_config).resolve() if not args.template_config.is_absolute() else args.template_config
    template_cfg = json.loads(template_path.read_text(encoding="utf-8"))

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    length_buckets = [int(x.strip()) for x in args.length_buckets.split(",") if x.strip()]

    for folder in langs:
        lang_dir = base_dir / folder
        print(f"\n=== Processing {folder} ===")
        process_language(
            lang_dir=lang_dir,
            template_cfg=template_cfg,
            num_samples=args.samples,
            length_buckets=length_buckets,
        )
        print(f"Done: {folder}")

    print("\nAll languages processed.")


if __name__ == "__main__":
    main()
