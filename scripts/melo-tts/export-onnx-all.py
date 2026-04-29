#!/usr/bin/env python3
# Export MeloTTS models and frontend files for all supported languages:
# EN, ZH, JP, KR, ES, FR

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import onnx
import torch
from melo.api import TTS
from melo.text import language_id_map, language_tone_start_map
from melo.text.chinese import pinyin_to_symbol_map
from melo.text.cleaner import clean_text
from melo.text.english import eng_dict, refine_syllables
from pypinyin import Style, lazy_pinyin, phrases_dict, pinyin_dict

for k, v in pinyin_to_symbol_map.items():
    if isinstance(v, list):
        break
    pinyin_to_symbol_map[k] = v.split()


LANGUAGE_SPECS = {
    "ZH": {"folder": "zh_en", "meta_language": "Chinese + English", "opset": 18},
    "EN": {"folder": "en", "meta_language": "English", "opset": 13},
    "JP": {"folder": "jp", "meta_language": "Japanese", "opset": 13},
    "KR": {"folder": "kr", "meta_language": "Korean", "opset": 13},
    "ES": {"folder": "es", "meta_language": "Spanish", "opset": 13},
    "FR": {"folder": "fr", "meta_language": "French", "opset": 13},
}


def get_initial_final_tone(word: str):
    initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)

    ans_phone = []
    ans_tone = []

    for c, v in zip(initials, finals):
        raw_pinyin = c + v
        v_without_tone = v[:-1]
        try:
            tone = v[-1]
        except Exception:
            return [], []

        pinyin = c + v_without_tone
        if tone not in "12345":
            return [], []

        if c:
            v_rep_map = {
                "uei": "ui",
                "iou": "iu",
                "uen": "un",
            }
            if v_without_tone in v_rep_map:
                pinyin = c + v_rep_map[v_without_tone]
        else:
            pinyin_rep_map = {
                "ing": "ying",
                "i": "yi",
                "in": "yin",
                "u": "wu",
            }
            if pinyin in pinyin_rep_map:
                pinyin = pinyin_rep_map[pinyin]
            else:
                single_rep_map = {
                    "v": "yu",
                    "e": "e",
                    "i": "y",
                    "u": "w",
                }
                if pinyin and pinyin[0] in single_rep_map:
                    pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

        if pinyin not in pinyin_to_symbol_map:
            # Keep behavior aligned with existing scripts: skip unknown entries.
            continue

        phone = pinyin_to_symbol_map[pinyin]
        ans_phone += phone
        ans_tone += [tone] * len(phone)

    return ans_phone, ans_tone


def generate_tokens(symbol_list: Iterable[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(symbol_list):
            f.write(f"{s} {i}\n")


def add_new_english_words(lexicon: Dict[str, List[List[str]]]) -> None:
    lexicon["kaldi"] = [["K", "AH0"], ["L", "D", "IH0"]]
    lexicon["SF"] = [["EH1", "S"], ["EH1", "F"]]


def generate_lexicon_zh_en(path: Path) -> None:
    word_dict = pinyin_dict.pinyin_dict
    phrases = phrases_dict.phrases_dict

    add_new_english_words(eng_dict)
    with path.open("w", encoding="utf-8") as f:
        for word in eng_dict:
            phones, tones = refine_syllables(eng_dict[word])
            tones = [t + language_tone_start_map["EN"] for t in tones]
            f.write(f"{word.lower()} {' '.join(phones)} {' '.join([str(t) for t in tones])}\n")

        for key in word_dict:
            if not (0x4E00 <= key <= 0x9FA5):
                continue
            w = chr(key)
            phone, tone = get_initial_final_tone(w)
            if not phone:
                continue
            f.write(f"{w} {' '.join(phone)} {' '.join(tone)}\n")

        for w in phrases:
            phone, tone = get_initial_final_tone(w)
            if not phone:
                continue
            if len(phone) != len(tone):
                continue
            f.write(f"{w} {' '.join(phone)} {' '.join(tone)}\n")


def generate_lexicon_en(path: Path) -> None:
    add_new_english_words(eng_dict)
    with path.open("w", encoding="utf-8") as f:
        for word in eng_dict:
            phones, tones = refine_syllables(eng_dict[word])
            tones = [t + language_tone_start_map["EN"] for t in tones]
            f.write(f"{word.lower()} {' '.join(phones)} {' '.join([str(t) for t in tones])}\n")


def strip_edge_blanks(phones: List[str], tones: List[int]) -> Tuple[List[str], List[int]]:
    left = 0
    right = len(phones)

    while left < right and phones[left] == "_":
        left += 1

    while right > left and phones[right - 1] == "_":
        right -= 1

    return phones[left:right], tones[left:right]


def to_lexicon_entry(text: str, language: str, tone_start: int) -> Optional[Tuple[str, str, List[str], List[int]]]:
    try:
        norm_text, phones, tones, _ = clean_text(text, language)
    except Exception:
        return None

    if not phones or not tones or len(phones) != len(tones):
        return None

    phones, tones = strip_edge_blanks(phones, tones)
    if not phones or len(phones) != len(tones):
        return None

    tones = [int(t) + tone_start for t in tones]

    key = norm_text if norm_text else text
    raw_key = text
    if not key.strip() or not raw_key.strip():
        return None

    return raw_key, key, phones, tones


def candidate_units_for_language(language: str) -> List[str]:
    punct = [
        "!",
        "?",
        ",",
        ".",
        "'",
        "-",
        ":",
        ";",
        "…",
        "，",
        "。",
        "！",
        "？",
        "、",
    ]

    if language == "JP":
        chars = punct[:]
        chars.extend([chr(i) for i in range(0x3041, 0x3097)])  # Hiragana
        chars.extend([chr(i) for i in range(0x30A1, 0x30FB)])  # Katakana
        chars.extend([chr(i) for i in range(0xFF66, 0xFF9E)])  # Half-width katakana
        chars.extend([chr(i) for i in range(0x4E00, 0xA000)])  # CJK Unified Ideographs
        return chars

    if language == "KR":
        chars = punct[:]
        chars.extend([chr(i) for i in range(0xAC00, 0xD7A4)])  # Hangul syllables
        return chars

    if language == "ES":
        base = list("abcdefghijklmnopqrstuvwxyz")
        extra = list("áéíóúüñ")
        upper = [x.upper() for x in base + extra]
        return punct + base + extra + upper + [str(i) for i in range(10)]

    if language == "FR":
        base = list("abcdefghijklmnopqrstuvwxyz")
        extra = list("àâæçéèêëîïôœùûüÿ")
        upper = [x.upper() for x in base + extra]
        return punct + base + extra + upper + [str(i) for i in range(10)]

    raise ValueError(f"Unexpected language for generic lexicon generation: {language}")


def generate_lexicon_generic(language: str, path: Path, allowed_tokens: Set[str]) -> None:
    tone_start = language_tone_start_map[language]
    units = candidate_units_for_language(language)

    seen_keys: Set[str] = set()
    with path.open("w", encoding="utf-8") as f:
        for u in units:
            item = to_lexicon_entry(u, language=language, tone_start=tone_start)
            if item is None:
                continue

            raw_key, normalized_key, phones, tones = item
            if any(phone not in allowed_tokens for phone in phones):
                continue

            keys = [normalized_key.lower()]
            if language == "JP":
                keys.insert(0, raw_key.lower())

            for key in keys:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                f.write(f"{key} {' '.join(phones)} {' '.join(str(t) for t in tones)}\n")


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: "SynthesizerTrn"):
        super().__init__()
        self.model = model
        self.lang_id = language_id_map[model.language]

    def forward(
        self,
        x,
        x_lengths,
        tones,
        sid,
        noise_scale,
        length_scale,
        noise_scale_w,
        max_len=None,
    ):
        # Keep x_lengths as a real dependency so export keeps this input.
        x = x + (x_lengths.to(x.dtype).sum() * 0).reshape(1, 1)

        bert = torch.zeros(x.shape[0], 1024, x.shape[1], dtype=torch.float32)
        ja_bert = torch.zeros(x.shape[0], 768, x.shape[1], dtype=torch.float32)
        lang_id = torch.zeros_like(x)
        lang_id[:, 1::2] = self.lang_id

        return self.model.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            tone=tones,
            language=lang_id,
            bert=bert,
            ja_bert=ja_bert,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale,
        )[0]


def export_one_language(language: str, out_dir: Path) -> None:
    spec = LANGUAGE_SPECS[language]
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TTS(language=language, device="cpu")

    tokens_path = out_dir / "tokens.txt"
    lexicon_path = out_dir / "lexicon.txt"
    model_path = out_dir / "model.onnx"

    generate_tokens(model.hps["symbols"], tokens_path)

    if language == "ZH":
        generate_lexicon_zh_en(lexicon_path)
    elif language == "EN":
        generate_lexicon_en(lexicon_path)
    else:
        generate_lexicon_generic(language, lexicon_path, set(model.hps["symbols"]))

    torch_model = ModelWrapper(model)
    x = torch.randint(low=0, high=10, size=(60,), dtype=torch.int64)
    x_lengths = torch.tensor([x.size(0)], dtype=torch.int64)
    sid_value = list(model.hps.data.spk2id.values())[0]
    sid = torch.tensor([sid_value], dtype=torch.int64)
    tones = torch.zeros_like(x)

    noise_scale = torch.tensor([1.0], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)
    noise_scale_w = torch.tensor([1.0], dtype=torch.float32)

    x = x.unsqueeze(0)
    tones = tones.unsqueeze(0)

    torch.onnx.export(
        torch_model,
        (
            x,
            x_lengths,
            tones,
            sid,
            noise_scale,
            length_scale,
            noise_scale_w,
        ),
        str(model_path),
        opset_version=spec["opset"],
        input_names=[
            "x",
            "x_lengths",
            "tones",
            "sid",
            "noise_scale",
            "length_scale",
            "noise_scale_w",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_lengths": {0: "N"},
            "tones": {0: "N", 1: "L"},
            "y": {0: "N", 1: "S", 2: "T"},
        },
    )

    meta_data = {
        "model_type": "melo-vits",
        "comment": "melo",
        "version": 2,
        "language": spec["meta_language"],
        "add_blank": int(model.hps.data.add_blank),
        "n_speakers": len(model.hps.data.spk2id),
        # Set jieba=1 for MeloTTS to ensure sherpa-onnx uses MeloTtsLexicon,
        # which provides tone IDs required by Melo models.
        "jieba": 1,
        "sample_rate": model.hps.data.sampling_rate,
        "bert_dim": 1024,
        "ja_bert_dim": 768,
        "speaker_id": sid_value,
        "lang_id": language_id_map[model.language],
        "tone_start": language_tone_start_map[model.language],
        "url": "https://github.com/myshell-ai/MeloTTS",
        "license": "MIT license",
        "description": "MeloTTS is a high-quality multi-lingual text-to-speech library by MyShell.ai",
    }
    add_meta_data(str(model_path), meta_data)

    # Keep the lexicon generation as the final step so the file on disk matches
    # the verified generic-lexicon path for JP/KR/ES/FR.
    if language == "ZH":
        generate_lexicon_zh_en(lexicon_path)
    elif language == "EN":
        generate_lexicon_en(lexicon_path)
    else:
        generate_lexicon_generic(language, lexicon_path, set(model.hps["symbols"]))


def main() -> None:
    base = Path(".").resolve()

    for language, spec in LANGUAGE_SPECS.items():
        out_dir = base / spec["folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Exporting {language} -> {out_dir.name} ===")
        export_one_language(language, out_dir)

    print("\nDone. Generated language folders:")
    for spec in LANGUAGE_SPECS.values():
        folder = Path(spec["folder"])
        print(f"- {folder}/model.onnx")
        print(f"- {folder}/lexicon.txt")
        print(f"- {folder}/tokens.txt")


if __name__ == "__main__":
    main()
