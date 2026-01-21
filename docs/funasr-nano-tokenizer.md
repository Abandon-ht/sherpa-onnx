# FunASR-Nano Tokenizer

A self-contained Qwen3 ByteLevel-BPE tokenizer implementation for the FunASR-Nano speech recognition model.

## Overview

The FunASR-Nano tokenizer (`FunASRNanoTokenizer`) is a standalone implementation that:
- Has **no dependency** on tokenizers-cpp or HuggingFace tokenizers library
- Implements the **Qwen3 ByteLevel-BPE** tokenization algorithm
- Supports **AddedTokens** via Trie-based longest-match
- Provides **streaming decoding** for real-time ASR output with callbacks
- Works on **multiple platforms**: Desktop (Windows/Linux/macOS), Android, HarmonyOS

## New: Streaming Callback Support

The tokenizer now supports streaming text output with callbacks for real-time ASR applications:

```cpp
#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"

// Callback receives text chunks as tokens are decoded
auto callback = [](const std::string &text, int64_t token_id, bool is_final) -> bool {
    std::cout << text << std::flush;  // Stream output in real-time
    return true;  // Return false to stop early
};

// Decode with streaming callback
tokenizer.DecodeWithCallback(token_ids, callback);

// Or use StreamingDecodeState for manual control
StreamingDecodeState state;
for (int64_t token_id : token_ids) {
    std::string text = tokenizer.DecodeTokenStreaming(token_id, &state);
    if (!text.empty()) {
        std::cout << text << std::flush;
    }
}
// Flush any remaining bytes
std::cout << state.FlushPendingBytes();
```

## Required Files

The tokenizer requires three files in a directory:

```
tokenizer_dir/
├── tokenizer.json    # Contains added_tokens definitions
├── vocab.json        # Token-to-ID mapping {"token": id, ...}
└── merges.txt        # BPE merge rules (one pair per line)
```

For FunASR-Nano, these files are located in the `Qwen3-0.6B` subdirectory of the model package.

## Architecture

### Class Diagram

```
FunASRNanoTokenizer
├── Public Methods
│   ├── Encode(text) -> vector<int64_t>
│   ├── Decode(token_ids) -> string
│   ├── DecodeTokenStreaming(token_id, state) -> string
│   ├── DecodeWithCallback(token_ids, callback) -> string
│   ├── GetTokenStringStreaming(token_id, pending_bytes) -> string
│   ├── IsSpecialToken(token_id) -> bool
│   ├── IdToToken(token_id) -> string
│   ├── GetEosTokenId() -> int64_t
│   ├── GetPadTokenId() -> int64_t
│   ├── GetImEndTokenId() -> int64_t
│   └── GetImStartTokenId() -> int64_t
│
├── Callback Type
│   └── StreamingTokenCallback = function<bool(text, token_id, is_final)>
│
├── State Class
│   └── StreamingDecodeState
│       ├── PendingBytes() -> string&
│       ├── TokenCount() -> int32_t
│       ├── IncrementTokenCount()
│       ├── Reset()
│       ├── HasPendingBytes() -> bool
│       └── FlushPendingBytes() -> string
│
├── Data Structures
│   ├── AddedToken {content, id, single_word, lstrip, rstrip, normalized, special}
│   └── TrieNode {next: map<byte, node_index>, token_index}
│
└── Internal State
    ├── token2id_: map<string, int32_t>      # Token to ID mapping
    ├── id2token_: vector<string>             # ID to token mapping
    ├── merges_rank_: map<string, int32_t>    # BPE merge priorities
    ├── bpe_cache_: map<string, vector>       # BPE encoding cache
    ├── byte_to_unicode_[256]: string         # ByteLevel encoding table
    ├── unicode_to_byte_: map<string, uint8_t># ByteLevel decoding table
    ├── added_tokens_: vector<AddedToken>     # Special tokens
    ├── special_ids_: set<int32_t>            # Set of special token IDs
    └── trie_: vector<TrieNode>               # Trie for added token matching
```

## Algorithm Details

### Encoding Pipeline

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│ 1. AddedToken Matching (Trie)       │  Match special tokens like <|im_start|>
│    - Longest-match using Trie       │
│    - Check single_word boundary     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Pre-tokenization (Qwen3 Pattern) │  Split text into words/subwords
│    - Handle contractions ('s, 't)   │
│    - Group letters, numbers, punct  │
│    - Handle whitespace/newlines     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. ByteLevel Encoding               │  Map bytes to unicode characters
│    - Each byte → unicode char       │
│    - Handles all 256 byte values    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. BPE Merging                      │  Apply byte-pair encoding
│    - Split into UTF-8 characters    │
│    - Iteratively merge best pairs   │
│    - Uses merge rank from merges.txt│
│    - Results cached for performance │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. Token ID Lookup                  │  Convert tokens to IDs
│    - Look up each BPE token         │
│    - Skip unknown tokens            │
└─────────────────────────────────────┘
    │
    ▼
Output: vector<int64_t>
```

### Decoding Pipeline

```
Input: vector<int64_t>
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Token Lookup                     │  Convert IDs to tokens
│    - Skip special token IDs         │
│    - id2token_ lookup               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Concatenate Tokens               │  Join all token strings
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. ByteLevel Decoding               │  Map unicode back to bytes
│    - unicode_to_byte_ lookup        │
│    - Reconstruct original bytes     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Post-processing                  │  Clean up output
│    - Remove special token strings   │
│    - Trim whitespace                │
└─────────────────────────────────────┘
    │
    ▼
Output: string (UTF-8)
```

### Streaming Decoding

For real-time ASR, `GetTokenStringStreaming()` handles partial UTF-8 sequences:

```cpp
std::string pending_bytes;  // Maintains state between calls

// For each token from ASR model:
std::string text = tokenizer.GetTokenStringStreaming(token_id, &pending_bytes);
// Returns valid UTF-8 prefix, keeps incomplete bytes in pending_bytes
```

**Key Features:**
- Accumulates bytes until valid UTF-8 sequence is formed
- Handles incomplete multi-byte characters across token boundaries
- Replaces invalid sequences with U+FFFD (replacement character)

## ByteLevel Encoding

The ByteLevel algorithm maps all 256 possible byte values to printable Unicode characters:

```
Byte Range        Unicode Mapping
─────────────────────────────────────
33-126 (ASCII)    Same codepoint (printable ASCII)
161-172           Same codepoint (Latin-1 printable)
174-255           Same codepoint (Latin-1 printable)
0-32, 127-160,    256 + offset (mapped to higher Unicode)
173
```

This ensures all byte sequences can be represented as valid Unicode strings in the vocabulary.

## Pre-tokenization (Qwen3 Pattern)

The tokenizer implements a manual approximation of the Qwen3 pre-tokenizer regex pattern, avoiding `std::regex` due to missing `\p{L}/\p{N}` support in standard libraries.

**Supported Patterns:**
- English contractions: `'s`, `'t`, `'m`, `'d`, `'re`, `'ve`, `'ll`
- Letter sequences (including CJK, Japanese, Korean)
- Individual digits
- Punctuation sequences
- Whitespace handling (space, tab, newlines)

**Unicode Support:**
- CJK Unified Ideographs (U+4E00-U+9FFF)
- CJK Extension A (U+3400-U+4DBF)
- Hiragana/Katakana (U+3040-U+30FF)
- Hangul Syllables (U+AC00-U+D7AF)
- Latin Extended (U+00C0-U+02AF)
- Fullwidth digits (U+FF10-U+FF19)

## Special Tokens

The tokenizer recognizes these special tokens from Qwen3:

| Token | Default ID | Purpose |
|-------|------------|---------|
| `<\|im_end\|>` | 151645 | End of message marker |
| `<\|im_start\|>` | (from vocab) | Start of message marker |
| `<\|endoftext\|>` | (from vocab) | End of text (EOS) |
| `<\|pad\|>` | (from vocab) | Padding token |

These tokens are:
- Matched first during encoding (via Trie)
- Excluded from decoded output
- Tracked in `special_ids_` set

## AddedToken Trie

Special tokens are stored in a Trie data structure for efficient longest-match:

```
Root
 ├─ '<' ─ '|' ─ 'i' ─ 'm' ─ '_' ─ 'e' ─ 'n' ─ 'd' ─ '|' ─ '>' [token_index=0]
 │                          └─ 's' ─ 't' ─ 'a' ─ 'r' ─ 't' ─ '|' ─ '>' [token_index=1]
 └─ ...
```

**Properties:**
- Byte-level matching (not character-level)
- Returns longest match found
- Supports `single_word` constraint (word boundary checking)

## Platform Support

### Desktop (Windows/Linux/macOS)
```cpp
FunASRNanoTokenizer tokenizer("/path/to/tokenizer_dir");
```

### Android (API >= 9)
```cpp
FunASRNanoTokenizer tokenizer(asset_manager, "tokenizer_dir");
```

### HarmonyOS (OHOS)
```cpp
FunASRNanoTokenizer tokenizer(native_resource_manager, "tokenizer_dir");
```

## Usage Example

```cpp
#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"

// Initialize
sherpa_onnx::FunASRNanoTokenizer tokenizer("./Qwen3-0.6B");

// Encode text to token IDs
std::string text = "Hello, 你好!";
std::vector<int64_t> ids = tokenizer.Encode(text);

// Decode token IDs back to text
std::string decoded = tokenizer.Decode(ids);

// Streaming decode (for real-time ASR)
std::string pending;
for (int64_t id : ids) {
    std::string chunk = tokenizer.GetTokenStringStreaming(id, &pending);
    std::cout << chunk;  // Output as soon as valid UTF-8 is available
}
```

## Performance Optimizations

1. **BPE Cache**: Computed BPE results are cached in `bpe_cache_` to avoid recomputation for repeated words.

2. **Trie Matching**: O(m) complexity for matching added tokens where m is the token length.

3. **Precomputed Tables**: `byte_to_unicode_` and `unicode_to_byte_` are built once during initialization.

4. **Reserved Memory**: Vectors use `reserve()` to minimize reallocations.

## Internal Components

### JsonReader
A lightweight JSON parser that handles:
- String parsing with escape sequences (`\n`, `\uXXXX`, surrogate pairs)
- Integer parsing
- Boolean parsing
- Object/array skipping

### UTF-8 Utilities
- `Utf8Next()`: Parse next UTF-8 character
- `AppendUtf8()`: Encode codepoint to UTF-8
- `ConsumeValidUtf8Prefix()`: Extract valid UTF-8 prefix for streaming

## File Format Details

### vocab.json
```json
{
  "token1": 0,
  "token2": 1,
  "Ġhello": 1234,
  ...
}
```

### merges.txt
```
#version: 0.2
Ġ t
Ġ a
Ġt he
...
```
Each line: `left_token right_token` (space-separated, rank = line number)

### tokenizer.json (added_tokens section)
```json
{
  "added_tokens": [
    {
      "id": 151643,
      "content": "<|endoftext|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    ...
  ]
}
```

## Error Handling

The tokenizer uses `SHERPA_ONNX_LOGE` for error logging and `SHERPA_ONNX_EXIT(-1)` for fatal errors:

- Missing required files
- Parse failures
- Uninitialized tokenizer access

## Thread Safety

The tokenizer is **not thread-safe** due to the mutable `bpe_cache_`. For multi-threaded use:
- Create separate tokenizer instances per thread, OR
- Add external synchronization around `Encode()` calls

## StreamingDecodeState Class

The `StreamingDecodeState` class manages state for streaming token decoding:

```cpp
class StreamingDecodeState {
 public:
  StreamingDecodeState() = default;

  // Access pending bytes buffer
  std::string &PendingBytes();
  const std::string &PendingBytes() const;

  // Get count of tokens decoded so far
  int32_t TokenCount() const;

  // Increment token counter
  void IncrementTokenCount();

  // Reset state for new sequence
  void Reset();

  // Check if there are pending bytes
  bool HasPendingBytes() const;

  // Flush remaining bytes as replacement characters (U+FFFD)
  std::string FlushPendingBytes();

 private:
  std::string pending_bytes_;
  int32_t token_count_ = 0;
};
```

### Usage Pattern

```cpp
StreamingDecodeState state;

// Process each token
for (int64_t token_id : generated_tokens) {
    std::string text = tokenizer.DecodeTokenStreaming(token_id, &state);
    if (!text.empty()) {
        std::cout << text << std::flush;
    }
}

// Flush any remaining incomplete UTF-8 sequences
std::string remaining = state.FlushPendingBytes();
if (!remaining.empty()) {
    std::cout << remaining;
}
```

## StreamingTokenCallback Type

The callback type used for streaming decoding:

```cpp
using StreamingTokenCallback =
    std::function<bool(const std::string &text, int64_t token_id, bool is_final)>;
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `const std::string&` | Decoded text chunk from this token |
| `token_id` | `int64_t` | The token ID that was decoded |
| `is_final` | `bool` | True if this is the last token in the sequence |

### Return Value

- Return `true` to continue decoding
- Return `false` to stop decoding early

### Example

```cpp
auto callback = [](const std::string &text, int64_t token_id, bool is_final) -> bool {
    std::cout << text << std::flush;

    if (is_final) {
        std::cout << std::endl;
    }

    // Stop if we encounter a specific token
    if (token_id == STOP_TOKEN_ID) {
        return false;  // Stop early
    }

    return true;  // Continue
};

std::string full_text = tokenizer.DecodeWithCallback(token_ids, callback);
```

## Model Download

Download the FunASR-Nano model package:
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
```

Tokenizer files are in: `sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B/`
