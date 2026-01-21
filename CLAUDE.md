# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sherpa-onnx is a speech processing framework supporting ASR (streaming/non-streaming), TTS, speaker diarization, VAD, keyword spotting, audio tagging, and more. It provides bindings for 12 languages (C++, C, Python, JavaScript, Java, Kotlin, C#, Swift, Go, Dart, Rust, Pascal) and runs on Linux, Windows, macOS, Android, iOS, HarmonyOS, WebAssembly, and various NPUs (Rockchip, Qualcomm, Ascend, Axera).

## Build Commands

### Standard CMake Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Common CMake Options
- `-DSHERPA_ONNX_ENABLE_PYTHON=ON` - Build Python bindings
- `-DSHERPA_ONNX_ENABLE_TESTS=ON` - Build tests
- `-DSHERPA_ONNX_ENABLE_JNI=ON` - Build JNI for Android/Java
- `-DSHERPA_ONNX_ENABLE_C_API=ON` - Build C API (default ON)
- `-DSHERPA_ONNX_ENABLE_GPU=ON` - Enable CUDA support
- `-DSHERPA_ONNX_ENABLE_DIRECTML=ON` - Enable DirectML (Windows)
- `-DBUILD_SHARED_LIBS=ON` - Build shared libraries

### Python Package
```bash
pip install -e .
# or with specific args
SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON" pip install -e .
```

### Platform-Specific Builds
Platform-specific build scripts are in the root directory:
- `build-android-*.sh` - Android builds
- `build-ios*.sh` - iOS builds
- `build-wasm-*.sh` - WebAssembly builds
- `build-*-linux-*.sh` - Cross-compilation for ARM/RISC-V

### Running Tests
```bash
# After building with -DSHERPA_ONNX_ENABLE_TESTS=ON
cd build && ctest

# Python tests
cd sherpa-onnx/python/tests
pytest
```

## Code Architecture

### Core Structure
```
sherpa-onnx/csrc/          # C++ core implementation
├── offline-*              # Non-streaming models and recognizers
├── online-*               # Streaming models and recognizers
├── speaker-*              # Speaker diarization/identification
├── keyword-spotter-*      # Keyword spotting
└── *-impl.h/cc            # Implementation details
```

### API Layers
The codebase follows a layered API design:
1. **C++ Core** (`sherpa-onnx/csrc/`) - Main implementation
2. **C API** (`sherpa-onnx/c-api/`) - C bindings for cross-language support
3. **Language Bindings** - Python, Java, etc. wrap the C API

### Model Types
- **Transducer models**: Zipformer, Conformer (streaming and non-streaming)
- **CTC models**: Paraformer, NeMo, WeNet
- **Whisper**: OpenAI Whisper variants
- **TTS**: Piper, Matcha, Kokoro, VITS, MeloTTS

### Configuration Pattern
Models are configured via config structs rather than hardcoded values. See `*-model-config.h` files for model configuration options.

## Code Style

### C++
- Google C++ style (configured in `.clang-format`)
- Pointer alignment: Right (`int *p`)
- Static analysis: clang-tidy with warnings as errors

### Python
- flake8 with max line length 120
- Excludes: `.git`, `./cmake`

## Key Directories

- `scripts/` - Model conversion scripts (50+ model types)
- `*-api-examples/` - Usage examples for each language
- `android/` - Android applications and libraries
- `flutter/` - Flutter packages
- `wasm/` - WebAssembly builds for browser/Node.js
- `.github/workflows/` - 130+ CI workflows for all platforms

## Version Management

Version is defined in `CMakeLists.txt` line 17 (`SHERPA_ONNX_VERSION`). When updating, also update:
- `CHANGELOG.md`
- `new-release.sh`

## Build and Run C++ Examples

The `cxx-api-examples/` directory contains C++ demo examples for various features (ASR, TTS, VAD, etc.).

### Build Examples
```bash
mkdir build && cd build

# Build with shared ONNX runtime (faster builds)
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release -j 8
```

### Run Examples
After building, executables are in `build/bin/` (Linux/macOS) or `build/bin/Release/` (Windows):
```bash
# Example: Run SenseVoice ASR
./bin/sense-voice-cxx-api

# Example: Run streaming zipformer
./bin/streaming-zipformer-cxx-api

# Example: Run VAD
./bin/vad-cxx-api
```

Note: Most examples require downloading models first. Check the source file for model paths and download instructions.

### Available Examples
- **ASR**: `sense-voice-cxx-api`, `whisper-cxx-api`, `streaming-zipformer-cxx-api`, `funasr-nano-cxx-api`, etc.
- **TTS**: `matcha-tts-*-cxx-api`, `kokoro-tts-*-cxx-api` (require `-DSHERPA_ONNX_ENABLE_TTS=ON`)
- **VAD**: `vad-cxx-api`
- **Audio Tagging**: `audio-tagging-ced-cxx-api`, `audio-tagging-zipformer-cxx-api`
- **Microphone**: `*-microphone-cxx-api` (require `-DSHERPA_ONNX_ENABLE_PORTAUDIO=ON`)

## Build Fast

- Don't `rm -rf build/` every time - incremental builds are much faster
- Use parallel builds: `cmake --build . --config Release -j N` (N = number of CPU cores)
- Use shared ONNX runtime for faster link times:
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
  ```
- Static builds (`BUILD_SHARED_LIBS=OFF`, default) produce standalone binaries but take longer to link

### IMPORTANT: Windows MSVC Static Build Issue

**DO NOT use static onnxruntime on Windows** (`BUILD_SHARED_LIBS=OFF`). The pre-compiled static onnxruntime library has MSVC ABI compatibility issues with newer Visual Studio versions (19.43+). You will get linker errors like:
```
error LNK2019: unresolved external symbol __std_find_first_of_trivial_pos_1
error LNK2019: unresolved external symbol __std_find_first_of_trivial_pos_2
```

**Solution**: Always use `-DBUILD_SHARED_LIBS=ON` for Windows MSVC builds:
```bash
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON ..
```

### ccache for Windows MSVC Builds
ccache is automatically enabled if found. Install ccache to `D:\soft\ccache` or set `CCACHE_DIR` environment variable.

Reference: https://github.com/ccache/ccache/wiki/MS-Visual-Studio

Build with Visual Studio generator:
```bash
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release -j 8
```

Verify ccache is working:
```bash
ccache -s  # Check statistics before/after build
```