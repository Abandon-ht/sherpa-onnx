#!/usr/bin/env bash

set -ex

function install() {
  pip install torch==2.3.1+cpu torchaudio==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

  pushd /tmp
  git clone https://github.com/myshell-ai/MeloTTS
  cd MeloTTS
  pip install -r ./requirements.txt

  pip install soundfile onnx==1.15.0 onnxruntime==1.16.3

  python3 -m unidic download
  popd
}

install

export PYTHONPATH=/tmp/MeloTTS:$PYTHONPATH

echo "pwd: $PWD"

./export-onnx-all.py

for lang in zh_en en jp kr es fr; do
  cp -v README.md "$lang"
  ls -lh "$lang"
  echo "---"
done

ls -lh
