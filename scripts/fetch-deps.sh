#!/usr/bin/bash
# Fetch and build the open-source pieces this project depends on. Nothing here is Qualcomm-licensed.
#   hexagonrpc/     linux-msm/hexagonrpc at the commit our patch applies to, patched and built
#   qcom-fastrpc/   qualcomm/fastrpc (BSD-3), used only to look up error-code names   [--refs]
#   ref/llvm/       LLVM's Hexagon instruction tables (Apache-2.0)                     [--refs]
#   data/mnist/     the four MNIST files                                               [--mnist]
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
if [ ! -d hexagonrpc ]; then
	git clone https://github.com/linux-msm/hexagonrpc.git
	git -C hexagonrpc checkout -q da7a374
	git -C hexagonrpc apply "$ROOT/patches/hexagonrpcd-npu-open.patch"
fi
[ -d hexagonrpc/build ] || meson setup hexagonrpc/build hexagonrpc -Dhexagonrpcd_verbose=true
ninja -C hexagonrpc/build
for arg in "$@"; do case "$arg" in
--refs)
	[ -d qcom-fastrpc ] || git clone --depth 1 https://github.com/qualcomm/fastrpc.git qcom-fastrpc
	mkdir -p ref/llvm
	for f in HexagonDepInstrInfo.td HexagonDepInstrFormats.td; do
		curl -sfL -o ref/llvm/$f https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/Hexagon/$f
	done ;;
--mnist)
	mkdir -p data/mnist
	for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
		[ -s data/mnist/$f ] || curl -sfL -o data/mnist/$f https://ossci-datasets.s3.amazonaws.com/mnist/$f
	done ;;
esac; done
echo "dependencies ready"
