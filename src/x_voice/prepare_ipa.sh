#!/bin/bash
set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root (sudo ./install.sh)"
  exit 1
fi

# Temporary build directory
BUILD_DIR="/tmp/espeak_build_tmp"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Number of parallel cores
CORES=$(nproc)

echo "Installing base build tools"
if command -v apt-get &> /dev/null; then
    apt-get update
    apt-get install -y build-essential autoconf automake libtool pkg-config git wget cmake
elif command -v yum &> /dev/null; then
    yum groupinstall -y "Development Tools"
    yum install -y autoconf automake libtool pkgconfig git wget cmake
fi

echo "Installing PCAudioLib"
if [ ! -d "pcaudiolib" ]; then
    git clone https://github.com/espeak-ng/pcaudiolib.git
fi
cd pcaudiolib
./autogen.sh
# Install to /usr
./configure --prefix=/usr --sysconfdir=/etc
make -j$CORES
make install
cd ..

echo "Installing Espeak-NG"
if [ ! -d "espeak-ng" ]; then
    git clone https://github.com/espeak-ng/espeak-ng.git
fi
cd espeak-ng
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build -j$CORES
cmake --install build

# Refresh dynamic library cache so the system can recognize new .so files immediately
ldconfig

echo "Cleaning up"
rm -rf "$BUILD_DIR"
echo "Verification:"
espeak-ng --version  
# Expected: Data at: /usr/share/espeak-ng-data
# python
# from phonemizer.backend import EspeakBackend
# print(EspeakBackend('en-us').phonemize(['Hello, world!']))
# Expected: ['həloʊ wɜːld ']
