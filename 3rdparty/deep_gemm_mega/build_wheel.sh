#!/usr/bin/env bash
set -euo pipefail

source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output_dir=${1:?usage: build_wheel.sh OUTPUT_DIRECTORY}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
cp -R "$source_dir/." "$build_dir/"
cd "$build_dir"
"${PYTHON:-python3}" test_package.py

fetch_dependency() {
    local name=$1 repo=$2 revision=$3 checksum=$4
    curl --fail --location --retry 3 \
        "https://codeload.github.com/$repo/tar.gz/$revision" -o "$name.tar.gz"
    printf '%s  %s\n' "$checksum" "$name.tar.gz" | sha256sum --check -
    mkdir -p "third-party/$name"
    tar -xzf "$name.tar.gz" --strip-components=1 -C "third-party/$name"
}

fetch_dependency actlize t-head/actlize 9771215b2119d19464c0663299885699d74985a8 \
    2f10118763f6f0e0355973f7683098279a7570557b25d0fe608be8395da61284
fetch_dependency fmt fmtlib/fmt a79df4504cd4e42ed004b1113fb82171e62ed822 \
    c2561661329c1ef3fdb52a38922fa879e0877f7a06b392f3378bbeabcd488330

# Build against the same Torch and SDK as the target PPU image.
"${PYTHON:-python3}" -m pip wheel --no-build-isolation --no-deps . -w "$output_dir"
