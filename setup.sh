#!/usr/bin/env bash
# Create the environment.
#
# MuJoCo 2.3.7 (the version the model card's results were measured with) has no wheel for Python 3.12, so it is built
# from source against the official C library, which is downloaded to .mujoco/. The Python bindings of 2.3.7 are built
# with pybind11 2.11, which is broken under NumPy 2 (LeRobot needs it): every uint8 array of the bindings gets stride
# 0, so robosuite cannot hide the collision geoms (vopt.geomgroup) and the cameras render flat-coloured collision
# meshes instead of the textured visuals. The policy fails on those images. Building the bindings with pybind11
# 2.13 fixes it and leaves the physics (the C library) untouched.
set -euo pipefail
cd "$(dirname "$0")"

MUJOCO_VERSION=2.3.7
PYBIND11_VERSION=2.13.6
for tool in git uv cmake g++ curl; do
    command -v "$tool" >/dev/null || { echo "setup.sh: '$tool' is required but not installed" >&2; exit 1; }
done

git submodule update --init

mkdir -p .mujoco
if [ ! -d ".mujoco/mujoco-$MUJOCO_VERSION" ]; then
    curl -fsSL "https://github.com/google-deepmind/mujoco/releases/download/$MUJOCO_VERSION/mujoco-$MUJOCO_VERSION-linux-x86_64.tar.gz" |
        tar -xz -C .mujoco
fi
if [ ! -d ".mujoco/pybind11-$PYBIND11_VERSION" ]; then
    curl -fsSL "https://github.com/pybind/pybind11/archive/refs/tags/v$PYBIND11_VERSION.tar.gz" | tar -xz -C .mujoco
fi

export MUJOCO_PATH="$PWD/.mujoco/mujoco-$MUJOCO_VERSION"
export MUJOCO_PLUGIN_PATH="$MUJOCO_PATH/bin/mujoco_plugin"
export MUJOCO_CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_PYBIND11=$PWD/.mujoco/pybind11-$PYBIND11_VERSION"

uv sync "$@"

# uv caches built wheels without regard to the build flags, so a wheel built earlier with the default pybind11 may
# have been reused: drop uv's cached mujoco builds and rebuild if the bindings are broken.
check='import mujoco; assert mujoco.MjvOption().geomgroup.strides == (1,), "mujoco bindings are broken"'
if ! uv run --no-sync python -c "$check" 2>/dev/null; then
    uv cache clean mujoco
    uv sync --reinstall-package mujoco "$@"
    uv run --no-sync python -c "$check"
fi
echo "setup.sh: done"
