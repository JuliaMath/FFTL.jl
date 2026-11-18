#!/bin/bash
# Compiles probe.c against the library of build.sh and runs it.
set -e
cd "$(dirname "$0")"
cc -O2 -o build/probe probe.c -Lbuild/lib -lfftlprobe -lm -lpthread -Wl,-rpath,"$PWD/build/lib"
build/probe
