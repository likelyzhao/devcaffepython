#!/usr/bin/env sh
set -e

./caffelog train --solver=Barcode.prototxt $@
