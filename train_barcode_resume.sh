#!/usr/bin/env sh
set -e

./caffe train --solver=Barcode.prototxt --snapshot=Resume$@
