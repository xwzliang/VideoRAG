#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <input_file> <output_file> [crop_pixels_each_side]

  <input_file>                e.g. input.mp4
  <output_file>               e.g. output_short.mp4
  [crop_pixels_each_side]     optional; defaults to 360
EOF
  exit 1
}

# require at least two args, no more than three
if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
fi

input="$1"
output="$2"
# default to 360 if not provided
crop_pixels="${3:-360}"

ffmpeg -y -i "$input" \
  -vf " \
    crop=iw-2*${crop_pixels}:ih:${crop_pixels}:0, \
    scale=1080:-1, \
    pad=1080:1920:(1080-iw)/2:(1920-ih)/2:color=black \
  " \
  -c:v libx264 -crf 18 -preset medium \
  -c:a copy \
  "$output"

echo "✔ Video portrait created: $out"