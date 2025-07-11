#!/usr/bin/env bash
set -euo pipefail

usage(){
  cat<<EOF
Usage:
  $0 <start_s> <end_s> <input.mkv> <output.mp4> <subtitle>

  <start_s>      clip start time in seconds (e.g. 4800)
  <end_s>        clip end time in seconds (e.g. 5100)
  <input.mkv>    source file (must contain your subtitle track if you're burning embedded subs)
  <output.mp4>   output file (will be MP4)
  <subtitle>     either:
                   • a track index (e.g. "2") → burns that embedded text sub,
                   • an external .srt/.ass file (e.g. "subs.srt").
EOF
  exit 1
}

[[ $# -eq 5 ]] || usage

start="${1%s}"
end="${2%s}"
in="$3"
out="$4"
sub="$5"
fontsize=30
# fontcolor="E0E0E0"
# fontcolor="D0D0D0"
fontcolor="C0C0C0"
outline_size=2

# compute duration
dur=$((end - start))
if (( dur <= 0 )); then
  echo "Error: end time must be > start time"
  exit 1
fi

# build the subtitles filter argument
if [[ "$sub" =~ ^[0-9]+$ ]]; then
  # embedded track
  subfilter="subtitles='${in}':stream_index=${sub}:force_style='FontName=Arial,FontSize=${fontsize},PrimaryColour=&H00${fontcolor}&,OutlineColour=&H00000000&,BorderStyle=1,Outline=${outline_size},Shadow=0'"
else
  # external file
  subfilter="subtitles='${sub}':force_style='FontName=Arial,FontSize=${fontsize},PrimaryColour=&H00${fontcolor}&,OutlineColour=&H00000000&,BorderStyle=1,Outline=${outline_size},Shadow=0'"
fi

echo "▶ Burning [$sub] into ${in}, then cutting ${start}s→${end}s → ${out}"

ffmpeg -y \
  -i "$in" \
  -filter_complex \
    "[0:v]$subfilter,trim=start=${start}:end=${end},setpts=PTS-STARTPTS[v]; \
     [0:a]atrim=start=${start}:end=${end},asetpts=PTS-STARTPTS[a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 \
    -preset medium -crf 23 \
    -profile:v baseline -level 3.0 -pix_fmt yuv420p \
  -c:a aac \
    -b:a 128k \
    -ac 2 \
    -profile:a aac_low \
  -movflags +faststart \
  -map_chapters -1 \
  "$out"

echo "✔ Done."