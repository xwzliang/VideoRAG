#!/usr/bin/env bash
input="$HOME/videos/Batman_Begins_2005_BluRay_1080p_x265_10bit_2Audio_MNHD-FRDS.mkv"

# 1. Get all subtitle stream indexes (e.g. “2”, “3”, …)
mapfile -t subs < <(
  ffprobe -v error \
          -select_streams s \
          -show_entries stream=index \
          -of csv=p=0 \
          "$input"
)

# 2. Loop through each subtitle stream
for idx in "${subs[@]}"; do
  # get the codec (subrip, ass, webvtt, hdmv_pgs_subtitle)…
  codec=$(ffprobe -v error \
                  -select_streams s:"$idx" \
                  -show_entries stream=codec_name \
                  -of default=noprint_wrappers=1:nokey=1 \
                  "$input")

  # …and the language tag (if any)  
  lang=$(ffprobe -v error \
                 -select_streams s:"$idx" \
                 -show_entries stream_tags=language \
                 -of default=noprint_wrappers=1:nokey=1 \
                 "$input")
  [[ -z "$lang" ]] && lang="und"

  # pick an extension based on codec
  case "$codec" in
    subrip)          ext="srt" ;;
    ass)             ext="ass" ;;
    webvtt)          ext="vtt" ;;
    hdmv_pgs_subtitle) ext="sup" ;;  # image‐based PGS
    *)               ext="sub" ;;   # fallback
  esac

  # extract
  ffmpeg -i "$input" \
         -map 0:$idx \
         -c:s copy \
         "${input%.*}.${lang}_${idx}.${ext}"
done