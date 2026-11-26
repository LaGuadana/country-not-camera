#!/bin/sh
# Resumable download with retries: fetch.sh URL OUTFILE EXPECTED_BYTES
# Works with GNU (stat -c) and BSD/macOS (stat -f) stat.
url="$1"; out="$2"; want="$3"
size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null || echo 0; }
for i in $(seq 1 60); do
  have=$(size "$out")
  [ "$have" -ge "$want" ] && { echo "OK $out $have"; exit 0; }
  curl -sS -L --http1.1 -C - --retry 5 --retry-delay 2 -o "$out" "$url" || echo "retry $i (have $have)"
  sleep 1
done
echo "FAILED $out"; exit 1
