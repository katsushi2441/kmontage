#!/usr/bin/env bash
# landing/ を heteml(kmontage.exbridge.jp)へデプロイする。
# index.html=英語 / kmontage.html=日本語 / assets=Kurageアバター。
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . /home/kojima/work/aixec/.env; set +a

REMOTE="/web/kmontage_exbridge_jp"
for f in index.html kmontage.html assets/kurage_avatar.webp assets/kurage_avatar.png; do
  curl --fail --ftp-create-dirs -T "landing/$f" "ftp://${FTP_USER}:${FTP_PASS}@${FTP_HOST}${REMOTE}/$f"
  echo "deployed landing/$f"
done
echo "-> https://kmontage.exbridge.jp/"
