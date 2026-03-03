#!/bin/bash
# Uso: ./push.sh "messaggio di commit"
#   oppure semplicemente ./push.sh  (usa messaggio automatico)

cd "$(dirname "$0")"

MSG="${1:-auto: $(date '+%Y-%m-%d %H:%M')}"

git add -A
git commit -m "$MSG"
git push -u origin master
