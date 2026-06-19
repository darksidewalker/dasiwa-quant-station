#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
go build -o quantstation.next ./cmd/dasiwa
mv quantstation.next quantstation
echo "Built quantstation"
