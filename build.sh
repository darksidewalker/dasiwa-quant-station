#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
go build -o quantstation.next ./cmd/quantstation
mv quantstation.next quantstation
echo "Built quantstation"
