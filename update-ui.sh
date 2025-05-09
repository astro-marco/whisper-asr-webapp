#!/usr/bin/env bash
set -e

# 1) build del front-end
cd frontend
npm run build

# 2) copia dei file in static/
cd ..
rm -rf static/*
cp -R frontend/dist/* static/

echo "[`date +'%H:%M:%S'`] UI rebuilt"