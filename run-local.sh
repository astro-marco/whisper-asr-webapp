#!/usr/bin/env bash
set -e

# 1) build frontend
pushd frontend
npm ci
npm run build
popd

# 2) deploy UI into static/
rm -rf static/*
cp -R frontend/dist/* static/

# 3) start backend using your existing .venv
pushd backend
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --reload
