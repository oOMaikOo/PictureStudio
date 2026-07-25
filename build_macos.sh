#!/usr/bin/env bash
#
# Build the Picture Studio macOS .app bundle and a DMG (local/ad-hoc).
#
# Usage:
#   ./build_macos.sh            # full build: .app + .dmg
#   ./build_macos.sh --no-dmg   # only the .app bundle
#
set -euo pipefail

cd "$(dirname "$0")"

APP_NAME="Picture Studio"
APP_PATH="dist/${APP_NAME}.app"
DMG_PATH="dist/Picture-Studio-2.5.1.dmg"
PY="${PYTHON:-.venv/bin/python}"

echo "▶ 1/4  Clean previous build…"
rm -rf build "dist/${APP_NAME}.app" "dist/${APP_NAME}" "$DMG_PATH"

echo "▶ 2/4  Run PyInstaller…"
"$PY" -m PyInstaller --noconfirm Picture.spec

if [[ ! -d "$APP_PATH" ]]; then
    echo "✗ Build failed: $APP_PATH not found" >&2
    exit 1
fi

echo "▶ 3/4  Ad-hoc code signing (local use)…"
# Ad-hoc signature ("-") lets the app run on THIS Mac without a Developer ID.
codesign --force --deep --sign - "$APP_PATH"
codesign --verify --deep --strict "$APP_PATH" && echo "  ✓ signature valid"

if [[ "${1:-}" == "--no-dmg" ]]; then
    echo "✓ Done: $APP_PATH (DMG skipped)"
    exit 0
fi

echo "▶ 4/4  Build DMG…"
STAGE="$(mktemp -d)"
cp -R "$APP_PATH" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "$APP_NAME" \
    -srcfolder "$STAGE" -ov -format UDZO "$DMG_PATH"
rm -rf "$STAGE"

echo
echo "✓ Done."
echo "   App: $APP_PATH"
echo "   DMG: $DMG_PATH"
echo
echo "Erststart (lokal): per Rechtsklick → Öffnen, oder Quarantäne entfernen:"
echo "   xattr -dr com.apple.quarantine \"$APP_PATH\""
