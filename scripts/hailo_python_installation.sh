#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Hailo Python Wheels: downloader & installer
# Works with the main installer (calls this with --hailort-version/--tappas-core-version).
# Presets for H8/H10; overrideable via flags.
# ------------------------------------------------------------------------------

# Defaults (server snapshots)
BASE_URL_DEFAULT="http://dev-public.hailo.ai/2025-07"   # latest snapshot you referenced
BASE_URL="$BASE_URL_DEFAULT"

# Version presets (adjust here when bumping)
H8_HAILORT_VERSION="4.22.0"
H8_TAPPAS_VERSION="5.0.0"
H10_HAILORT_VERSION="5.0.0"
H10_TAPPAS_VERSION="5.0.0"

# Effective versions (can be overridden by flags)
HAILORT_VERSION=""
TAPPAS_CORE_VERSION=""

# Behavior flags
HW_ARCHITECTURE="H8"          # H8 | H10  (affects defaults if versions not passed)
DOWNLOAD_DIR="/usr/local/hailo/resources/deb_whl_packages"
DOWNLOAD_ONLY=false
QUIET=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --arch=(H8|H10)                Choose hardware preset for default versions (default: H8)
  --hailort-version=VER          Force a specific HailoRT wheel version (overrides preset)
  --tappas-core-version=VER      Force a specific TAPPAS core wheel version (overrides preset)
  --base-url=URL                 Override base URL (default: ${BASE_URL_DEFAULT})
  --download-dir=DIR             Where to place wheels (default: ${DOWNLOAD_DIR})
  --download-only                Only download wheels; do not install
  -q, --quiet                    Less output
  -h, --help                     Show this help

Notes:
- If you pass neither --hailort-version nor --tappas-core-version, the chosen --arch preset is used.
- If you pass only one of them, only that package is downloaded/installed.
EOF
}

log() { $QUIET || echo -e "$*"; }

# -------------------- Parse flags --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch=*)
      HW_ARCHITECTURE="${1#*=}"
      [[ "$HW_ARCHITECTURE" =~ ^(H8|H10)$ ]] || { echo "Invalid --arch: $HW_ARCHITECTURE"; exit 1; }
      shift
      ;;
    --hailort-version=*)
      HAILORT_VERSION="${1#*=}"
      shift
      ;;
    --tappas-core-version=*)
      TAPPAS_CORE_VERSION="${1#*=}"
      shift
      ;;
    --base-url=*)
      BASE_URL="${1#*=}"
      shift
      ;;
    --download-dir=*)
      DOWNLOAD_DIR="${1#*=}"
      shift
      ;;
    --download-only)
      DOWNLOAD_ONLY=true
      shift
      ;;
    -q|--quiet)
      QUIET=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

# -------------------- Resolve preset versions if missing --------------------
if [[ -z "$HAILORT_VERSION" || -z "$TAPPAS_CORE_VERSION" ]]; then
  if [[ "$HW_ARCHITECTURE" == "H10" ]]; then
    : "${HAILORT_VERSION:=$H10_HAILORT_VERSION}"
    : "${TAPPAS_CORE_VERSION:=$H10_TAPPAS_VERSION}"
  else
    : "${HAILORT_VERSION:=$H8_HAILORT_VERSION}"
    : "${TAPPAS_CORE_VERSION:=$H8_TAPPAS_VERSION}"
  fi
fi

# If user specified only one version, we install only that one.
INSTALL_HAILORT=false
INSTALL_TAPPAS=false
[[ -n "$HAILORT_VERSION" ]] && INSTALL_HAILORT=true
[[ -n "$TAPPAS_CORE_VERSION" ]] && INSTALL_TAPPAS=true

if [[ "$INSTALL_HAILORT" == false && "$INSTALL_TAPPAS" == false ]]; then
  log "Nothing to do (no versions requested)."
  exit 0
fi

# -------------------- Compute tags --------------------
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
PY_TAG="cp${PY_MAJOR}${PY_MINOR}-cp${PY_MAJOR}${PY_MINOR}"

# Map uname -m to wheel platform tag
UNAME_M="$(uname -m)"
case "$UNAME_M" in
  x86_64)  ARCH_TAG="linux_x86_64" ;;
  aarch64) ARCH_TAG="linux_aarch64" ;;
  *)
    echo "Unsupported architecture: $UNAME_M"
    exit 1
    ;;
esac

mkdir -p "$DOWNLOAD_DIR"

log "→ BASE_URL            = $BASE_URL"
log "→ ARCH preset         = $HW_ARCHITECTURE"
log "→ Python tag          = $PY_TAG"
log "→ Wheel arch tag      = $ARCH_TAG"
$INSTALL_HAILORT && log "→ HailoRT version     = $HAILORT_VERSION"
$INSTALL_TAPPAS && log "→ TAPPAS core version = $TAPPAS_CORE_VERSION"
log "→ Download dir        = $DOWNLOAD_DIR"
log "→ Download only?      = $DOWNLOAD_ONLY"

# -------------------- Helpers --------------------
fetch() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    log "  - Exists: $(basename "$out")"
    return 0
  fi
  log "  - GET $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
  else
    wget -q --tries=3 --timeout=20 -O "$out" "$url"
  fi
}

# -------------------- Download wheels --------------------
if [[ "$INSTALL_TAPPAS" == true ]]; then
  TAPPAS_FILE="tappas_core_python_binding-${TAPPAS_CORE_VERSION}-py3-none-any.whl"
  TAPPAS_URL="${BASE_URL}/${TAPPAS_FILE}"
  fetch "$TAPPAS_URL" "${DOWNLOAD_DIR}/${TAPPAS_FILE}"
fi

if [[ "$INSTALL_HAILORT" == true ]]; then
  HAILORT_FILE="hailort-${HAILORT_VERSION}-${PY_TAG}-${ARCH_TAG}.whl"
  HAILORT_URL="${BASE_URL}/${HAILORT_FILE}"
  fetch "$HAILORT_URL" "${DOWNLOAD_DIR}/${HAILORT_FILE}"
fi

if [[ "$DOWNLOAD_ONLY" == true ]]; then
  log "✅ Download(s) complete (download-only)."
  exit 0
fi

# -------------------- Install into current environment --------------------
log "→ Upgrading pip / wheel / setuptools…"
python3 -m pip install --upgrade pip setuptools wheel >/dev/null

if [[ "$INSTALL_HAILORT" == true ]]; then
  log "→ Installing HailoRT wheel…"
  python3 -m pip install "${DOWNLOAD_DIR}/${HAILORT_FILE}"
fi

if [[ "$INSTALL_TAPPAS" == true ]]; then
  log "→ Installing TAPPAS core wheel…"
  python3 -m pip install "${DOWNLOAD_DIR}/${TAPPAS_FILE}"
fi

log "✅ Installation complete."
