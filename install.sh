#!/usr/bin/env bash
set -euo pipefail

# SLMFS — Developer Install Script
# Detects OS, installs dependencies, builds C++ engine, installs Python package.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/eccyan/slmfs/main/install.sh | bash
#   # or locally:
#   ./install.sh

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BOLD}[slmfs]${NC} $*"; }
ok()    { echo -e "${GREEN}[slmfs]${NC} $*"; }
warn()  { echo -e "${YELLOW}[slmfs]${NC} $*"; }
fail()  { echo -e "${RED}[slmfs]${NC} $*"; exit 1; }

# ── Detect OS ──

OS="$(uname -s)"
ARCH="$(uname -m)"

info "Detected: ${OS} ${ARCH}"

# ── Install system dependencies ──

install_deps_macos() {
    if ! command -v brew &>/dev/null; then
        fail "Homebrew not found. Install from https://brew.sh"
    fi

    info "Installing macOS dependencies via Homebrew..."

    # C++ compiler: Xcode Command Line Tools (comes with clang)
    if ! xcode-select -p &>/dev/null; then
        info "Installing Xcode Command Line Tools..."
        xcode-select --install 2>/dev/null || true
        warn "Please complete the Xcode CLT install dialog, then re-run this script."
        exit 0
    fi

    # CMake
    if ! command -v cmake &>/dev/null; then
        brew install cmake
    fi
    ok "CMake: $(cmake --version | head -1)"

    # SQLite (usually pre-installed on macOS)
    if ! pkg-config --exists sqlite3 2>/dev/null; then
        brew install sqlite3
    fi
    ok "SQLite3: available"

    # FUSE-T (optional, for filesystem mount)
    if ! [ -f /usr/local/lib/libfuse-t.dylib ]; then
        info "Installing FUSE-T (no kernel extensions needed)..."
        brew install macos-fuse-t/homebrew-cask/fuse-t || warn "FUSE-T install failed — filesystem mount won't work, but engine + init + add still work"
    else
        ok "FUSE-T: already installed"
    fi

    # Python
    if ! command -v python3 &>/dev/null; then
        brew install python@3.12
    fi
    ok "Python: $(python3 --version)"
}

install_deps_linux() {
    info "Installing Linux dependencies..."

    if command -v apt-get &>/dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            libsqlite3-dev \
            libfuse3-dev \
            python3 \
            python3-pip \
            python3-venv

        # Prefer GCC 14 if available, fall back to GCC 13, then system default
        if apt-cache show g++-14 &>/dev/null 2>&1; then
            sudo apt-get install -y g++-14
            export CC=gcc-14 CXX=g++-14
        elif apt-cache show g++-13 &>/dev/null 2>&1; then
            sudo apt-get install -y g++-13
            export CC=gcc-13 CXX=g++-13
        else
            warn "Using system GCC ($(g++ --version | head -1)). C++23 requires GCC 13+."
        fi
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y \
            gcc-c++ \
            cmake \
            sqlite-devel \
            fuse3-devel \
            python3 \
            python3-pip
    elif command -v pacman &>/dev/null; then
        sudo pacman -Sy --noconfirm \
            base-devel \
            cmake \
            sqlite \
            fuse3 \
            python \
            python-pip
    else
        fail "Unsupported package manager. Install manually: cmake, g++14+, sqlite3, libfuse3, python3"
    fi

    ok "System dependencies installed"
}

case "$OS" in
    Darwin) install_deps_macos ;;
    Linux)  install_deps_linux ;;
    *)      fail "Unsupported OS: $OS" ;;
esac

# ── Build C++ engine ──

info "Building C++ engine..."

# Determine repo location. BASH_SOURCE may be unset in curl|bash mode.
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# If running from the repo, build in place. Otherwise, clone first.
if [ -n "$SCRIPT_DIR" ] && [ -f "${SCRIPT_DIR}/CMakeLists.txt" ]; then
    cd "$SCRIPT_DIR"
elif [ -f "./CMakeLists.txt" ]; then
    : # already in repo root
else
    # Ensure git is available for cloning
    if ! command -v git &>/dev/null; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y git
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y git
        elif command -v pacman &>/dev/null; then
            sudo pacman -Sy --noconfirm git
        else
            fail "git not found. Please install git first."
        fi
    fi
    info "Cloning SLMFS repository..."
    git clone https://github.com/eccyan/slmfs.git
    cd slmfs
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

ok "Engine built: build/src/engine/slmfs_engine"

# ── Run C++ tests ──

info "Running C++ tests..."
cd build && ctest --output-on-failure
cd ..
ok "All C++ tests passed"

# ── Install Python package ──

info "Installing Python frontend..."

cd python

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"

cd ..

ok "Python package installed"

# ── Run Python tests ──

info "Running Python tests..."
cd python && .venv/bin/python -m pytest ../tests/python/ -v
cd ..
ok "All Python tests passed"

# ── Install engine binary ──

INSTALL_DIR="${HOME}/.local/bin"
mkdir -p "$INSTALL_DIR"

cp build/src/engine/slmfs_engine "$INSTALL_DIR/slmfs_engine"
chmod +x "$INSTALL_DIR/slmfs_engine"

ok "Engine installed to ${INSTALL_DIR}/slmfs_engine"

# ── Summary ──

REPO_DIR="$(pwd)"

echo ""
echo -e "${BOLD}━━━ SLMFS installed successfully ━━━${NC}"
echo ""
echo "  Engine:  ${INSTALL_DIR}/slmfs_engine"
echo "  Python:  ${REPO_DIR}/python/slmfs/"
echo "  Repo:    ${REPO_DIR}"
echo ""
echo -e "${BOLD}Quick start:${NC}"
echo "  # 1. Migrate existing notes"
echo "  cd ${REPO_DIR}"
echo "  source python/.venv/bin/activate"
echo "  python -m slmfs init ~/MEMORY.md"
echo ""
echo "  # 2. Start engine"
echo "  slmfs_engine --db-path=.slmfs/memory.db"
echo ""
echo "  # 3. Mount filesystem (requires FUSE)"
echo "  python -m slmfs fuse --mount=.agent_memory"
echo ""

if ! echo "$PATH" | grep -q "${INSTALL_DIR}"; then
    warn "Add ${INSTALL_DIR} to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
fi
