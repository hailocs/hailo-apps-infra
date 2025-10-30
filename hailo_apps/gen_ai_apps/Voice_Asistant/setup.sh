#!/bin/bash
# Complete setup script for Whisper+LLM
WHISPER_HEF_URL="https://dev-public.hailo.ai/v5.1.0/blob/Whisper-Base.hef"
LLM_HEF_URL="https://dev-public.hailo.ai/v5.1.0/blob/Qwen2.5-Coder-1.5B-Instruct.hef"

set -e  # Exit on error

# Parse command line arguments
USE_SYSTEM_SITE_PACKAGES=true  # Default to system-site-packages
for arg in "$@"; do
    case $arg in
        --no-system-site-packages)
        USE_SYSTEM_SITE_PACKAGES=false
        shift
        ;;
        --help|-h)
        echo "Usage: $0 [--no-system-site-packages]"
        echo ""
        echo "Options:"
        echo "  --no-system-site-packages  Create isolated virtual environment (no system packages)"
        echo "  --help, -h                Show this help message"
        echo ""
        echo "Default behavior:"
        echo "  - Uses system-site-packages (recommended for Raspberry Pi and most setups)"
        echo "  - On Raspberry Pi: Hailo packages are pre-installed via apt"
        echo "  - On other platforms: Use if you installed Hailo wheel files system-wide"
        echo ""
        echo "Use --no-system-site-packages only if:"
        echo "  - You want a completely isolated environment"
        echo "  - You plan to install Hailo wheel files in the virtual environment"
        exit 0
        ;;
    esac
done

echo "Starting app setup..."

# Install system dependencies (skip if already installed)
echo "Checking system dependencies..."
if ! dpkg -l | grep -q "portaudio19-dev\|python3-dev\|alsa-utils"; then
    echo "System dependencies not found. Please install manually:"
    echo "sudo apt update && sudo apt install -y portaudio19-dev python3-dev alsa-utils"
    echo "Then rerun the setup.sh script"
else
    echo "System dependencies already installed."
fi

# Check architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Check Hailo dependencies
echo "Checking Hailo dependencies..."
HAILO_AVAILABLE=false

if $USE_SYSTEM_SITE_PACKAGES; then
    echo "Checking for Hailo packages in system Python..."
    if pip show hailort >/dev/null 2>&1; then
        HAILO_VERSION=$(pip show hailort | grep "Version:" | awk '{print $2}')
        echo "✅ Hailo packages found in system Python (version: $HAILO_VERSION)"
        HAILO_AVAILABLE=true
    else
        echo "❌ Hailo packages not found in system Python"
    fi
else
    echo "Will check for Hailo packages in virtual environment after creation..."
fi

if ! $HAILO_AVAILABLE && ! $USE_SYSTEM_SITE_PACKAGES; then
    echo ""
    echo "⚠️  Hailo packages not found. You need to install them manually."
    echo ""
    echo "Required Hailo packages:"
    echo "  - hailort=5.1.0 (exact version required)"
    echo ""
    echo "Installation options:"
    echo "  1. Raspberry Pi: Install via RPi apt server"
    echo "  2. Other platforms: Download wheel file from Hailo repository and install with:"
    echo "     pip install <path-to-hailort-wheel-file>"
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Please install Hailo dependencies first."
        exit 1
    fi
fi


# Create virtual environment
if $USE_SYSTEM_SITE_PACKAGES; then
    echo "Creating virtual environment with system-site-packages..."
    if [ ! -d "venv_asr" ]; then
        python3 -m venv venv_asr --system-site-packages
    fi
else
    echo "Creating clean virtual environment..."
    if [ ! -d "venv_asr" ]; then
        python3 -m venv venv_asr
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_asr/bin/activate

# Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt --quiet

# Check Hailo packages in virtual environment if not using system-site-packages
if ! $USE_SYSTEM_SITE_PACKAGES; then
    echo "Checking for Hailo packages in virtual environment..."
    if pip show hailort >/dev/null 2>&1; then
        HAILO_VERSION=$(pip show hailort | grep "Version:" | awk '{print $2}')
        echo "✅ Hailo packages found in virtual environment (version: $HAILO_VERSION)"
    else
        echo "❌ Hailo packages not found in virtual environment"
        echo ""
        echo "You need to install Hailo packages manually:"
        echo "  1. Download hailort wheel file from Hailo repository"
        echo "  2. Install with: pip install <path-to-hailort-wheel-file>"
        echo ""
        echo "Or re-run setup without --no-system-site-packages if you have them installed system-wide."
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Setup cancelled. Please install Hailo dependencies first."
            exit 1
        fi
    fi
fi

# Download Piper TTS voice
echo "Downloading Piper TTS voice..."
python3 -m piper.download_voices en_US-amy-low

# Download LLM model
if [ ! -f "Qwen2.5-Coder-1.5B-Instruct.hef" ]; then
    echo "Downloading LLM model..."
    wget "$LLM_HEF_URL" -O Qwen2.5-Coder-1.5B-Instruct.hef
else
    echo "LLM model already exists."
fi

# Download Whisper model
if [ ! -f "Whisper-Base.hef" ]; then
    echo "Downloading Whisper model..."
    wget "$WHISPER_HEF_URL" -O Whisper-Base.hef
else
    echo "Whisper model already exists."
fi

echo "Setup completed successfully!"
