#!/bin/bash
set -e

# ========================================
# Garner Table Downloader
# ========================================

# Configuration
BUCKET_NAME="lqg-crt-inverse-matrix-data"
REGION="eu-north-1"

# Construct base URL - removed the VERSION variable and garner-tables/ prefix
# since your files are directly in the bucket root
if [ "$REGION" = "us-east-1" ]; then
    BASE_URL="https://${BUCKET_NAME}.s3.amazonaws.com"
else
    BASE_URL="https://${BUCKET_NAME}.s3.${REGION}.amazonaws.com"
fi

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "Garner Table Downloader"
echo -e "==========================================${NC}"
echo ""
echo "Available tables (S3 hosted):"
echo -e "  ${YELLOW}[1]${NC} garner_k10k.bin  (800 MB)   - up to 2.5M bits"
echo -e "  ${YELLOW}[2]${NC} garner_k20k.bin  (3.2 GB)   - up to 5M bits"
echo -e "  ${YELLOW}[3]${NC} garner_k40k.bin  (12.8 GB)  - up to 10M bits"
echo -e "  ${YELLOW}[4]${NC} All tables above 4k"
echo ""
read -p "Select option [1-4]: " choice

# Function to download a single file
download_file() {
    local filename=$1
    local filesize=$2
    local url="${BASE_URL}/${filename}"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check if file already exists
    if [ -f "${filename}" ]; then
        echo -e "${YELLOW}⚠️  ${filename} already exists.${NC}"
        read -p "Overwrite? [y/N]: " overwrite
        if [[ ! $overwrite =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}Skipping ${filename}${NC}"
            return
        fi
        rm "${filename}"
    fi
    
    echo -e "${BLUE}📥 Downloading ${filename} (${filesize})...${NC}"
    echo -e "${BLUE}   From: ${url}${NC}"
    
    # Try wget first, fall back to curl
    if command -v wget &> /dev/null; then
        wget --progress=bar:force -O "${filename}" "${url}" 2>&1
        download_status=$?
    elif command -v curl &> /dev/null; then
        curl -# -L -o "${filename}" "${url}"
        download_status=$?
    else
        echo -e "${RED}❌ Error: Neither wget nor curl found.${NC}"
        echo "Please install wget or curl:"
        echo "  Ubuntu/Debian: sudo apt-get install wget"
        echo "  macOS: brew install wget"
        exit 1
    fi
    
    # Check download status
    if [ $download_status -eq 0 ]; then
        echo -e "${GREEN}✅ Downloaded ${filename} successfully${NC}"
        
        # Show file size
        if command -v du &> /dev/null; then
            actual_size=$(du -h "${filename}" | cut -f1)
            echo -e "${GREEN}   Size: ${actual_size}${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to download ${filename}${NC}"
        echo "Please check:"
        echo "  1. Internet connection"
        echo "  2. S3 bucket permissions (should be public-read)"
        echo "  3. URL is correct: ${url}"
        exit 1
    fi
}

# Function to verify download (optional, if you provide checksums)
verify_file() {
    local filename=$1
    local checksum_file="${filename}.sha256"
    local url="${BASE_URL}/${checksum_file}"
    
    # Try to download checksum file
    if command -v wget &> /dev/null; then
        wget -q -O "${checksum_file}" "${url}" 2>/dev/null || return
    elif command -v curl &> /dev/null; then
        curl -s -L -o "${checksum_file}" "${url}" 2>/dev/null || return
    else
        return
    fi
    
    if [ -f "${checksum_file}" ]; then
        echo -e "${BLUE}🔍 Verifying integrity...${NC}"
        if sha256sum -c "${checksum_file}" 2>/dev/null; then
            echo -e "${GREEN}✅ Checksum verified${NC}"
            rm "${checksum_file}"
        else
            echo -e "${RED}⚠️  Checksum mismatch! File may be corrupted.${NC}"
            rm "${checksum_file}"
        fi
    fi
}

# Main download logic
case $choice in
    1)
        download_file "garner_k10k.bin" "800 MB"
        verify_file "garner_k10k.bin"
        ;;
    2)
        download_file "garner_k20k.bin" "3.2 GB"
        verify_file "garner_k20k.bin"
        ;;
    3)
        download_file "garner_k40k.bin" "12.8 GB"
        verify_file "garner_k40k.bin"
        ;;
    4)
        echo -e "${YELLOW}Downloading all large tables...${NC}"
        download_file "garner_k10k.bin" "800 MB"
        verify_file "garner_k10k.bin"
        
        download_file "garner_k20k.bin" "3.2 GB"
        verify_file "garner_k20k.bin"
        
        download_file "garner_k40k.bin" "12.8 GB"
        verify_file "garner_k40k.bin"
        ;;
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=========================================="
echo "✅ Download scomplete!"
echo -e "==========================================${NC}"
echo "Files are ready to use in the data/ directory."
echo ""
echo "You can now run benchmarks requiring these tables."
```

