#!/bin/bash
# Download paper files from frontier-evals repo for local caching
# This bypasses Docker Git LFS issues by downloading directly from GitHub raw URLs

set -e

CACHE_DIR="${1:-./paper_cache}"
TASKS="${2:-pinn,bam,rice,lbcs}"  # comma-separated list of tasks to download

mkdir -p "$CACHE_DIR"

GITHUB_RAW_BASE="https://github.com/openai/preparedness/raw/main/project/paperbench/data/papers"

# Convert comma-separated to array
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

for task in "${TASK_ARRAY[@]}"; do
    task=$(echo "$task" | xargs)  # trim whitespace
    echo "📥 Downloading files for task: $task"
    
    TARGET_DIR="$CACHE_DIR/$task"
    mkdir -p "$TARGET_DIR/assets"
    
    # Download paper.md
    echo "  Downloading paper.md..."
    if curl -sL "$GITHUB_RAW_BASE/$task/paper.md" -o "$TARGET_DIR/paper.md"; then
        # Verify it's not an LFS pointer
        if head -1 "$TARGET_DIR/paper.md" | grep -q "version https://git-lfs"; then
            echo "❌ Failed: $task/paper.md is still an LFS pointer"
            rm -f "$TARGET_DIR/paper.md"
        else
            echo "  ✅ paper.md downloaded ($(wc -c < "$TARGET_DIR/paper.md") bytes)"
        fi
    else
        echo "  ❌ Failed to download paper.md"
    fi
    
    # Download addendum.md (if exists)
    echo "  Downloading addendum.md..."
    if curl -sL "$GITHUB_RAW_BASE/$task/addendum.md" -o "$TARGET_DIR/addendum.md" 2>/dev/null; then
        if head -1 "$TARGET_DIR/addendum.md" 2>/dev/null | grep -q "version https://git-lfs"; then
            rm -f "$TARGET_DIR/addendum.md"
        elif [ -s "$TARGET_DIR/addendum.md" ]; then
            echo "  ✅ addendum.md downloaded"
        else
            rm -f "$TARGET_DIR/addendum.md"
        fi
    fi
    
    # Download paper.pdf
    echo "  Downloading paper.pdf..."
    if curl -sL "$GITHUB_RAW_BASE/$task/paper.pdf" -o "$TARGET_DIR/paper.pdf" 2>/dev/null; then
        # PDF should be large (not a pointer)
        PDF_SIZE=$(wc -c < "$TARGET_DIR/paper.pdf" 2>/dev/null || echo "0")
        if [ "$PDF_SIZE" -gt 1000 ]; then
            echo "  ✅ paper.pdf downloaded ($PDF_SIZE bytes)"
        else
            rm -f "$TARGET_DIR/paper.pdf"
            echo "  ⚠️  paper.pdf skipped (too small or failed)"
        fi
    fi
    
    # Download config.yaml (not LFS, should work)
    echo "  Downloading config.yaml..."
    if curl -sL "$GITHUB_RAW_BASE/$task/config.yaml" -o "$TARGET_DIR/config.yaml" 2>/dev/null; then
        if [ -s "$TARGET_DIR/config.yaml" ]; then
            echo "  ✅ config.yaml downloaded"
        else
            rm -f "$TARGET_DIR/config.yaml"
        fi
    fi
    
    # Download rubric.json (not LFS, should work)
    echo "  Downloading rubric.json..."
    if curl -sL "$GITHUB_RAW_BASE/$task/rubric.json" -o "$TARGET_DIR/rubric.json" 2>/dev/null; then
        if [ -s "$TARGET_DIR/rubric.json" ]; then
            echo "  ✅ rubric.json downloaded"
        else
            rm -f "$TARGET_DIR/rubric.json"
        fi
    fi
    
    # Download assets (try common image names)
    echo "  Downloading assets..."
    ASSET_COUNT=0
    for i in $(seq 1 30); do
        for ext in jpg png gif jpeg; do
            ASSET_URL="$GITHUB_RAW_BASE/$task/assets/asset_$i.$ext"
            ASSET_FILE="$TARGET_DIR/assets/asset_$i.$ext"
            
            if curl -sL --fail "$ASSET_URL" -o "$ASSET_FILE" 2>/dev/null; then
                ASSET_SIZE=$(wc -c < "$ASSET_FILE" 2>/dev/null || echo "0")
                if [ "$ASSET_SIZE" -gt 1000 ]; then
                    ASSET_COUNT=$((ASSET_COUNT + 1))
                else
                    rm -f "$ASSET_FILE"
                fi
            else
                rm -f "$ASSET_FILE" 2>/dev/null
            fi
        done
    done
    
    if [ "$ASSET_COUNT" -gt 0 ]; then
        echo "  ✅ Downloaded $ASSET_COUNT assets"
    else
        rmdir "$TARGET_DIR/assets" 2>/dev/null || true
    fi
    
    echo "  📁 Contents of $TARGET_DIR:"
    ls -la "$TARGET_DIR/"
    echo ""
done

echo "📁 Paper cache summary:"
du -sh "$CACHE_DIR"/*
