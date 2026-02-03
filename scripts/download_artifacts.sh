#!/bin/bash
# Downloads all artifacts from GitHub Actions runs
# Usage: ./download_artifacts.sh [limit]

# Default to last 50 runs to prevent filling disk/quota inadvertently
# You can increase this number: ./download_artifacts.sh 2000
LIMIT=${1:-50} 

echo "🔍 Fetching list of last $LIMIT runs..."

# Get run IDs
# uses current git repo context
run_ids=$(gh run list --limit "$LIMIT" --json databaseId --jq '.[].databaseId')

if [ -z "$run_ids" ]; then
    echo "❌ No runs found or error listing runs."
    exit 1
fi

mkdir -p artifacts_archive

echo "📥 Starting download..."

count=0
for run_id in $run_ids; do
    target_dir="artifacts_archive/run_$run_id"
    
    # Skip if already downloaded (simple check if dir exists and is not empty)
    if [ -d "$target_dir" ] && [ "$(ls -A $target_dir)" ]; then
        echo "⏭️  Run $run_id: Already downloaded, skipping."
        continue
    fi

    echo "📦 Downloading Run $run_id..."
    
    # Try to download
    # --dir specifies where to put the files
    mkdir -p "$target_dir"
    if gh run download "$run_id" --dir "$target_dir" > /dev/null 2>&1; then
        echo "   ✅ Success"
    else
        # If failure (often means no artifacts found for this run), remove the empty dir
        echo "   ⚠️  No artifacts found or expired"
        rm -rf "$target_dir"
    fi
    
    count=$((count + 1))
done

echo "🎉 Finished! Processed $count runs."
echo "📂 Artifacts are in: $(pwd)/artifacts_archive"
