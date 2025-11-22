#!/bin/bash

#================================================================
# SLURM Job Chain Manager for Image Processing
# Processes multiple directories sequentially with dependencies
#================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

#----------------------------------------------------------------
# USAGE
#----------------------------------------------------------------

usage() {
    cat << 'EOF'
Usage: process_images_chain.sh [OPTIONS]

DESCRIPTION:
  This script manages a SLURM job chain for processing images in multiple 
  directories sequentially. Each directory's images are processed as parallel 
  array jobs, with dependencies ensuring sequential processing across 
  directories.

REQUIRED ARGUMENTS:
  -d, --directories <dirs_file>    File with directory paths (one per line) or "auto" for auto-discovery
  -o, --outdir <path>              Output directory for results
  -v, --venv <path>                Path to virtual environment
  -s, --script <path>              Path to phenotypic.pipeline JSON (used by inline Python)

OPTIONAL SLURM ARGUMENTS:
  -c, --cpus <num>                 CPUs per task (default: 6)
  -m, --memory <mem>               Memory per task (default: 14G)
  -t, --time <time>                Time limit HH:MM:SS (default: 2:00:00)
  -p, --partition <part>           SLURM partition (default: short)
  --max-concurrent <num>           Max concurrent array tasks (default: 64)
  -l, --log-mode <mode>            Log mode: separate, combined, or single (default: combined)

IMAGE READING OPTIONS:
  --nrows <int>                    Number of rows for GridImage (default: 8)
  --ncols <int>                    Number of columns for GridImage (default: 12)

OPTIONAL EMAIL:
  -e, --email <email>              Email for job notifications (optional)

EXAMPLES:
  # With directory file
  process_images_chain.sh -d dirs.txt -o ./data -v .venv -s ./my_pipeline.json

  # With auto-discovery
  process_images_chain.sh -d auto -o ~/data -v .venv -s ./my_pipeline.json --email user@ucr.edu

  # With custom SLURM settings
  process_images_chain.sh -d dirs.txt -o ~/data -v ~/.venv -s ./my_pipeline.json \
    -c 8 -m 20G -t 4:00:00 -p long --nrows 8 --ncols 12
EOF
    exit 1
}

#----------------------------------------------------------------
# PARSE ARGUMENTS
#----------------------------------------------------------------

OUTDIR=""
VENV_PATH=""
PIPELINE_JSON=""
DIRECTORIES_INPUT=""
EMAIL=""
CPUS_PER_TASK=6
MEMORY="14G"
TIME_LIMIT="2:00:00"
PARTITION="short"
MAX_CONCURRENT=64
LOG_MODE="combined"
NROWS=8
NCOLS=12

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directories)
            DIRECTORIES_INPUT="$2"
            shift 2
            ;;
        -o|--outdir)
            OUTDIR="$2"
            shift 2
            ;;
        -v|--venv)
            VENV_PATH="$2"
            shift 2
            ;;
        -s|--script)
            PIPELINE_JSON="$2"
            shift 2
            ;;
        -e|--email)
            EMAIL="$2"
            shift 2
            ;;
        -c|--cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        -l|--log-mode)
            LOG_MODE="$2"
            shift 2
            ;;
        --nrows)
            NROWS="$2"
            shift 2
            ;;
        --ncols)
            NCOLS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            ;;
    esac
done

#----------------------------------------------------------------
# VALIDATE REQUIRED ARGUMENTS
#----------------------------------------------------------------

if [ -z "$OUTDIR" ] || [ -z "$VENV_PATH" ] || [ -z "$PIPELINE_JSON" ] || [ -z "$DIRECTORIES_INPUT" ]; then
    echo "ERROR: Missing required arguments"
    usage
fi

#----------------------------------------------------------------
# SETUP
#----------------------------------------------------------------

# Create main log directory
MAIN_LOG_DIR="./slurm_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MAIN_LOG_DIR"

# Job tracking file
JOB_TRACKING_FILE="${MAIN_LOG_DIR}/job_chain.log"
echo "=== Job Chain Started: $(date) ===" > "$JOB_TRACKING_FILE"
echo "Log Directory: $MAIN_LOG_DIR" >> "$JOB_TRACKING_FILE"
echo "Log Mode: $LOG_MODE" >> "$JOB_TRACKING_FILE"
echo "" >> "$JOB_TRACKING_FILE"

#----------------------------------------------------------------
# LOAD DIRECTORIES
#----------------------------------------------------------------

DIRECTORIES=()

if [ "$DIRECTORIES_INPUT" = "auto" ]; then
    # Auto-discover subdirectories from OUTDIR
    echo "Auto-discovering directories in: $OUTDIR"
    mapfile -t DIRECTORIES < <(find "$OUTDIR" -mindepth 1 -maxdepth 1 -type d | sort)
elif [ -f "$DIRECTORIES_INPUT" ]; then
    # Read from file (skip comments and empty lines)
    echo "Reading directories from: $DIRECTORIES_INPUT"
    mapfile -t DIRECTORIES < <(grep -v '^#' "$DIRECTORIES_INPUT" | grep -v '^[[:space:]]*$')
else
    echo "ERROR: Directories input must be 'auto' or a valid file path"
    echo "  Provided: $DIRECTORIES_INPUT"
    exit 1
fi

#----------------------------------------------------------------
# VALIDATION
#----------------------------------------------------------------

if [ ${#DIRECTORIES[@]} -eq 0 ]; then
    echo "ERROR: No directories specified"
    exit 1
fi

echo "Found ${#DIRECTORIES[@]} directories to process"
echo "Log directory: $MAIN_LOG_DIR"
echo ""

#----------------------------------------------------------------
# MAIN PROCESSING LOOP
#----------------------------------------------------------------

PREV_JOB=""
JOB_COUNT=0

for IMAGE_DIR in "${DIRECTORIES[@]}"; do
    
    # Validate directory exists
    if [ ! -d "$IMAGE_DIR" ]; then
        echo "WARNING: Directory '$IMAGE_DIR' does not exist, skipping"
        echo "SKIPPED,$IMAGE_DIR,Directory not found" >> "$JOB_TRACKING_FILE"
        continue
    fi
    
    DIR_NAME=$(basename "$IMAGE_DIR")
    echo "===== Processing: $DIR_NAME ====="
    
    # Count images in directory
    shopt -s nullglob
    image_files=("$IMAGE_DIR"/*.{jpg,jpeg,JPG,JPEG,tif,tiff,TIF,TIFF,png,PNG})
    NUM_IMAGES=${#image_files[@]}
    shopt -u nullglob
    
    if [ $NUM_IMAGES -eq 0 ]; then
        echo "  WARNING: No images found in '$IMAGE_DIR', skipping"
        echo "SKIPPED,$IMAGE_DIR,No images found" >> "$JOB_TRACKING_FILE"
        continue
    fi
    
    echo "  Images found: $NUM_IMAGES"
    
    # Create per-directory log subdirectory
    DIR_LOG_PATH="${MAIN_LOG_DIR}/${DIR_NAME}"
    mkdir -p "$DIR_LOG_PATH"
    
    # Build dependency flag
    if [ -z "$PREV_JOB" ]; then
        DEPENDENCY_FLAG=""
        DEP_MSG="none"
    else
        DEPENDENCY_FLAG="--dependency=afterany:$PREV_JOB"
        DEP_MSG="$PREV_JOB"
    fi
    
    # Determine output/error file configuration based on LOG_MODE
    case "$LOG_MODE" in
        "separate")
            OUTPUT_FILE="${DIR_LOG_PATH}/${DIR_NAME}_%A_%a.out"
            ERROR_FILE="${DIR_LOG_PATH}/${DIR_NAME}_%A_%a.error"
            ;;
        "combined")
            OUTPUT_FILE="${DIR_LOG_PATH}/${DIR_NAME}_%A_%a.log"
            ERROR_FILE="${DIR_LOG_PATH}/${DIR_NAME}_%A_%a.log"
            ;;
        "single")
            OUTPUT_FILE="${DIR_LOG_PATH}/${DIR_NAME}_%A.log"
            ERROR_FILE="${DIR_LOG_PATH}/${DIR_NAME}_%A.log"
            echo "  WARNING: Using single log file - possible write conflicts"
            ;;
        *)
            echo "ERROR: Invalid LOG_MODE: $LOG_MODE"
            exit 1
            ;;
    esac
    
    # Determine mail type (only final job sends completion email)
    JOB_COUNT=$((JOB_COUNT + 1))
    if [ $JOB_COUNT -eq ${#DIRECTORIES[@]} ]; then
        MAIL_TYPE="ALL"
        echo "  This is the final job - will send completion email"
    else
        MAIL_TYPE="FAIL"
    fi
    
    # Build email directive (only if email is provided)
    MAIL_DIRECTIVE=""
    if [ -n "$EMAIL" ]; then
        MAIL_DIRECTIVE="#SBATCH --mail-user=$EMAIL"
    fi
    
    # Submit the array job
    CURRENT_JOB=$(sbatch --parsable $DEPENDENCY_FLAG <<EOF
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEMORY
#SBATCH --time=$TIME_LIMIT
$MAIL_DIRECTIVE
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --job-name="pheno-${DIR_NAME}"
#SBATCH -p $PARTITION
#SBATCH --array=1-${NUM_IMAGES}%${MAX_CONCURRENT}
#SBATCH --output=$OUTPUT_FILE
#SBATCH --error=$ERROR_FILE

# Print job information
echo "======================================"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Array Task ID: \${SLURM_ARRAY_TASK_ID}"
echo "Directory: $IMAGE_DIR"
echo "Node: \${SLURMD_NODENAME}"
echo "Start Time: \$(date)"
echo "======================================"

# Load images into array
shopt -s nullglob
IMAGES=("$IMAGE_DIR"/*.{jpg,jpeg,JPG,JPEG,tif,tiff,TIF,TIFF,png,PNG})
shopt -u nullglob

# Get current image file
CURRENT_IMAGE="\${IMAGES[\$((SLURM_ARRAY_TASK_ID-1))]}"

if [ -z "\$CURRENT_IMAGE" ]; then
    echo "ERROR: No image found for task ID \${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Processing: \$CURRENT_IMAGE"
echo ""

# Activate virtual environment
source $VENV_PATH/bin/activate

# Run the processing pipeline (inline Python using the provided pipeline JSON)
python - "$PIPELINE_JSON" "\$CURRENT_IMAGE" "$OUTDIR" "$NROWS" "$NCOLS" <<'PYINLINED'
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

try:
    from phenotypic import Image, GridImage, ImagePipeline
except Exception as e:
    print(f"Failed to import phenotypic: {e}", file=sys.stderr)
    sys.exit(1)

def main(pipeline_json: Path, image_path: Path, output_dir: Path, nrows: int, ncols: int) -> int:
    # Ensure output subdirectories
    meas_dir = output_dir / 'measurements'
    overlay_dir = output_dir / 'overlays'
    meas_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    try:
        pipeline = ImagePipeline.from_json(pipeline_json)
    except Exception as e:
        print(f"Failed to load pipeline '{pipeline_json}': {e}", file=sys.stderr)
        return 1

    # Read image (default to GridImage as in CLI, with provided plate layout)
    try:
        image = GridImage.imread(image_path, nrows=nrows, ncols=ncols)
    except Exception:
        # Fallback to plain Image if GridImage reading fails
        try:
            image = Image.imread(image_path)
        except Exception as e2:
            print(f"Failed to read image '{image_path}': {e2}", file=sys.stderr)
            return 1

    # Apply pipeline and measure
    try:
        meas = pipeline.apply_and_measure(image, inplace=True)
    except Exception as e:
        print(f"Pipeline failed on '{image_path.name}': {e}", file=sys.stderr)
        return 1

    # Save outputs
    try:
        stem = image_path.stem
        meas.to_csv(meas_dir / f"{stem}.csv", index=False)
        fig, ax = image.show_overlay()
        fig.savefig(overlay_dir / f"{stem}.png", bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Failed to save outputs for '{image_path.name}': {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python - PIPELINE_JSON IMAGE_PATH OUTPUT_DIR NROWS NCOLS", file=sys.stderr)
        sys.exit(2)
    pipeline_json = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    try:
        nrows = int(sys.argv[4])
        ncols = int(sys.argv[5])
    except Exception:
        print("NROWS and NCOLS must be integers", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(pipeline_json, image_path, output_dir, nrows, ncols))
PYINLINED

EXIT_CODE=\$?

echo ""
echo "======================================"
echo "Exit Code: \$EXIT_CODE"
echo "End Time: \$(date)"
echo "======================================"

exit \$EXIT_CODE
EOF
)
    
    # Log the job submission
    echo "  Job ID: $CURRENT_JOB"
    echo "  Depends on: $DEP_MSG"
    echo "  Log path: $DIR_LOG_PATH"
    echo "$CURRENT_JOB,$IMAGE_DIR,$NUM_IMAGES,$DEP_MSG" >> "$JOB_TRACKING_FILE"
    echo ""
    
    # Update for next iteration
    PREV_JOB=$CURRENT_JOB
    
done

#----------------------------------------------------------------
# SUMMARY
#----------------------------------------------------------------

echo "========================================="
echo "All jobs submitted successfully!"
echo "========================================="
echo "Total directories: ${#DIRECTORIES[@]}"
echo "Jobs submitted: $JOB_COUNT"
echo "Final job ID: $PREV_JOB"
echo "Tracking file: $JOB_TRACKING_FILE"
echo "Log directory: $MAIN_LOG_DIR"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "Check job details:"
echo "  ./check_job_status.sh $JOB_TRACKING_FILE"
echo ""
echo "Combine logs (after completion):"
echo "  ./combine_logs.sh $MAIN_LOG_DIR"
echo "========================================="

# Save final job ID for easy reference
echo "$PREV_JOB" > "${MAIN_LOG_DIR}/final_job_id.txt"
