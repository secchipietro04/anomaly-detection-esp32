#!/usr/bin/env bash

# This script runs clang-tidy static analysis on user firmware files,
# excluding built, managed, and generated components.

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. Try to source the ESP-IDF activation script if idf.py is not in PATH
if ! command -v idf.py &> /dev/null; then
    ACTIVATION_SCRIPT="/home/pietro/.espressif/tools/activate_idf_v6.0.2.sh"
    if [ -f "${ACTIVATION_SCRIPT}" ]; then
        echo "Loading ESP-IDF environment..."
        OLD_PATH="$PATH"
        while IFS= read -r line; do
            if [[ "$line" =~ ^[A-Za-z0-9_]+=.*$ ]]; then
                export "$line"
            fi
        done < <(bash "${ACTIVATION_SCRIPT}" -e)
        export PATH="${PATH}:${OLD_PATH}"
    else
        echo "Warning: idf.py not found and activation script not found at ${ACTIVATION_SCRIPT}."
    fi
fi

# Ensure PATH contains idf.py directory
if [ -n "${IDF_PATH}" ]; then
    export PATH="${IDF_PATH}/tools:${PATH}"
fi

# 2. Query compiler search paths to supply them to clang-tidy (resolves system headers like <cstddef>)
EXTRA_ARGS=()
GCC_BIN=$(command -v xtensa-esp32s3-elf-g++ || command -v xtensa-esp-elf-g++ || command -v g++)
if [ -n "${GCC_BIN}" ]; then
    echo "Querying compiler include paths from ${GCC_BIN}..."
    in_search_list=0
    while IFS= read -r line; do
        if [[ "$line" == "#include <...> search starts here:" ]]; then
            in_search_list=1
            continue
        fi
        if [[ "$line" == "End of search list." ]]; then
            in_search_list=0
            continue
        fi
        if [ "$in_search_list" -eq 1 ]; then
            # Trim leading/trailing whitespace
            path=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
            if [ -d "$path" ]; then
                EXTRA_ARGS+=("-extra-arg=-isystem${path}")
            fi
        fi
    done < <("${GCC_BIN}" -E -v -xc++ /dev/null 2>&1)
fi

# Query clang internal includes for stddef.h/cstddef resolution
CLANG_DIR="/home/pietro/.espressif/tools/esp-clang"
if [ -d "${CLANG_DIR}" ]; then
    INTERNAL_PATH=$(find "${CLANG_DIR}" -type d -path "*/lib/clang/*/include" -print -quit)
    if [ -n "${INTERNAL_PATH}" ]; then
        echo "Found clang internal include: ${INTERNAL_PATH}"
        EXTRA_ARGS+=("-extra-arg=-isystem${INTERNAL_PATH}")
    fi
fi

# 3. Parse command line arguments for --fix option
FIX_OPTION=""
FILTERED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--fix" ]]; then
        FIX_OPTION="--run-clang-tidy-options=-fix"
    else
        FILTERED_ARGS+=("$arg")
    fi
done

# 4. Run idf.py clang-check with exclusions
echo "Running static analysis using idf.py clang-check..."
idf.py clang-check \
    --exclude-paths "managed_components" \
    --exclude-paths "build" \
    --exclude-paths "main/cbor" \
    ${FIX_OPTION} \
    "${FILTERED_ARGS[@]}" \
    -- -header-filter="^(main/(?!cbor/)|components/).*$" \
    "${EXTRA_ARGS[@]}"
