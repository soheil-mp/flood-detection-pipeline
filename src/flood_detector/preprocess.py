"""
preprocess.py

This script handles the preprocessing of raw satellite data, particularly
Sentinel-1 SAR imagery, using the ESA SNAP Graph Processing Tool (GPT).

This script is designed to be a command-line interface to a predefined
SNAP processing graph (.xml file). The graph should be created in the SNAP
Desktop application and should define the full preprocessing chain.

A typical Sentinel-1 preprocessing chain for flood mapping includes:
1. Apply Orbit File
2. Thermal Noise Removal
3. Radiometric Calibration (to Sigma0)
4. Speckle Filtering (e.g., Refined Lee)
5. Geometric Terrain Correction (using a DEM)
6. Conversion to decibels (dB)

Usage:
    python src/flood_detector/preprocess.py \
        --input_path path/to/raw/S1_image.zip \
        --output_path path/to/processed/S1_image_processed.tif \
        --graph_xml path/to/your/graph.xml
"""

import argparse
import os
import subprocess
import sys

# Import configuration from the config file
from flood_detector import config


def run_snap_gpt(input_path, output_path, graph_xml_path):
    """
    Executes a SNAP GPT graph for preprocessing.

    This function constructs and runs a command-line call to the SNAP GPT.
    It passes the input file, output file, and other parameters to the graph.

    Args:
        input_path (str): Path to the raw input Sentinel-1 product (.zip).
        output_path (str): Path where the processed GeoTIFF will be saved.
        graph_xml_path (str): Path to the SNAP GPT XML graph file.
    """
    # Ensure the GPT executable path from config is valid
    gpt_path = config.SNAP_GPT_PATH
    if not os.path.exists(gpt_path) and gpt_path == "gpt":
        print(
            f"Warning: '{gpt_path}' not found as an absolute path. \
                Assuming it's in the system's PATH."
        )
    elif not os.path.exists(gpt_path):
        print(f"Error: SNAP GPT executable not found at '{gpt_path}'.")
        print(
            "Please update the SNAP_GPT_PATH in \
            'src/flood_detector/config.py'."
        )
        sys.exit(1)

    # Ensure the input and graph files exist
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        sys.exit(1)
    if not os.path.exists(graph_xml_path):
        print(f"Error: SNAP graph XML not found at '{graph_xml_path}'")
        sys.exit(1)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print("--- Starting SNAP GPT Preprocessing ---")
    print(f"  Input: {os.path.basename(input_path)}")
    print(f"  Output: {os.path.basename(output_path)}")
    print(f"  Graph: {os.path.basename(graph_xml_path)}")

    # Construct the command-line arguments for GPT
    # The -Pfile and -Ptarget parameters are used to pass file paths
    # into the SNAP graph. Your Read and Write operators in the graph
    # must be configured to use these variables (e.g., ${file}, ${target}).
    command = [
        gpt_path,
        graph_xml_path,
        f"-Pinput_file={input_path}",
        f"-Poutput_file={output_path}",
    ]

    print(f"Executing command: {' '.join(command)}")

    try:
        # Execute the command
        # We use subprocess.run to capture stdout and stderr
        process = subprocess.run(
            command,
            check=True,  # Raises CalledProcessError if non-zero exit code
            capture_output=True,  # Captures stdout and stderr
            text=True,  # Decodes stdout/stderr as text
        )
        print("--- SNAP GPT process completed successfully ---")
        print("GPT STDOUT:")
        print(process.stdout)

    except FileNotFoundError:
        print(f"Error: Command '{gpt_path}' not found.")
        print(
            "Please ensure SNAP is installed and the GPT \
                executable is in your system's PATH,"
        )
        print("or set the correct path in 'src/flood_detector/config.py'.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("--- SNAP GPT process failed ---")
        print(f"Return code: {e.returncode}")
        print("GPT STDOUT:")
        print(e.stdout)
        print("GPT STDERR:")
        print(e.stderr)
        print(
            "Please check the SNAP logs and ensure the \
                graph is configured correctly."
        )
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Preprocess Sentinel-1 data using ESA SNAP GPT."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the raw Sentinel-1 product (usually a .zip file).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the preprocessed output GeoTIFF file.",
    )
    parser.add_argument(
        "--graph_xml",
        type=str,
        default=config.SAR_PREPROCESSING_GRAPH_XML,
        help="Path to the SNAP GPT graph XML file. \
            Defaults to the one in config.",
    )

    args = parser.parse_args()

    # --- Run Preprocessing ---
    # This is a placeholder call. In a real scenario, this would trigger
    # a lengthy computation. For this example, we will just print the command.
    # To actually run this, you would uncomment the
    # line below and have SNAP installed.

    print("--- Preprocessing Script Invoked ---")
    print("This script is a wrapper for the ESA SNAP GPT.")
    print("It will now attempt to call the GPT tool.")
    print(
        "NOTE: This requires a working ESA SNAP installation \
            and a correctly configured graph XML."
    )

    run_snap_gpt(args.input_path, args.output_path, args.graph_xml)
