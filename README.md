# DICOM Structs

Library and CLI interface for manipulating DICOM files, primarly focusing on DICOM to Parquet conversion and
the preparation of DICOM files for ingestion into a data warehouse. Note, [Data Wranger](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.datawrangler) is a great tool for inspecting Parquet files.

## Usage

### Convert files Parquet

Use `dicom-to-parquet` to convert DICOM files to parquet. It accepts a directory of DICOM files to convert and an output directory to write the parquet files to. One Parquet file will be created for each DICOM file.

```
Convert DICOM files to parquet

Usage: dicom-to-parquet [OPTIONS] <SOURCE_DIR> <OUTPUT_DIR>

Arguments:
  <SOURCE_DIR>  Directory of DICOM files to process
  <OUTPUT_DIR>  Directory to write outputs to

Options:
      --header-only  Don't include pixel data in the output
      --hash         Include a hash of the pixel data in the output. Uses xxh3-64 with seed=0.
  -t, --tag <TAGS>   Override a DICOM tag with a constant value
  -s, --snake-case   Convert DICOM tag names to snake case
  -h, --help         Print help
  -V, --version      Print version
```

### Collect and Merge Parquet files

Use `collect-parquet` to collect individual DICOM parquet files into a single parquet file. It accepts a directory of parquet files to aggregate and an output path to write the aggregated parquet file to.

```
Aggregate Parquet files into a single file

Usage: collect-parquet <SOURCE_DIR> <OUTPUT_PATH>

Arguments:
  <SOURCE_DIR>   Directory of Parquet files to process
  <OUTPUT_PATH>  Path to write output to

Options:
  -h, --help     Print help
  -V, --version  Print version
```