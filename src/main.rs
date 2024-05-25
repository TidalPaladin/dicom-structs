use clap::Parser;
use dicom::dictionary_std::tags;
use dicom::object::AccessError;
use dicom_structs::dicom::{is_dicom_file, open_dicom};
use dicom_structs::parquet::dicom_to_parquet;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rayon::prelude::*;
use rust_search::SearchBuilder;
use std::io::Error;
use std::path::PathBuf;
use std::time::Instant;

use thiserror::Error;

#[derive(Error, Debug)]
enum DicomConversionError {
    #[error("IO error: {0}")]
    Io(#[from] Error),
    #[error("DICOM access error: {0}")]
    Access(#[from] AccessError),
    #[error("DICOM access error: {0}")]
    Dicom(String),
    #[error("Other error: {0}")]
    #[allow(dead_code)]
    Other(String),
}

// Find the files to be processed with a progress bar
fn find_dicom_files(dir: &PathBuf) -> impl Iterator<Item = PathBuf> {
    // Set up spinner, iterating may files may take some time
    let spinner = ProgressBar::new_spinner();
    spinner.set_message("Searching for DICOM files");
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );

    // Yield from the search
    SearchBuilder::default()
        .location(dir)
        .build()
        .inspect(move |_| spinner.tick())
        .map(PathBuf::from)
        .filter(move |file| is_dicom_file(file, false) && file.is_file())
}

fn convert_dicom_file(
    file: &PathBuf,
    output_dir: &PathBuf,
    header_only: bool,
    hash_pixel_data: bool,
) -> Result<PathBuf, DicomConversionError> {
    // Open and read SOPInstanceUID and StudyInstanceUID (required fields)
    let dicom = open_dicom(file, header_only)?;
    let sop_instance_uid = dicom
        .element(tags::SOP_INSTANCE_UID)?
        .value()
        .to_str()
        .map_err(|_| {
            DicomConversionError::Dicom("Failed to convert SOPInstanceUID to str".to_string())
        })
        .and_then(|s| {
            if s.is_empty() {
                Err(DicomConversionError::Dicom(
                    "SOPInstanceUID is empty".to_string(),
                ))
            } else {
                Ok(s)
            }
        })?;

    let study_instance_uid = dicom
        .element(tags::STUDY_INSTANCE_UID)?
        .value()
        .to_str()
        .map_err(|_| {
            DicomConversionError::Dicom("Failed to convert StudyInstanceUID to str".to_string())
        })
        .and_then(|s| {
            if s.is_empty() {
                Err(DicomConversionError::Dicom(
                    "StudyInstanceUID is empty".to_string(),
                ))
            } else {
                Ok(s)
            }
        })?;

    // Use metadata to build a destination filepath
    let subpath = format!("{}/{}.parquet", study_instance_uid, sop_instance_uid);
    let output_file = output_dir.join(subpath);
    assert!(
        output_file.starts_with(output_dir),
        "Output {:?} is not a subdirectory of {:?}",
        output_file,
        output_dir
    );

    // Create parent directories if they don't exist
    std::fs::create_dir_all(output_file.parent().unwrap())?;

    // Run the conversion
    dicom_to_parquet(&dicom, &output_file, header_only, hash_pixel_data)?;
    Ok(output_file)
}

// Convert all DICOM files to parquet
fn convert_dicom_files<'a>(
    files: Vec<&'a PathBuf>,
    output_dir: &'a PathBuf,
    header_only: bool,
    hash_pixel_data: bool,
) -> Vec<PathBuf> {
    // Prepare args
    let header_only = header_only;
    let output_dir = output_dir.canonicalize().unwrap();

    // Create progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Converting DICOM files to parquet");

    // Parallel map convert_dicom_file to each file
    files
        .into_par_iter()
        .filter_map(move |file| {
            match convert_dicom_file(file, &output_dir, header_only, hash_pixel_data) {
                Ok(r) => Some(r),
                Err(e) => {
                    warn!("Failed to convert DICOM file {}", e);
                    None
                }
            }
        })
        .progress_with(pb)
        .collect()
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(help = "Directory of DICOM files to process")]
    source_dir: PathBuf,
    #[arg(help = "Directory to write outputs to")]
    output_dir: PathBuf,
    #[arg(
        long = "header-only",
        help = "Don't include pixel data in the output",
        action = clap::ArgAction::SetTrue
    )]
    header_only: bool,
    #[arg(
        long = "hash",
        help = "Include a hash of the pixel data in the output. Uses xxh3-64 with seed=0.",
        action = clap::ArgAction::SetTrue
    )]
    hash: bool,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    info!("Starting DICOM file processor");
    info!("Source directory: {:?}", args.source_dir);
    info!("Output directory: {:?}", args.output_dir);
    info!("Header only: {:?}", args.header_only);
    info!("Hash: {:?}", args.hash);

    if !args.output_dir.exists() {
        error!("Output directory does not exist: {:?}", args.output_dir);
        std::process::exit(1);
    }
    if !args.output_dir.is_dir() {
        error!("Output path is not a directory: {:?}", args.output_dir);
        std::process::exit(1);
    }
    if std::fs::metadata(&args.output_dir)
        .map(|m| m.permissions().readonly())
        .unwrap_or(true)
    {
        error!("Output directory is not writable: {:?}", args.output_dir);
        std::process::exit(1);
    }

    let start = Instant::now();
    let dicom_files: Vec<_> = find_dicom_files(&args.source_dir).collect();
    let dicom_files = dicom_files.iter().collect();
    let parquet_files =
        convert_dicom_files(dicom_files, &args.output_dir, args.header_only, args.hash);

    let end = Instant::now();
    println!(
        "Converted {:?} files in {:?}",
        parquet_files.len(),
        end.duration_since(start)
    );
}
