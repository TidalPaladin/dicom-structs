use clap::Parser;
use dicom::dictionary_std::tags;
use dicom_structs_core::dicom::{is_dicom_file, open_dicom};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rayon::prelude::*;
use rust_search::SearchBuilder;
use std::io::Error;
use std::os::unix::fs::symlink;
use std::path::PathBuf;
use std::time::Instant;

/// Find the files to be processed with a progress bar
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

fn symlink_dicom_file(
    file: &PathBuf,
    output_dir: &PathBuf,
    resolve: bool,
) -> Result<PathBuf, Error> {
    // Open and read
    let dicom = open_dicom(file, true)?;

    // Read SOPInstanceUID and StudyInstanceUID (required fields)
    let sop_instance_uid = dicom
        .element(tags::SOP_INSTANCE_UID)
        .map_err(|_| {
            Error::new(
                std::io::ErrorKind::Other,
                "Failed to read SOPInstanceUID".to_string(),
            )
        })?
        .value()
        .to_str()
        .map_err(|_| {
            Error::new(
                std::io::ErrorKind::Other,
                "Failed to convert SOPInstanceUID to str".to_string(),
            )
        })
        .and_then(|s| {
            if s.is_empty() {
                Err(Error::new(
                    std::io::ErrorKind::Other,
                    "SOPInstanceUID is empty".to_string(),
                ))
            } else {
                Ok(s)
            }
        })?;

    let study_instance_uid = dicom
        .element(tags::STUDY_INSTANCE_UID)
        .map_err(|_| {
            Error::new(
                std::io::ErrorKind::Other,
                "Failed to read StudyInstanceUID".to_string(),
            )
        })?
        .value()
        .to_str()
        .map_err(|_| {
            Error::new(
                std::io::ErrorKind::Other,
                "Failed to convert StudyInstanceUID to str".to_string(),
            )
        })
        .and_then(|s| {
            if s.is_empty() {
                Err(Error::new(
                    std::io::ErrorKind::Other,
                    "StudyInstanceUID is empty".to_string(),
                ))
            } else {
                Ok(s)
            }
        })?;

    // Use metadata to build a destination filepath
    let subpath = format!("{}/{}.dcm", study_instance_uid, sop_instance_uid);
    let output_file = output_dir.join(subpath);
    assert!(
        output_file.starts_with(output_dir),
        "Output {:?} is not a subdirectory of {:?}",
        output_file,
        output_dir
    );

    // Create parent directories if they don't exist
    std::fs::create_dir_all(output_file.parent().unwrap())?;

    // If the source file is a symlink, resolve it first
    let file = if resolve {
        file.canonicalize()?
    } else {
        file.clone()
    };

    symlink(file, output_file.clone())?;
    assert!(
        output_file.is_symlink(),
        "Output file {:?} is not a symlink",
        output_file
    );
    Ok(output_file)
}

/// Create symlinks for all DICOM files
fn symlink_dicom_files<'a>(
    files: Vec<&'a PathBuf>,
    output_dir: &'a PathBuf,
    resolve: bool,
) -> Vec<PathBuf> {
    // Prepare args
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
    pb.set_message("Symlinking DICOM files");

    // Parallel map symlink_dicom_file to each file
    files
        .into_par_iter()
        .filter_map(
            move |file| match symlink_dicom_file(file, &output_dir, resolve) {
                Ok(r) => Some(r),
                Err(e) => {
                    warn!("Failed to symlink DICOM file {}", e);
                    None
                }
            },
        )
        .progress_with(pb)
        .collect()
}

#[derive(Parser, Debug)]
#[command(
    author = "Scott Chase Waggener",
    version = "0.1.0",
    about = "Symlink DICOM files into a consistent file structure"
)]
struct Args {
    #[arg(help = "Directory of DICOM files to process")]
    source_dir: PathBuf,

    #[arg(help = "Directory to write outputs to")]
    output_dir: PathBuf,

    #[arg(help = "Resolve symlinks in the source directory")]
    resolve: bool,
}

fn run(args: Args) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    info!("Starting DICOM file symlinker");
    info!("Source directory: {:?}", args.source_dir);
    info!("Output directory: {:?}", args.output_dir);
    info!("Resolve symlinks: {:?}", args.resolve);

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
    let output_symlinks = symlink_dicom_files(dicom_files, &args.output_dir, args.resolve);

    let end = Instant::now();
    info!(
        "Linked {:?} files in {:?}",
        output_symlinks.len(),
        end.duration_since(start)
    );
    Ok(output_symlinks)
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    run(args).unwrap();
}

#[cfg(test)]
mod tests {
    use super::{run, symlink_dicom_file, Args};
    use std::fs;
    use std::os::unix::fs::symlink;

    #[test]
    fn test_main() {
        // Get the expected SOPInstanceUID from the DICOM
        let dicom_file_path = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();

        // Create a temp directory and copy the test file to it
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dicom_path = temp_dir.path().join("SC_rgb.dcm");
        std::fs::copy(&dicom_file_path, &temp_dicom_path).unwrap();

        // Create a temp directory to hold the output
        let output_dir = tempfile::tempdir().unwrap();

        // Run the main function
        let args = Args {
            source_dir: temp_dir.path().to_path_buf(),
            output_dir: output_dir.path().to_path_buf(),
            resolve: false,
        };
        run(args).unwrap(); // Assuming `main` is adapted to take `Args` struct

        // Check that an output Parquet file was created
        let output_files = std::fs::read_dir(output_dir.path()).unwrap();
        assert!(
            output_files.count() > 0,
            "No files were created in the output directory."
        );
    }

    #[test]
    fn test_symlink_inputs_resolved() {
        // Create a temporary directory to hold the symlink
        let temp_dir = tempfile::tempdir().unwrap();
        let symlink_path = temp_dir.path().join("symlink.dcm");

        // Create a symlink to the DICOM file
        let dicom_file_path = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();
        symlink(&dicom_file_path, &symlink_path).unwrap();

        // Create a temporary directory to hold the output
        let output_dir = tempfile::tempdir().unwrap();

        // Run the main function
        let output_symlink =
            symlink_dicom_file(&symlink_path, &output_dir.path().to_path_buf(), true).unwrap();

        // Check that the symlink in the output directory points to the original DICOM file
        let resolved_path = fs::read_link(output_symlink).unwrap();
        assert_eq!(
            resolved_path, dicom_file_path,
            "Symlink does not resolve to the original DICOM file."
        );
    }
}
