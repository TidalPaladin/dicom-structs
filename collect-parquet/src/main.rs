use arrow::array::RecordBatch;
use arrow::compute::cast;
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use dicom_structs_core::error::Error;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{info, warn};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::arrow::parquet_to_arrow_schema;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use rayon::prelude::*;
use rust_search::SearchBuilder;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const CHANNEL_SIZE: usize = 1024;

/// Find the files to be processed with a progress bar
fn find_parquet_files(dir: &PathBuf) -> impl Iterator<Item = PathBuf> {
    // Set up spinner, iterating may files may take some time
    let spinner = ProgressBar::new_spinner();
    spinner.set_message("Searching for Parquet files");
    spinner.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );

    // Yield from the search
    // NOTE: Directories with `tmp` are ignored
    SearchBuilder::default()
        .location(dir)
        .ext("parquet")
        .build()
        .inspect(move |_| spinner.tick())
        .map(PathBuf::from)
        .filter(move |file| file.is_file())
}

/// Read only the schema from a parquet file
fn read_parquet_schema(
    path: &PathBuf,
) -> Result<Schema, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file).unwrap();
    let metadata = reader.metadata();
    let schema_descriptor = metadata.file_metadata().schema_descr_ptr();
    let key_value_metadata = metadata.file_metadata().key_value_metadata();
    let schema = parquet_to_arrow_schema(&schema_descriptor, key_value_metadata).unwrap();
    Ok(schema)
}

/// Drop private tags from a schema.
/// Private tags are identified by starting with a '('
fn drop_private_tags(schema: &Schema) -> Schema {
    let fields = schema
        .fields()
        .into_iter()
        .filter(|f| !f.name().starts_with("("))
        .map(|f| Field::new(f.name(), f.data_type().clone(), f.is_nullable()))
        .collect::<Vec<_>>();
    Schema::new(fields)
}

/// Convert all fields in a schema to UTF8
fn convert_fields_to_utf8(schema: &Schema) -> Schema {
    let fields = schema
        .fields()
        .into_iter()
        .map(|f| Field::new(f.name(), arrow::datatypes::DataType::Utf8, f.is_nullable()))
        .collect::<Vec<_>>();
    Schema::new(fields)
}

/// Read a single Parquet file
fn read_parquet(
    path: &PathBuf,
) -> Result<RecordBatch, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let parquet_file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file)?;
    let mut reader = builder.build().unwrap();
    let mut batch = reader
        .next()
        .unwrap_or(Err(ArrowError::ParseError("No data found".to_string())))?;

    // Drop the PixelData column, it is large and we aren't putting it into the aggregate
    let _ = batch
        .schema()
        .index_of("PixelData")
        .and_then(|i| Ok(batch.remove_column(i)));

    Ok(batch)
}

/// Create a schema from a vector of parquet file paths
fn create_schema(paths: &Vec<PathBuf>) -> Result<Schema, Error> {
    if paths.is_empty() {
        return Err(Error::Whatever {
            message: "No parquet files found".to_string(),
            source: None,
        });
    }

    // Create progress bar
    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Determining schema from parquet files");

    // Read the schemas from the parquet files and aggregate them
    let acc = read_parquet_schema(&paths[0]).unwrap();
    let acc = drop_private_tags(&acc);
    let acc = convert_fields_to_utf8(&acc);
    let result = paths
        .par_iter()
        .progress_with(pb)
        .filter_map(|p| match read_parquet_schema(p) {
            Ok(r) => Some(r),
            Err(e) => {
                warn!("Failed to read schema from Parquet file {}", e);
                None
            }
        })
        .map(|s| drop_private_tags(&s))
        .map(|s| convert_fields_to_utf8(&s))
        .reduce(
            || acc.clone(),
            |acc, s| match Schema::try_merge(vec![acc.clone(), s]) {
                Ok(r) => r,
                Err(e) => {
                    warn!("Failed to merge schemas: {}", e);
                    acc
                }
            },
        );
    Ok(result)
}

/// Cast a record to a schema
fn cast_record_to_schema(record: &RecordBatch, schema: &Schema) -> Result<RecordBatch, Error> {
    // Cast every column in record to match the schema
    let casted_columns = schema
        .fields()
        .iter()
        .map(|f| {
            match record.schema().index_of(f.name()) {
                // If the column exists, cast it to the correct type
                Ok(i) => {
                    let input = record.column(i);
                    let target = f.data_type();
                    cast(input, target).map_err(|e| Error::Whatever {
                        message: format!("Failed to create output file: {}", e),
                        source: Some(Box::new(e)),
                    })
                }
                // Otherwise create a null array of the correct type
                _ => Ok(arrow::array::new_null_array(
                    f.data_type(),
                    record.num_rows() as usize,
                )),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert!(
        casted_columns.len() == schema.fields().len(),
        "Casted column count {} should match schema column count: {}",
        casted_columns.len(),
        schema.fields().len()
    );

    let batch = RecordBatch::try_new(schema.clone().into(), casted_columns).map_err(|e| {
        Error::Whatever {
            message: format!("Failed to create batch: {}", e),
            source: Some(Box::new(e)),
        }
    })?;
    Ok(batch)
}

/// Get the path to a shard file
#[inline]
fn shard_path(path: &PathBuf, index: usize) -> PathBuf {
    let shard_extension = format!("{:05}.parquet", index);
    path.with_extension("").with_extension(shard_extension)
}

/// Opens a new shard file
#[inline]
fn open_output_shard(path: &PathBuf, index: usize) -> Result<File, Error> {
    let shard_path = shard_path(path, index);
    File::create(shard_path).map_err(|e| Error::Whatever {
        message: format!("Failed to create output file: {}", e),
        source: Some(Box::new(e)),
    })
}

/// Collect multiple parquet files and write them to an output path
fn collect_parquet_files(
    paths: &Vec<PathBuf>,
    output_path: PathBuf,
    schema: &Schema,
    props: WriterProperties,
    shard_size_mb: usize,
) -> Result<Vec<PathBuf>, Error> {
    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Processing Parquet files");
    let (sender, receiver): (Sender<RecordBatch>, Receiver<RecordBatch>) = bounded(CHANNEL_SIZE);

    // Receiver thread
    let schema_clone = schema.clone();
    let receiver_thread = thread::spawn(move || {
        // Set up variables
        let shard_idx = 0;
        let mut shards = vec![shard_path(&output_path, shard_idx)];
        let shard_file = open_output_shard(&output_path, shard_idx)?;
        let mut writer = ArrowWriter::try_new(
            shard_file,
            Arc::new(schema_clone.clone()),
            Some(props.clone()),
        )
        .map_err(|e| Error::Whatever {
            message: format!("Failed to create struct array: {}", e),
            source: Some(Box::new(e)),
        })?;

        // Take items from the sender
        while let Ok(record) = receiver.recv() {
            // If we have exceeded the shard size, set up a new writer
            let need_new_shard = writer.in_progress_size() > shard_size_mb * (1024 * 1024)
                || writer.in_progress_rows() > props.max_row_group_size();
            if need_new_shard {
                // Close the current writer
                writer.close().map_err(|e| Error::Whatever {
                    message: format!("Failed to close writer: {}", e),
                    source: Some(Box::new(e)),
                })?;

                // Create a new writer
                let shard_idx = shards.len();
                shards.push(shard_path(&output_path, shard_idx));
                let shard_file = open_output_shard(&output_path, shard_idx)?;
                writer = ArrowWriter::try_new(
                    shard_file,
                    Arc::new(schema_clone.clone()),
                    Some(props.clone()),
                )
                .map_err(|e| Error::Whatever {
                    message: format!("Failed to create struct array: {}", e),
                    source: Some(Box::new(e)),
                })?;
            }

            // Write this item to the writer
            writer.write(&record).map_err(|e| Error::Whatever {
                message: format!("Failed to write batch: {}", e),
                source: Some(Box::new(e)),
            })?;
        }

        // Finalize the active writer
        writer.close().map_err(|e| Error::Whatever {
            message: format!("Failed to close writer: {}", e),
            source: Some(Box::new(e)),
        })?;
        Ok(shards)
    });

    // Parallel map to open each Parquet file, convert to shared schema, and send to receiver with a shard index
    paths
        .into_par_iter()
        .progress_with(pb)
        .filter_map(move |file| match read_parquet(&file) {
            Ok(r) => Some(r),
            Err(e) => {
                warn!("Failed to read Parquet file {}", e);
                None
            }
        })
        .filter_map(|batch| match cast_record_to_schema(&batch, schema) {
            Ok(r) => Some(r),
            Err(e) => {
                warn!("Failed to convert Parquet file {} to shared schema", e);
                None
            }
        })
        .for_each_with(sender, |s, record| {
            s.send(record).expect("Failed to send item");
        });

    // Wait for the receiver thread to finish
    receiver_thread.join().unwrap()
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = "0.1.0", about = "Aggregate Parquet files into a single file", long_about = None)]
struct Args {
    #[arg(help = "Directory of Parquet files to process")]
    source_dir: PathBuf,

    #[arg(help = "Path to write output to")]
    output_path: PathBuf,

    #[arg(
        short = 's',
        long = "shard-size",
        help = "Size of each Parquet output shard in MB",
        default_value = "128"
    )]
    shard_size: usize,

    #[arg(
        short = 'c',
        long = "compression-level",
        help = "ZSTD compression level",
        default_value = "6",
        value_parser = clap::value_parser!(i32).range(1..=22)
    )]
    compression_level: i32,
}

fn run(args: Args) -> Result<Vec<PathBuf>, Error> {
    info!("Starting Parquet file aggregator");
    info!("Source directory: {:?}", args.source_dir);
    info!("Output path: {:?}", args.output_path);
    info!("Shard size: {:?}MB", args.shard_size);
    info!("ZSTD compression level: {:?}", args.compression_level);
    let start = Instant::now();

    // Find the parquet files
    let parquet_files: Vec<_> = find_parquet_files(&args.source_dir).collect();
    println!("Found {:?} parquet files", parquet_files.len());

    // Determine the schema from the parquet files
    info!("Determining schema from parquet files");
    let schema = create_schema(&parquet_files)?;
    for (i, f) in schema.fields().iter().enumerate() {
        info!(
            "Field {}: {:?} ({:?}, {:?})",
            i,
            f.name(),
            f.data_type(),
            f.is_nullable()
        );
    }

    // Set compression to Zstd
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(args.compression_level).unwrap(),
        ))
        .build();

    // Run the collection
    let shard_paths = collect_parquet_files(
        &parquet_files,
        args.output_path,
        &schema,
        props,
        args.shard_size,
    )?;

    let end = Instant::now();
    info!(
        "Aggregated {:?} files in {:?}",
        parquet_files.len(),
        end.duration_since(start)
    );
    Ok(shard_paths)
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    run(args).unwrap();
}

#[cfg(test)]
mod tests {
    use super::{run, Args};
    use crate::create_schema;
    use dicom_structs_core::parquet::dicom_file_to_parquet;
    use std::path::Path;
    use std::path::PathBuf;

    fn setup_test_files(parquet_dir: &Path) -> (PathBuf, PathBuf) {
        let dicom_path_1 = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();
        let dicom_path_2 = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();

        // Create a temp directory and copy the test files to it
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dicom_path_1 = temp_dir.path().join("SC_rgb.dcm");
        let temp_dicom_path_2 = temp_dir.path().join("CT_small.dcm");
        std::fs::copy(&dicom_path_1, &temp_dicom_path_1).unwrap();
        std::fs::copy(&dicom_path_2, &temp_dicom_path_2).unwrap();

        // Create a temp directory and convert the DICOMs to parquet
        let parquet_path_1 = parquet_dir.join("SC_rgb.parquet");
        let parquet_path_2 = parquet_dir.join("CT_small.parquet");
        dicom_file_to_parquet(
            &temp_dicom_path_1,
            &parquet_path_1.to_path_buf(),
            true,
            false,
            None,
            true,
        )
        .unwrap();
        assert!(parquet_path_1.is_file());
        dicom_file_to_parquet(
            &temp_dicom_path_2,
            &parquet_path_2.to_path_buf(),
            true,
            false,
            None,
            true,
        )
        .unwrap();
        assert!(parquet_path_2.is_file());
        (parquet_path_1, parquet_path_2)
    }

    #[test]
    fn test_create_schema() {
        let parquet_dir = tempfile::tempdir().unwrap();
        let (parquet_path_1, parquet_path_2) = setup_test_files(&parquet_dir.path());
        let paths = vec![parquet_path_1, parquet_path_2];
        let schema = create_schema(&paths).unwrap();
        let num_fields = schema.fields().len();
        let expected = 86;
        assert!(
            num_fields == expected,
            "Expected {} fields, got {}",
            expected,
            num_fields
        );
    }

    #[test]
    fn test_main() {
        let parquet_dir = tempfile::tempdir().unwrap();
        setup_test_files(&parquet_dir.path());
        let source_dir = parquet_dir.path();

        let output_path = source_dir.join("output.parquet");

        // Run the main function
        let args = Args {
            source_dir: source_dir.to_path_buf(),
            output_path: output_path.to_path_buf(),
            shard_size: 128,
            compression_level: 3,
        };
        let output_files = run(args).unwrap();

        // Check that an output Parquet file was created
        assert!(output_files.len() == 1);
        for file in output_files {
            assert!(
                file.is_file(),
                "Parquet file {} not created.",
                file.display()
            );
        }
    }
}
