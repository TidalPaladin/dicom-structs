use arrow::array::RecordBatch;
use arrow::compute::{cast, concat_batches};
use arrow::datatypes::Schema;
use arrow::error::ArrowError;
use clap::Parser;
use dicom_structs_core::error::Error;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use log::{info, warn};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use rust_search::SearchBuilder;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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

fn create_schema(batches: &Vec<RecordBatch>) -> Result<Schema, Error> {
    // Extract schemas from each batch
    let schemas: Vec<_> = batches
        .iter()
        .map(|b| Schema::new(b.schema().fields().clone()))
        .collect();

    //all fields that are nested in at least one record
    let nested_fields = schemas
        .iter()
        .flat_map(|s| s.fields().iter())
        .filter(|f| f.data_type().is_nested())
        .map(|f| (f.name(), f))
        .collect::<HashMap<_, _>>();

    // Update the schemas, removing any fields that are nested in at least one record
    let schemas: Vec<_> = schemas
        .iter()
        .map(|s| {
            let keep_fields: Vec<_> = s
                .fields()
                .iter()
                .filter(|f| !nested_fields.contains_key(f.name()))
                .map(|f| f.clone())
                .collect();
            Schema::new(keep_fields)
        })
        .collect();

    // Merge the non-nested fields
    let merged_schema = Schema::try_merge(schemas).map_err(|e| Error::Whatever {
        message: format!("Failed to merge schemas: {}", e),
        source: Some(Box::new(e)),
    })?;

    // Merge with the nested fields
    let nested_schema = Schema::new(
        nested_fields
            .into_values()
            .map(|f| f.clone())
            .collect::<Vec<_>>(),
    );
    println!("Nested schema: {:?}", nested_schema);
    Ok(
        Schema::try_merge(vec![merged_schema, nested_schema]).map_err(|e| Error::Whatever {
            message: format!("Failed to merge schemas: {}", e),
            source: Some(Box::new(e)),
        })?,
    )
}

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

/// Collect multiple parquet files into a single RecordBatch
fn collect_parquet_files(paths: Vec<PathBuf>) -> Result<RecordBatch, Error> {
    // Create progress bar
    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Aggregating Parquet files");

    // Parallel map to open each Parquet file
    let batches = paths
        .into_par_iter()
        .filter_map(move |file| match read_parquet(&file) {
            Ok(r) => Some(r),
            Err(e) => {
                warn!("Failed to read Parquet file {}", e);
                None
            }
        })
        .progress_with(pb)
        .collect::<Vec<RecordBatch>>();

    // Merge schemas from each record
    info!("Creating schema");
    let schema = create_schema(&batches)?;
    info!("Aggregated schema with {} fields", schema.fields().len());
    schema.fields().iter().for_each(|f| {
        info!(
            "Field: {:?} ({:?}, {:?})",
            f.name(),
            f.data_type(),
            f.is_nullable()
        )
    });

    // Cast all records to match the aggregate schema
    let pb = ProgressBar::new(batches.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta} @ {per_sec})",
            )
            .unwrap(),
    );
    pb.set_message("Converting records to shared schema");
    let batches = batches
        .par_iter()
        .map(|batch| cast_record_to_schema(&batch, &schema))
        .progress_with(pb)
        .collect::<Result<Vec<_>, _>>()?;

    info!(
        "Concatenating {} batches, this may take some time",
        batches.len()
    );
    let batch = concat_batches(&Arc::new(schema), batches.iter()).map_err(|e| Error::Whatever {
        message: format!("Failed to concatenate batches: {}", e),
        source: Some(Box::new(e)),
    })?;
    Ok(batch)
}

#[derive(Parser, Debug)]
#[command(author = "Scott Chase Waggener", version = "0.1.0", about = "Aggregate Parquet files into a single file", long_about = None)]
struct Args {
    #[arg(help = "Directory of Parquet files to process")]
    source_dir: PathBuf,

    #[arg(help = "Path to write output to")]
    output_path: PathBuf,
}

fn run(args: Args) -> Result<RecordBatch, Error> {
    info!("Starting Parquet file aggregator");
    info!("Source directory: {:?}", args.source_dir);
    info!("Output path: {:?}", args.output_path);
    let start = Instant::now();

    // Find the parquet files
    let parquet_files: Vec<_> = find_parquet_files(&args.source_dir).collect();
    println!("Found {:?} parquet files", parquet_files.len());

    // Collect the parquet files into a single batch
    let batch = collect_parquet_files(parquet_files)?;
    let schema = batch.schema();

    // Set compression to SNAPPY
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    // Setup the writer
    let file = File::create(args.output_path).map_err(|e| Error::Whatever {
        message: format!("Failed to create output file: {}", e),
        source: Some(Box::new(e)),
    })?;
    let mut arrow_writer =
        ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(|e| Error::Whatever {
            message: format!("Failed to create struct array: {}", e),
            source: Some(Box::new(e)),
        })?;

    // Run write op
    arrow_writer.write(&batch).map_err(|e| Error::Whatever {
        message: format!("Failed to write batch: {}", e),
        source: Some(Box::new(e)),
    })?;
    arrow_writer.close().map_err(|e| Error::Whatever {
        message: format!("Failed to close writer: {}", e),
        source: Some(Box::new(e)),
    })?;

    let end = Instant::now();
    info!(
        "Aggregated {:?} files in {:?}",
        batch.num_rows(),
        end.duration_since(start)
    );
    Ok(batch)
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    run(args).unwrap();
}

#[cfg(test)]
mod tests {
    use super::{run, Args};
    use dicom_structs_core::parquet::dicom_file_to_parquet;

    #[test]
    fn test_main() {
        let dicom_path_1 = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();
        let dicom_path_2 = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();

        // Create a temp directory and copy the test files to it
        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dicom_path_1 = temp_dir.path().join("SC_rgb.dcm");
        let temp_dicom_path_2 = temp_dir.path().join("CT_small.dcm");
        std::fs::copy(&dicom_path_1, &temp_dicom_path_1).unwrap();
        std::fs::copy(&dicom_path_2, &temp_dicom_path_2).unwrap();

        // Create a temp directory and convert the DICOMs to parquet
        let parquet_dir = tempfile::tempdir().unwrap();
        let parquet_path_1 = parquet_dir.path().join("SC_rgb.parquet");
        let parquet_path_2 = parquet_dir.path().join("CT_small.parquet");
        dicom_file_to_parquet(
            &temp_dicom_path_1,
            &parquet_path_1.to_path_buf(),
            true,
            false,
            None,
            true,
        )
        .unwrap();
        dicom_file_to_parquet(
            &temp_dicom_path_2,
            &parquet_path_2.to_path_buf(),
            true,
            false,
            None,
            true,
        )
        .unwrap();

        let output_path = parquet_dir.path().join("output.parquet");

        // Run the main function
        let args = Args {
            source_dir: temp_dir.path().to_path_buf(),
            output_path: output_path.to_path_buf(),
        };
        run(args).unwrap(); // Assuming `main` is adapted to take `Args` struct

        // Check that an output Parquet file was created
        assert!(output_path.is_file(), "Parquet file not created.");
    }
}
