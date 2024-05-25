use crate::dicom::open_dicom;
use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use dicom::core::dictionary::DataDictionaryEntry;
use dicom::core::header::HasLength;
use dicom::core::value::PrimitiveValue;
use dicom::core::DataDictionary;
use dicom::core::DicomValue;
use dicom::dictionary_std::tags;
use dicom::dictionary_std::StandardDataDictionary;
use dicom::object::InMemDicomObject;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::io::Error;
use std::path::Path;
use std::sync::Arc;

// Match a DICOM value type to an Arrow type
fn match_primitive_value(tag_name: &str, v: &PrimitiveValue, nullable: bool) -> (Field, ArrayRef) {
    let data_type = match v {
        PrimitiveValue::F32(_) => DataType::Float32,
        PrimitiveValue::F64(_) => DataType::Float64,
        PrimitiveValue::I16(_) => DataType::Int16,
        PrimitiveValue::U16(_) => DataType::UInt16,
        PrimitiveValue::I32(_) => DataType::Int32,
        PrimitiveValue::U32(_) => DataType::UInt32,
        PrimitiveValue::I64(_) => DataType::Int64,
        PrimitiveValue::U64(_) => DataType::UInt64,
        PrimitiveValue::U8(_) => DataType::UInt8,
        PrimitiveValue::Strs(_) => DataType::Utf8,
        PrimitiveValue::Str(_) => DataType::Utf8,
        _ => DataType::Utf8,
    };
    let field = Field::new(tag_name, data_type.clone(), nullable);

    let array: ArrayRef = match v {
        PrimitiveValue::F32(f) => Arc::new(Float32Array::from(f.to_vec())),
        PrimitiveValue::F64(f) => Arc::new(Float64Array::from(f.to_vec())),
        PrimitiveValue::I16(f) => Arc::new(Int16Array::from(f.to_vec())),
        PrimitiveValue::U16(f) => Arc::new(UInt16Array::from(f.to_vec())),
        PrimitiveValue::I32(f) => Arc::new(Int32Array::from(f.to_vec())),
        PrimitiveValue::U32(f) => Arc::new(UInt32Array::from(f.to_vec())),
        PrimitiveValue::I64(f) => Arc::new(Int64Array::from(f.to_vec())),
        PrimitiveValue::U64(f) => Arc::new(UInt64Array::from(f.to_vec())),
        PrimitiveValue::U8(f) => Arc::new(UInt8Array::from(f.to_vec())),
        PrimitiveValue::Strs(f) => Arc::new(StringArray::from(f.to_vec())),
        PrimitiveValue::Str(f) => Arc::new(StringArray::from(vec![f.as_str()])),
        f => Arc::new(StringArray::from(vec![f.to_string()])),
    };

    // If the data has non-unit length, wrap it in a length-1 list
    let (array, field) = match array.len() {
        0 => panic!("Array should never be length 0"),
        1 => (array, field),
        _ => {
            let nested_field = Arc::new(Field::new("item", array.data_type().clone(), false));
            let offsets = Int32Array::from(vec![0, array.len() as i32]);
            let list_data = ArrayData::builder(DataType::List(nested_field))
                .len(1)
                .add_buffer(offsets.to_data().buffers()[0].clone())
                .add_child_data(array.to_data())
                .build()
                .unwrap();

            let array = ListArray::from(list_data);
            let field = Field::new(tag_name, array.data_type().clone(), nullable);
            (Arc::new(array) as ArrayRef, field)
        }
    };

    assert_eq!(array.len(), 1, "Arrow array should have length 1");
    (field, array)
}

/// Hashes pixel data if required and writes DICOM data to a Parquet file.
///
/// This function reads a DICOM file from the specified `dicom_path`, extracts its metadata and data,
/// and writes it to a Parquet file at the specified `parquet_path`. If `header_only` is set to true,
/// only the metadata (header) of the DICOM file is written to the Parquet file. Otherwise, both the
/// metadata and the data are written.
///
/// # Arguments
///
/// * `dicom_path` - Path to the input DICOM file.
/// * `parquet_path` - Path to the output Parquet file.
/// * `header_only` - Boolean flag indicating whether to write only the metadata (header) or both metadata and data.
///
/// # Returns
///
/// * `Result<(), Error>` - Ok if the operation is successful, otherwise an Error.
///
/// # Errors
///
/// This function will return an error if the DICOM file cannot be read, if the Parquet file cannot be written,
/// or if there is an issue with the DICOM or Parquet schema.
pub fn dicom_file_to_parquet(
    dicom_path: &Path,
    parquet_path: &Path,
    header_only: bool,
    hash_pixel_data: bool,
) -> Result<(), Error> {
    let obj = open_dicom(dicom_path, header_only)?;
    dicom_to_parquet(&obj, parquet_path, header_only, hash_pixel_data)
}

/// Hashes pixel data if required and writes DICOM data to a Parquet file.
///
/// This function processes the DICOM data provided, optionally hashes pixel data if specified,
/// and writes the resulting data to a Parquet file. It handles both cases where only metadata
/// is required or both metadata and pixel data are needed.
///
/// # Arguments
///
/// * `dicom` - A reference to the in-memory DICOM object.
/// * `parquet_path` - The file path where the Parquet file will be saved.
/// * `header_only` - If true, only DICOM headers are processed; otherwise, full data is processed.
/// * `hash_pixel_data` - If true, pixel data is hashed for additional processing.
///
/// # Returns
///
/// * `Result<(), Error>` - Result indicating the success or failure of the operation.
///
/// # Errors
///
/// This function will return an error if there is an issue with reading the DICOM data,
/// processing the pixel data, or writing to the Parquet file.
pub fn dicom_to_parquet(
    dicom: &InMemDicomObject,
    parquet_path: &Path,
    header_only: bool,
    hash_pixel_data: bool,
) -> Result<(), Error> {
    // Extract metadata and data
    let mut fields = vec![];
    let mut arrays: Vec<ArrayRef> = vec![];

    // Loop through tags and add to schema and arrays
    for element in dicom.into_iter() {
        let (header, value) = (&element.header(), &element.value());
        if value.is_empty() {
            continue;
        }

        let tag_name = StandardDataDictionary
            .by_tag(header.tag)
            .map_or_else(|| header.tag.to_string(), |t| t.alias().to_string());

        let nullable =
            header.tag != tags::SOP_INSTANCE_UID && header.tag != tags::STUDY_INSTANCE_UID;

        match value {
            DicomValue::Primitive(v) => {
                let (field, array) = match_primitive_value(&tag_name, v, nullable);
                assert!(array.len() == 1, "Array should have length 1");
                fields.push(field);
                arrays.push(array);
            }
            // Sequence types are added to the arrays as lists of strings
            DicomValue::Sequence(seq) => {
                let seq_values: Vec<String> = seq
                    .items()
                    .iter()
                    .map(|item| format!("{:?}", item))
                    .collect();
                if !seq_values.is_empty() {
                    fields.push(Field::new(&tag_name, DataType::Utf8, true));
                    arrays.push(Arc::new(StringArray::from(seq_values)) as ArrayRef);
                }
            }
            // Multi-frame pixel data
            DicomValue::PixelSequence(pixels) => {
                if !header_only {
                    // Read the offset table and fragments and flatten them into a stream
                    let offset_table = pixels.offset_table().iter().flat_map(|x| x.to_le_bytes());
                    let fragments = pixels.fragments().iter().flatten().copied();
                    let raw_data = offset_table.chain(fragments).collect::<Vec<u8>>();

                    let array = BinaryArray::from(vec![raw_data.as_slice()]);
                    if !array.is_empty() {
                        fields.push(Field::new(&tag_name, DataType::Binary, true));
                        arrays.push(Arc::new(array) as ArrayRef);
                    }
                }
            }
        }
    }

    // Hash the pixel data if requested
    if hash_pixel_data {
        // Check if the pixel data is present
        let has_pixel_data = dicom
            .element_opt(tags::PIXEL_DATA)
            .unwrap_or(None)
            .is_some_and(|v| !v.is_empty());

        // Add the hashed pixel data to the schema and arrays
        if has_pixel_data {
            let hashed_pixel_data = crate::dicom::hash_pixel_data(dicom)
                .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            fields.push(Field::new("PixelDataHash", DataType::UInt64, true));
            arrays.push(Arc::new(UInt64Array::from(vec![hashed_pixel_data])) as ArrayRef);
        }
    }

    // Create schema
    let schema = Arc::new(Schema::new(fields));

    // Set compression to SNAPPY
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    // Setup the writer
    let file = File::create(parquet_path)?;
    let mut arrow_writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Set up the batch to write
    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Run write op
    arrow_writer
        .write(&batch)
        .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    arrow_writer
        .close()
        .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow::array::{ListArray, PrimitiveArray, StringArray, UInt64Array};

    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use dicom::object::OpenFileOptions;
    use rstest::rstest;

    use super::dicom_file_to_parquet;

    use std::fs::File;

    #[test]
    fn dicom_to_parquet_sopuid() {
        // Get the expected SOPInstanceUID from the DICOM
        let dicom_file_path = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();
        let obj = OpenFileOptions::new()
            .open_file(dicom_file_path.clone())
            .unwrap();
        let expected = obj
            .element_by_name("SOPInstanceUID")
            .unwrap()
            .value()
            .to_str();

        // Convert to parquet
        let parquet_file_path = tempfile::NamedTempFile::new().unwrap();
        dicom_file_to_parquet(&dicom_file_path, parquet_file_path.path(), true, true).unwrap();

        // Read parquet
        let parquet_file = File::open(parquet_file_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).unwrap();
        let mut reader = builder.build().unwrap();

        // Check that the SOPInstanceUID is correct
        let batch = reader.next().unwrap().unwrap();
        let column = batch.column(batch.schema().index_of("SOPInstanceUID").unwrap());
        let sop_instance_uid = column
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(sop_instance_uid, expected.unwrap());
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn dicom_to_parquet_pixel_data(#[case] header_only: bool) {
        // Get the expected PixelData from the DICOM
        let dicom_file_path = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();
        let obj = OpenFileOptions::new()
            .open_file(dicom_file_path.clone())
            .unwrap();
        let expected: Vec<u8> = obj
            .element_by_name("PixelData")
            .unwrap()
            .value()
            .to_multi_int()
            .unwrap();

        // Convert to parquet
        let parquet_file_path = tempfile::NamedTempFile::new().unwrap();
        dicom_file_to_parquet(
            &dicom_file_path,
            parquet_file_path.path(),
            header_only,
            true,
        )
        .unwrap();

        // Read parquet
        let parquet_file = File::open(parquet_file_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).unwrap();
        let mut reader = builder.build().unwrap();

        // Check that the PixelData is correct
        let batch = reader.next().unwrap().unwrap();
        match header_only {
            true => {
                // PixelData should not be present in the schema
                let index = batch.schema().index_of("PixelData");
                println!("{:?}", batch.schema());
                assert!(index.is_err());
            }
            false => {
                // PixelData should be present in the schema
                let column = batch.column(batch.schema().index_of("PixelData").unwrap());
                let pixel_data = column
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .unwrap()
                    .value(0);
                assert_eq!(*pixel_data, PrimitiveArray::from(expected));

                // PixelData hash should be present in the schema
                let column = batch.column(batch.schema().index_of("PixelDataHash").unwrap());
                let expected_hash = 10240938377863354873_u64;
                let hash = column
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .value(0);
                assert_eq!(hash, expected_hash);
            }
        };
    }
}
