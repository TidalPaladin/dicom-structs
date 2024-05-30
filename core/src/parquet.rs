use crate::dicom::open_dicom;
use crate::error::Error;
use arrow::array::*;
use arrow::compute::kernels::cast_utils::Parser;
use arrow::datatypes::{DataType, Date64Type, Field, Schema};
use arrow::record_batch::RecordBatch;
use dicom::core::dictionary::DataDictionaryEntry;
use dicom::core::header::HasLength;
use dicom::core::value::{DataSetSequence, PixelFragmentSequence, PrimitiveValue};
use dicom::core::DataElement;
use dicom::core::DicomValue;
use dicom::core::{DataDictionary, VR};
use dicom::dictionary_std::tags;
use dicom::dictionary_std::StandardDataDictionary;
use dicom::object::mem::InMemElement;
use dicom::object::FileDicomObject;
use dicom::object::InMemDicomObject;
use dicom_json::DicomJson;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub type DicomElement = DataElement<InMemDicomObject>;

pub trait ArrayWrap {
    /// Wrap the array in a list array of length 1
    fn wrap_as_list(&self) -> Self;
}

impl ArrayWrap for ArrayRef {
    fn wrap_as_list(&self) -> Self {
        let nested_field = Arc::new(Field::new("item", self.data_type().clone(), true));
        let offsets = Int32Array::from(vec![0, self.len() as i32]);
        let list_data = ArrayData::builder(DataType::List(nested_field))
            .len(1)
            .add_buffer(offsets.to_data().buffers()[0].clone())
            .add_child_data(self.to_data())
            .build()
            .unwrap();
        Arc::new(ListArray::from(list_data))
    }
}

impl ArrayWrap for Field {
    fn wrap_as_list(&self) -> Self {
        let nested_field = Arc::new(Field::new("item", self.data_type().clone(), true));
        Field::new(
            self.name(),
            DataType::List(nested_field),
            self.is_nullable(),
        )
    }
}

trait FromDicom: Sized {
    /// Create from a DICOM PrimitiveValue
    ///
    /// # Arguments
    ///
    /// * `value` - A reference to a PrimitiveValue object
    ///
    /// # Returns
    ///
    /// * Converted value, or None if the value is empty
    fn from_dicom_primitive(value: &PrimitiveValue) -> Option<Self>;

    /// Create from a DICOM sequence
    ///
    /// # Arguments
    ///
    /// * `seq` - A reference to a DataSetSequence object
    ///
    /// # Returns
    ///
    /// * Converted value, or None if the sequence is empty
    fn from_dicom_sequence(seq: &DataSetSequence<InMemDicomObject>) -> Option<Self>;

    /// Create from a DICOM pixel sequence
    ///
    /// # Arguments
    ///
    /// * `seq` - A reference to a PixelFragmentSequence object
    ///
    /// # Returns
    ///
    /// * Converted value, or None if the sequence is empty
    fn from_dicom_pixel_sequence(seq: &PixelFragmentSequence<Vec<u8>>) -> Option<Self>;

    /// Create from a DICOM element
    ///
    /// # Arguments
    ///
    /// * `elem` - A reference to a DicomElement object
    ///
    /// # Returns
    ///
    /// * Converted value, or None if the element is of sequence type and the sequence is empty
    fn from_dicom_element(elem: &DicomElement) -> Option<Self>;
}

impl FromDicom for DataType {
    fn from_dicom_primitive(value: &PrimitiveValue) -> Option<DataType> {
        // For multiplicity > 1, we will convert the contents to string
        if value.multiplicity() > 1 {
            return Some(DataType::Utf8);
        } else if value.is_empty() {
            return None;
        }
        // https://dicom.nema.org/dicom/2013/output/chtml/part05/sect_6.2.html
        Some(match value {
            PrimitiveValue::Date(_) => DataType::Date64, // 128 bits per DICOM standard, Arrow max is 64
            PrimitiveValue::U8(_) => DataType::UInt8,
            PrimitiveValue::U16(_) => DataType::UInt16,
            PrimitiveValue::I16(_) => DataType::Int16,
            PrimitiveValue::U32(_) => DataType::UInt32,
            PrimitiveValue::I32(_) => DataType::Int32,
            PrimitiveValue::F32(_) => DataType::Float32,
            PrimitiveValue::U64(_) => DataType::UInt64,
            PrimitiveValue::I64(_) => DataType::Int64,
            PrimitiveValue::F64(_) => DataType::Float64,
            PrimitiveValue::Strs(_) => DataType::Utf8,
            PrimitiveValue::Str(_) => DataType::Utf8,
            _ => DataType::Utf8,
        })
    }

    fn from_dicom_sequence(seq: &DataSetSequence<InMemDicomObject>) -> Option<Self> {
        // Sequences can be nested so we will flatten them to string
        match seq.is_empty() {
            true => None,
            false => Some(DataType::Utf8),
        }
    }

    fn from_dicom_pixel_sequence(pixels: &PixelFragmentSequence<Vec<u8>>) -> Option<Self> {
        match pixels.is_empty() {
            true => None,
            false => Some(DataType::Binary),
        }
    }

    fn from_dicom_element(elem: &DicomElement) -> Option<Self> {
        // Sometimes PixelData is not a PixelSequence, so catch it manually and force to binary
        if elem.header().tag == tags::PIXEL_DATA {
            return Some(DataType::Binary).filter(|_| !elem.value().is_empty());
        }
        match elem.value() {
            DicomValue::Primitive(v) => Self::from_dicom_primitive(v),
            DicomValue::Sequence(v) => Self::from_dicom_sequence(v),
            DicomValue::PixelSequence(v) => Self::from_dicom_pixel_sequence(v),
        }
    }
}

impl FromDicom for ArrayRef {
    fn from_dicom_primitive(value: &PrimitiveValue) -> Option<ArrayRef> {
        // For multiplicity > 1, we will convert the contents to string
        if value.multiplicity() > 1 {
            return Some(Arc::new(StringArray::from(vec![value.to_string()])));
        } else if value.is_empty() {
            return None;
        }
        Some(match value {
            PrimitiveValue::Date(f) => Arc::new(Date64Array::from(
                f.iter()
                    .map(|x| {
                        Date64Type::parse(&format!(
                            "{:04}-{:02}-{:02}",
                            x.year(),
                            x.month().unwrap_or(&0),
                            x.day().unwrap_or(&0)
                        ))
                    })
                    .collect::<Vec<_>>(),
            )),
            PrimitiveValue::U8(f) => Arc::new(UInt8Array::from(f.to_vec())),
            PrimitiveValue::U16(f) => Arc::new(UInt16Array::from(f.to_vec())),
            PrimitiveValue::I16(f) => Arc::new(Int16Array::from(f.to_vec())),
            PrimitiveValue::U32(f) => Arc::new(UInt32Array::from(f.to_vec())),
            PrimitiveValue::I32(f) => Arc::new(Int32Array::from(f.to_vec())),
            PrimitiveValue::F32(f) => Arc::new(Float32Array::from(f.to_vec())),
            PrimitiveValue::U64(f) => Arc::new(UInt64Array::from(f.to_vec())),
            PrimitiveValue::I64(f) => Arc::new(Int64Array::from(f.to_vec())),
            PrimitiveValue::F64(f) => Arc::new(Float64Array::from(f.to_vec())),
            PrimitiveValue::Strs(f) => Arc::new(StringArray::from(f.to_vec())),
            PrimitiveValue::Str(f) => Arc::new(StringArray::from(vec![f.as_str()])),
            f => Arc::new(StringArray::from(vec![f.to_string()])),
        })
    }

    fn from_dicom_sequence(seq: &DataSetSequence<InMemDicomObject>) -> Option<ArrayRef> {
        if seq.is_empty() {
            return None;
        }

        // Convert each sequence element to a JSON string
        let sub_jsons = seq
            .items()
            .into_iter()
            .map(DicomJson::from)
            .filter_map(|x| serde_json::to_value(&x).ok())
            .map(|x| x.to_string())
            .collect::<Vec<_>>();

        match serde_json::to_value(sub_jsons) {
            Ok(v) => Some(Arc::new(StringArray::from(vec![v.to_string()]))),
            Err(_) => None,
        }
    }

    fn from_dicom_pixel_sequence(pixels: &PixelFragmentSequence<Vec<u8>>) -> Option<Self> {
        match pixels.is_empty() {
            true => None,
            false => {
                // Read the offset table and fragments and flatten them into a stream
                let offset_table = pixels.offset_table().iter().flat_map(|x| x.to_le_bytes());
                let fragments = pixels.fragments().iter().flatten().copied();
                let raw_data = offset_table.chain(fragments).collect::<Vec<u8>>();
                Some(Arc::new(BinaryArray::from(vec![raw_data.as_slice()])))
            }
        }
    }

    fn from_dicom_element(elem: &DicomElement) -> Option<ArrayRef> {
        match (elem.header().tag, elem.value()) {
            // Sometimes PixelData is not a PixelSequence, so catch it manually and force to binary
            (tags::PIXEL_DATA, DicomValue::Primitive(p)) => match p.is_empty() {
                true => None,
                false => Some(Arc::new(BinaryArray::from(vec![p.to_bytes().as_ref()]))),
            },
            (_, DicomValue::PixelSequence(v)) => Self::from_dicom_pixel_sequence(v),
            (_, DicomValue::Sequence(v)) => Self::from_dicom_sequence(v),
            (_, DicomValue::Primitive(v)) => Self::from_dicom_primitive(v),
        }
    }
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
/// * `hash_pixel_data` - Boolean flag indicating whether to hash the pixel data.
/// * `overrides` - Optional hash map of tag names to values to override the original DICOM data.
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
    overrides: Option<&HashMap<String, String>>,
) -> Result<(), Error> {
    let mut obj =
        open_dicom(dicom_path, header_only && !hash_pixel_data).map_err(|e| Error::Whatever {
            message: format!("Failed to open DICOM file: {}", e),
            source: Some(Box::new(e)),
        })?;

    // Add each tag string and value string to the DICOM
    if overrides.is_some() {
        for (tag, value) in overrides.unwrap() {
            // Map string tag to a tag enum
            let tag = StandardDataDictionary::default()
                .by_name(tag)
                .ok_or(Error::TagNotFound {
                    tag_name: tag.to_string(),
                })?
                .tag();

            // Create DICOM element and inject value
            let element = DataElement::new(tag, VR::LO, value.to_string());
            obj.put(element);
        }
    }

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
    dicom: &FileDicomObject<InMemDicomObject>,
    parquet_path: &Path,
    header_only: bool,
    hash_pixel_data: bool,
) -> Result<(), Error> {
    // Allocate containers to hold Parquet fields and arrays
    let mut fields = vec![];
    let mut arrays: Vec<ArrayRef> = vec![];

    // Extract metadata header elements and make them compatible with the rest of the data
    let meta_elems = dicom
        .meta()
        .to_element_iter()
        .map(|e| e.into_parts())
        .filter_map(|(header, value)| match value.into_primitive() {
            Some(v) => Some(InMemElement::new(header.tag, header.vr(), v)),
            None => None,
        })
        .collect::<Vec<_>>();

    // Loop through header and body tags and add to schema / arrays
    let elements = dicom.into_iter().chain(meta_elems.iter());
    for element in elements {
        let (header, value) = (&element.header(), element.value());
        if value.is_empty() || (header_only && header.tag == tags::PIXEL_DATA) {
            continue;
        }

        let tag_name = StandardDataDictionary
            .by_tag(header.tag)
            .map_or_else(|| header.tag.to_string(), |t| t.alias().to_string());

        let nullable =
            header.tag != tags::SOP_INSTANCE_UID && header.tag != tags::STUDY_INSTANCE_UID;

        let field = DataType::from_dicom_element(element)
            .and_then(|dtype| Some(Field::new(tag_name, dtype, nullable)));
        let array = ArrayRef::from_dicom_element(element);

        match (field, array) {
            (Some(f), Some(a)) => {
                fields.push(f);
                arrays.push(a);
            }
            _ => {}
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
            let hashed_pixel_data =
                crate::dicom::hash_pixel_data(dicom).map_err(|_| Error::PixelHashError)?;
            fields.push(Field::new("PixelDataHash", DataType::UInt64, true));
            arrays.push(Arc::new(UInt64Array::from(vec![hashed_pixel_data])) as ArrayRef);
        }
    }

    fields.iter().zip(arrays.iter()).for_each(|(field, array)| {
        assert!(
            array.len() == 1,
            "Array {} should have length 1, found {}",
            field.name(),
            array.len()
        );
    });

    // Create schema
    let schema = Arc::new(Schema::new(fields));

    // Set compression to SNAPPY
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    // Setup the writer
    let file = File::create(parquet_path).map_err(|e| Error::Whatever {
        message: format!("Error while creating file: {}", e),
        source: Some(Box::new(e)),
    })?;
    let mut arrow_writer =
        ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(|e| Error::Whatever {
            message: format!("Error while creating writer: {}", e),
            source: Some(Box::new(e)),
        })?;

    // Set up the batch to write
    let batch = RecordBatch::try_new(schema, arrays).map_err(|e| Error::Whatever {
        message: format!("Error while creating batch: {}", e),
        source: Some(Box::new(e)),
    })?;

    // Run write op
    arrow_writer.write(&batch).map_err(|e| Error::Whatever {
        message: format!("Error while writing batch: {}", e),
        source: Some(Box::new(e)),
    })?;
    arrow_writer.close().map_err(|e| Error::Whatever {
        message: format!("Error while closing writer: {}", e),
        source: Some(Box::new(e)),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow::array::{BinaryArray, StringArray, UInt64Array};

    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use dicom::object::OpenFileOptions;
    use rstest::rstest;
    use std::collections::HashMap;

    use super::dicom_file_to_parquet;

    use std::fs::File;

    #[test]
    fn test_dicom_to_parquet_sopuid() {
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
        dicom_file_to_parquet(&dicom_file_path, parquet_file_path.path(), true, true, None)
            .unwrap();

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
    fn test_dicom_to_parquet_pixel_data(#[case] header_only: bool) {
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
            None,
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
                assert!(index.is_err());
            }
            false => {
                // PixelData should be present in the schema
                println!("{:?}", batch.schema());
                let column = batch.column(batch.schema().index_of("PixelData").unwrap());
                let pixel_data = column
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .unwrap()
                    .value(0);
                assert_eq!(*pixel_data, expected);

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

    #[test]
    fn test_dicom_file_to_parquet_with_tag_injection() {
        let dicom_file_path = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();

        // Prepare the overrides with the new tag
        let mut overrides = HashMap::new();
        overrides.insert("DataSetName".to_string(), "dataset".to_string());

        // Convert to parquet
        let parquet_file_path = tempfile::NamedTempFile::new().unwrap();
        dicom_file_to_parquet(
            &dicom_file_path,
            parquet_file_path.path(),
            false,
            false,
            Some(&overrides),
        )
        .unwrap();

        // Read the parquet file and verify the injected tag
        let parquet_file = File::open(parquet_file_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();

        // Check that the DataSetName is present and correct
        let index = batch.schema().index_of("DataSetName").unwrap();
        let column = batch.column(index);
        let data_set_name = column
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(data_set_name, "dataset");
    }

    #[test]
    fn test_dicom_file_to_parquet_file_meta() {
        let dicom_file_path = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();

        // Convert to parquet
        let parquet_file_path = tempfile::NamedTempFile::new().unwrap();
        dicom_file_to_parquet(
            &dicom_file_path,
            parquet_file_path.path(),
            false,
            false,
            None,
        )
        .unwrap();

        // Read the parquet file and verify TransferSyntaxUID
        let parquet_file = File::open(parquet_file_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();

        // Check that the DataSetName is present and correct
        let index = batch.schema().index_of("TransferSyntaxUID").unwrap();
        let column = batch.column(index);
        let tsuid = column
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0);
        assert_eq!(tsuid, "1.2.840.10008.1.2.1\0");
    }

    #[test]
    fn test_hash_but_not_store_pixels() {
        // Convert to parquet
        let dicom_file_path = dicom_test_files::path("pydicom/SC_rgb.dcm").unwrap();
        let parquet_file_path = tempfile::NamedTempFile::new().unwrap();
        let (header_only, hash_pixel_data) = (true, true);
        dicom_file_to_parquet(
            &dicom_file_path,
            parquet_file_path.path(),
            header_only,
            hash_pixel_data,
            None,
        )
        .unwrap();

        // Read parquet
        let parquet_file = File::open(parquet_file_path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_file).unwrap();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();

        // PixelData shouldn't be in the output
        assert!(
            batch.schema().index_of("PixelData").is_err(),
            "PixelData should not be present when header_only is true"
        );

        // But PixelData should have been hashed
        let column = batch.column(batch.schema().index_of("PixelDataHash").unwrap());
        let expected_hash = 10240938377863354873_u64;
        let hash = column
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        assert_eq!(hash, expected_hash);
    }

    #[rstest]
    #[case("pydicom/SC_rgb.dcm")]
    #[case("pydicom/CT_small.dcm")]
    #[case("pydicom/vlut_04.dcm")]
    #[case("pydicom/JPEG-LL.dcm")]
    #[case("pydicom/JPEG2000_UNC.dcm")]
    #[case("pydicom/MR-SIEMENS-DICOM-WithOverlays.dcm")]
    #[case("pydicom/MR2_J2KI.dcm")]
    #[case("pydicom/US1_UNCR.dcm")]
    #[case("pydicom/emri_small_RLE.dcm")]
    #[case("pydicom/bad_sequence.dcm")]
    #[case("pydicom/SC_rgb_expb_32bit_2frame.dcm")]
    #[case("pydicom/693_J2KR.dcm")]
    fn test_convert_various_files(#[case] dicom_file_name: &str) {
        let dicom_file_path = dicom_test_files::path(dicom_file_name).unwrap();
        let parquet_file_path = tempfile::NamedTempFile::new().unwrap();
        let parquet_file_path = parquet_file_path.path();
        dicom_file_to_parquet(&dicom_file_path, parquet_file_path, false, true, None).unwrap();
        assert!(parquet_file_path.is_file());
    }
}
