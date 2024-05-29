use dicom::core::DicomValue;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject, OpenFileOptions};
use std::borrow::Cow;
use std::fs::File;
use std::io::{self, Error, Read, Seek};
use std::path::Path;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use dicom::core::header::HasLength;
use dicom::core::header::Tag;

const DICM_PREFIX: &[u8; 4] = b"DICM";
const HASH_SEED: u64 = 0;

/// Checks if the input path has a ".dcm" or ".dicom" suffix, ignoring case
///
/// # Arguments
///
/// * `path` - A reference to a Path object
///
/// # Returns
///
/// * `bool` - True if the path has a ".dcm" or ".dicom" suffix, ignoring case
pub fn has_dicom_suffix(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => {
            let ext_lower = ext.to_lowercase();
            ext_lower == "dcm" || ext_lower == "dicom"
        }
        None => false,
    }
}

/// Checks if the file has a DICOM prefix by reading the preamble and checking for "DICM"
///
/// # Arguments
///
/// * `filename` - A reference to a Path object
///
/// # Returns
///
/// * `bool` - True if the file has a DICOM prefix
pub fn has_dicom_prefix(filename: &Path) -> bool {
    let mut file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => return false,
    };
    if file.seek(io::SeekFrom::Start(128)).is_err() {
        return false;
    }
    let mut buffer = [0; 4];
    if file.read_exact(&mut buffer).is_err() {
        return false;
    }
    &buffer == DICM_PREFIX
}

/// Determines if a given path points to a DICOM file by checking both the suffix and the prefix.
///
/// This function first checks if the file has a ".dcm" or ".dicom" suffix, ignoring case.
/// If the suffix is not present and the file has no extension, it then checks if the file contains the "DICM" prefix
/// at the appropriate position in the file.
///
/// # Arguments
///
/// * `path` - A reference to a Path object representing the file path
/// * `check_file` - A boolean indicating whether to check if the file exists. Disable this if you know the file is valid
///
/// # Returns
///
/// * `bool` - True if the file is identified as a DICOM file, otherwise false
#[allow(dead_code)]
pub fn is_dicom_file(path: &Path, check_file: bool) -> bool {
    let has_extension = path.extension().is_some();
    let has_suffix = has_dicom_suffix(path);
    match (has_suffix, has_extension) {
        (true, true) => !check_file || path.is_file(),
        (false, false) => has_dicom_prefix(path),
        _ => false,
    }
}

/// Compute a MD5 hash of (encapculated) pixel data
#[allow(dead_code)]
pub fn hash_pixel_data(dcm: &InMemDicomObject) -> Result<u64, Box<dyn std::error::Error>> {
    let element = dcm.element(tags::PIXEL_DATA)?;
    match element.value() {
        DicomValue::PixelSequence(v) => {
            // NOTE: Offset table isn't hashed, only 
            let fragments = v.fragments().into_iter().map(|f| Cow::from(f));
            // Hash each fragment and sum with wrapping overflow
            let hash_sum = fragments.fold(0u64, |acc, f| acc.wrapping_add(xxh3_64_with_seed(&f, HASH_SEED)));
            Ok(hash_sum)
        }
        DicomValue::Primitive(v) => {
            let pixel_data = v.to_bytes();
            Ok(xxh3_64_with_seed(&pixel_data, HASH_SEED))
        }
        _ => panic!("Should never encounter pixel data as something other than a primitive or pixel sequence"),
    }
}

// Open a DICOM file, optionally reading only the header
pub fn open_dicom(
    dicom_path: &Path,
    header_only: bool,
) -> Result<FileDicomObject<InMemDicomObject>, Error> {
    // Open the DICOM file, reading only to pixel data if header_only is true
    let obj = OpenFileOptions::new()
        .read_until(if header_only {
            tags::PIXEL_DATA
        } else {
            Tag(0xFFFF, 0xFFFF)
        })
        .open_file(dicom_path)
        .map_err(|e| Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Ensure no PixelData is being read if header_only is true
    if header_only {
        let has_pixel_data = obj
            .element_opt(tags::PIXEL_DATA)
            .unwrap_or(None)
            .is_some_and(|v| !v.is_empty());
        assert!(
            !has_pixel_data,
            "PixelData should not be present in the schema if header_only is true"
        )
    }
    Ok(obj)
}

#[cfg(test)]
mod tests {
    use super::{has_dicom_prefix, has_dicom_suffix, hash_pixel_data, is_dicom_file, open_dicom};

    use dicom::object::OpenFileOptions;
    use rstest::rstest;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_has_dicom_suffix() {
        let test_cases = vec![
            ("image.dcm", true),
            ("document.dicom", true),
            ("scan.DCM", true),
            ("report.DICOM", true),
            ("photo.jpg", false),
            ("archive.zip", false),
            ("archive", false),
        ];

        for (file_name, expected) in test_cases {
            let path = Path::new(file_name);
            assert_eq!(has_dicom_suffix(path), expected, "Failed for {}", file_name);
        }
    }

    #[test]
    fn test_has_dicom_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.dcm");

        // Create a file with DICM prefix
        {
            let mut file = File::create(&file_path).unwrap();
            file.write_all(&[0; 128]).unwrap();
            file.write_all(b"DICM").unwrap();
        }
        assert!(has_dicom_prefix(&file_path));

        // Create a file without DICM prefix
        {
            let file_path = dir.path().join("test_no_dicm.dcm");
            let mut file = File::create(&file_path).unwrap();
            file.write_all(&[0; 128]).unwrap();
            file.write_all(b"NOPE").unwrap();
            assert!(!has_dicom_prefix(&file_path));
        }

        // Create a file that is too short
        {
            let file_path = dir.path().join("test_short.dcm");
            let mut file = File::create(&file_path).unwrap();
            file.write_all(&[0; 100]).unwrap();
            assert!(!has_dicom_prefix(&file_path));
        }
    }

    #[test]
    fn test_is_dicom_file() {
        let dir = tempfile::tempdir().unwrap();

        // Test case: valid DICOM file with DICM prefix
        let valid_dicom_path = dir.path().join("valid.dcm");
        {
            let mut file = File::create(&valid_dicom_path).unwrap();
            file.write_all(&[0; 128]).unwrap();
            file.write_all(b"DICM").unwrap();
        }
        assert!(is_dicom_file(&valid_dicom_path, true));

        // Test case: valid DICOM file with DICM prefix but no suffix
        let valid_dicom_path = dir.path().join("valid");
        {
            let mut file = File::create(&valid_dicom_path).unwrap();
            file.write_all(&[0; 128]).unwrap();
            file.write_all(b"DICM").unwrap();
        }
        assert!(is_dicom_file(&valid_dicom_path, true));

        // Test case: non-DICOM file
        let non_dicom_path = dir.path().join("non_dicom.txt");
        {
            let mut file = File::create(&non_dicom_path).unwrap();
            file.write_all(b"This is not a DICOM file").unwrap();
        }
        assert!(!is_dicom_file(&non_dicom_path, true));

        // Test case: no-suffix non-DICOM file
        let short_file_path = dir.path().join("invalid");
        {
            let mut file = File::create(&short_file_path).unwrap();
            file.write_all(&[0; 100]).unwrap();
        }
        assert!(!is_dicom_file(&short_file_path, true));
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn test_open_dicom(#[case] header_only: bool) {
        let dicom_file_path = dicom_test_files::path("pydicom/CT_small.dcm").unwrap();

        // Attempt to open the DICOM file with the specified header_only option
        let result = open_dicom(&dicom_file_path, header_only);

        // Check that the file is opened successfully
        assert!(
            result.is_ok(),
            "Failed to open DICOM file: {:?}",
            result.err()
        );
    }

    #[rstest]
    #[case("pydicom/SC_rgb.dcm")]
    #[case("pydicom/CT_small.dcm")]
    #[case("pydicom/vlut_04.dcm")]
    #[case("pydicom/JPEG-LL.dcm")]
    #[case("pydicom/JPEG2000_UNC.dcm")]
    #[case("pydicom/MR-SIEMENS-DICOM-WithOverlays.dcm")]
    fn test_hash_pixel_data(#[case] file: &str) {
        // Get the expected PixelData from the DICOM
        let dicom_file_path = dicom_test_files::path(file).unwrap();
        let obj = OpenFileOptions::new()
            .open_file(dicom_file_path.clone())
            .unwrap();

        let hash = hash_pixel_data(&obj).unwrap();
        println!("{}", hash);
    }
}
