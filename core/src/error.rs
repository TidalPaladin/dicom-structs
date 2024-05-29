use arrow::datatypes::Schema;
use dicom::core::DataElement;
use snafu::prelude::*;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Unable to convert {}", elem.header().tag))]
    ConvertPrimitiveError { elem: DataElement },

    #[snafu(display("Unable to convert {}", elem.header().tag))]
    ConvertSequenceError { elem: DataElement },

    #[snafu(display("Unable to convert {}", elem.header().tag))]
    ConvertDicomError { elem: DataElement },

    #[snafu(display("Tag not found: {}", tag_name))]
    TagNotFound { tag_name: String },

    #[snafu(display("Error in schema {}", schema))]
    SchemaError { schema: Schema },

    #[snafu(display("Error hashing pixel data"))]
    PixelHashError,

    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error + Send + Sync>, Some)))]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}
