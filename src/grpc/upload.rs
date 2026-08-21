//! Assembling a streamed upload, and refusing one that will not fit.
//!
//! A client sends a header of [`BufferSpec`]s and then the bytes in chunks. This
//! validates the header before a single byte is reserved, tracks what has
//! arrived, and hands back finished [`NamedBuffer`]s — so the ECS tick that
//! applies the upload never does the transfer.
//!
//! The size ceilings live here because this is the only thing that enforces
//! them.

use tonic::Status;

use crate::scene::{BufferMeta, Dtype, NamedBuffer};

use super::proto::{BufferSpec, Chunk, Dtype as ProtoDtype};

/// Ceiling on the total declared size of a single object. Generous enough for
/// a large point cloud, small enough that a malformed or malicious header
/// cannot ask the process to reserve everything.
const MAX_OBJECT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Ceiling on how many named arrays one object may declare.
const MAX_BUFFERS_PER_OBJECT: usize = 64;

/// Ceiling on the text one string array may carry.
///
/// Strings travel inline in the header, which is a single gRPC message, so a
/// string array cannot grow past what the transport will decode — and hitting
/// that limit gives an opaque transport failure rather than anything a caller
/// can act on. This rejects it first, with a message that says what to do.
///
/// Well under the 8 MiB decode limit on purpose: a header carries every array's
/// declaration at once, so one array must not be able to fill it alone. A
/// million distinct names still fit, which no real name column approaches —
/// text columns are low-cardinality, and one that is not should arrive as an
/// index array and a dictionary, exactly as the hierarchy arrays do.
const MAX_TEXT_BYTES: u64 = 1024 * 1024;

/// How much of a declared buffer to reserve up front. Beyond this the vector
/// grows as bytes actually arrive, so an inflated `byte_length` costs the
/// server nothing until the client backs it up with real data.
const MAX_EAGER_RESERVE: u64 = 64 * 1024 * 1024;

/// An upload in progress: validated metadata plus the bytes received so far.
#[derive(Debug)]
pub(crate) struct Upload {
    metas: Vec<BufferMeta>,
    /// Byte length each buffer must reach before the upload is complete.
    declared: Vec<u64>,
    data: Vec<Vec<u8>>,
    /// Strings taken straight from the header, one entry per declared buffer
    /// and empty for every numeric one.
    ///
    /// Complete the moment the header arrives, unlike `data`, because a string
    /// array is not chunked. Its declared length is 0, so it needs no waiting
    /// and the completeness check in `finish` passes it without a special case.
    strings: Vec<Vec<String>>,
}

impl Upload {
    /// Validates a bare list of arrays: the same checks, with no object around
    /// Validated once here so every array that reaches the scene is well formed.
    /// what counts as a well-formed declaration.
    pub(crate) fn open(mut specs: Vec<BufferSpec>) -> Result<Self, Status> {
        if specs.is_empty() {
            return Err(Status::invalid_argument("header declared no buffers"));
        }
        if specs.len() > MAX_BUFFERS_PER_OBJECT {
            return Err(Status::invalid_argument(format!(
                "header declared {} buffers, limit is {MAX_BUFFERS_PER_OBJECT}",
                specs.len()
            )));
        }

        let mut metas = Vec::with_capacity(specs.len());
        let mut declared = Vec::with_capacity(specs.len());
        let mut data = Vec::with_capacity(specs.len());
        let mut strings = Vec::with_capacity(specs.len());
        let mut total: u64 = 0;

        for (index, spec) in specs.iter_mut().enumerate() {
            let meta = buffer_meta(index, spec)?;

            let expected = meta.byte_length().ok_or_else(|| {
                Status::invalid_argument(format!("buffer {index}: shape overflows a u64"))
            })?;
            if expected != spec.byte_length {
                return Err(Status::invalid_argument(format!(
                    "buffer {index} (\"{}\"): byte_length is {} but {} {} elements need {expected}",
                    meta.name,
                    spec.byte_length,
                    meta.dtype,
                    element_count(&meta.shape),
                )));
            }

            total = total
                .checked_add(held_bytes(expected, spec))
                .ok_or_else(|| Status::invalid_argument("declared object size overflows a u64"))?;
            if total > MAX_OBJECT_BYTES {
                return Err(Status::invalid_argument(format!(
                    "object declares {total} bytes, limit is {MAX_OBJECT_BYTES}"
                )));
            }

            if metas
                .iter()
                .any(|other: &BufferMeta| other.name == meta.name)
            {
                return Err(Status::invalid_argument(format!(
                    "duplicate buffer name \"{}\"",
                    meta.name
                )));
            }

            metas.push(meta);
            declared.push(expected);
            data.push(Vec::with_capacity(expected.min(MAX_EAGER_RESERVE) as usize));
            // Taken rather than cloned: the strings are the largest thing in
            // the header and the spec is not read again after validation.
            strings.push(std::mem::take(&mut spec.values));
        }

        Ok(Self {
            metas,
            declared,
            data,
            strings,
        })
    }

    /// Appends one chunk, rejecting anything out of order or oversized.
    pub(crate) fn write(&mut self, chunk: Chunk) -> Result<(), Status> {
        let index = chunk.buffer_index as usize;
        let buffer = self.data.get_mut(index).ok_or_else(|| {
            Status::invalid_argument(format!(
                "chunk names buffer {index}, header declared {}",
                self.declared.len()
            ))
        })?;

        let written = buffer.len() as u64;
        if chunk.offset != written {
            return Err(Status::invalid_argument(format!(
                "buffer {index}: chunk offset {} does not continue from {written}",
                chunk.offset
            )));
        }
        if chunk.data.is_empty() {
            return Err(Status::invalid_argument(format!(
                "buffer {index}: empty chunk at offset {written}"
            )));
        }

        let remaining = self.declared[index] - written;
        if chunk.data.len() as u64 > remaining {
            return Err(Status::invalid_argument(format!(
                "buffer {index}: chunk of {} bytes overruns the declared length by {}",
                chunk.data.len(),
                chunk.data.len() as u64 - remaining
            )));
        }

        buffer.extend_from_slice(&chunk.data);
        Ok(())
    }

    /// Confirms every array is complete and yields the finished bytes.
    pub(crate) fn finish(self) -> Result<Vec<NamedBuffer>, Status> {
        for (index, (buffer, declared)) in self.data.iter().zip(&self.declared).enumerate() {
            if buffer.len() as u64 != *declared {
                return Err(Status::data_loss(format!(
                    "buffer {index} (\"{}\"): stream ended with {} of {declared} bytes",
                    self.metas[index].name,
                    buffer.len()
                )));
            }
        }

        let buffers = self
            .metas
            .into_iter()
            .zip(self.data)
            .zip(self.strings)
            .map(|((meta, data), strings)| NamedBuffer {
                meta,
                data,
                strings,
            })
            .collect();

        Ok(buffers)
    }
}

/// What one declared buffer will actually occupy, against the size ceiling.
///
/// A string array declares 0 bytes and would otherwise slip past the ceiling
/// entirely, however much text it carries. The bound exists to stop one upload
/// asking for all the memory, and text is memory, so it is counted here
/// alongside the chunked bytes.
pub(crate) fn held_bytes(declared: u64, spec: &BufferSpec) -> u64 {
    let text: u64 = spec.values.iter().map(|value| value.len() as u64).sum();
    declared.saturating_add(text)
}

/// Validates one `BufferSpec` and converts it to its domain equivalent.
pub(crate) fn buffer_meta(index: usize, spec: &BufferSpec) -> Result<BufferMeta, Status> {
    if spec.name.is_empty() {
        return Err(Status::invalid_argument(format!(
            "buffer {index}: name is required"
        )));
    }
    if spec.shape.is_empty() {
        return Err(Status::invalid_argument(format!(
            "buffer {index} (\"{}\"): shape is required",
            spec.name
        )));
    }
    if spec.shape.contains(&0) {
        return Err(Status::invalid_argument(format!(
            "buffer {index} (\"{}\"): shape has a zero-length axis",
            spec.name
        )));
    }

    let dtype = match decode_dtype(spec.dtype) {
        Some(dtype) => dtype,
        None => {
            return Err(Status::invalid_argument(format!(
                "buffer {index} (\"{}\"): unknown dtype {}",
                spec.name, spec.dtype
            )));
        }
    };

    // An array is bytes or it is text, never both: `values` belongs to
    // DTYPE_STRING alone. Accepting it beside a numeric buffer would leave two
    // sources for the same element and no rule for which wins.
    //
    // The declared `byte_length` needs no check of its own here. A string
    // array's element size is 0, so the length the caller must declare comes
    // out at 0 and the equality test every buffer already faces rejects
    // anything else.
    if dtype == Dtype::Str {
        let expected = element_count(&spec.shape);
        if spec.values.len() as u64 != expected {
            return Err(Status::invalid_argument(format!(
                "buffer {index} (\"{}\"): shape needs {expected} strings but values has {}",
                spec.name,
                spec.values.len()
            )));
        }

        let text: u64 = spec.values.iter().map(|value| value.len() as u64).sum();
        if text > MAX_TEXT_BYTES {
            return Err(Status::invalid_argument(format!(
                "buffer {index} (\"{}\"): {text} bytes of text, limit is {MAX_TEXT_BYTES}. \
                 Strings travel whole in the header, so a text array cannot grow with the \
                 element count. Send an integer index per element and a string array of the \
                 distinct values instead.",
                spec.name
            )));
        }
    } else if !spec.values.is_empty() {
        return Err(Status::invalid_argument(format!(
            "buffer {index} (\"{}\"): values is set on a {dtype} array; it belongs to string \
             arrays alone",
            spec.name
        )));
    }

    Ok(BufferMeta {
        name: spec.name.clone(),
        dtype,
        shape: spec.shape.clone(),
    })
}

/// Wire dtype to its domain equivalent, `None` for unset or unrecognised.
pub(crate) fn decode_dtype(dtype: i32) -> Option<Dtype> {
    Some(match ProtoDtype::try_from(dtype) {
        Ok(ProtoDtype::Uint8) => Dtype::Uint8,
        Ok(ProtoDtype::Int8) => Dtype::Int8,
        Ok(ProtoDtype::Uint16) => Dtype::Uint16,
        Ok(ProtoDtype::Int16) => Dtype::Int16,
        Ok(ProtoDtype::Uint32) => Dtype::Uint32,
        Ok(ProtoDtype::Int32) => Dtype::Int32,
        Ok(ProtoDtype::Uint64) => Dtype::Uint64,
        Ok(ProtoDtype::Int64) => Dtype::Int64,
        Ok(ProtoDtype::Float32) => Dtype::Float32,
        Ok(ProtoDtype::Float64) => Dtype::Float64,
        Ok(ProtoDtype::String) => Dtype::Str,
        Ok(ProtoDtype::Unspecified) | Err(_) => return None,
    })
}

pub(crate) fn element_count(shape: &[u64]) -> u64 {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpc::convert::buffer_spec;

    /// A well-formed string array of `count` elements.
    fn text(count: usize) -> BufferSpec {
        BufferSpec {
            name: "res_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![count as u64],
            byte_length: 0,
            values: (0..count).map(|i| format!("R{i}")).collect(),
        }
    }

    /// Strings arrive complete in the header, so an upload of nothing else
    /// finishes without a single chunk. This is the case a client uploading a
    /// residue table hits, and it must not be mistaken for a truncated stream.
    #[test]
    fn a_string_array_is_complete_without_chunks() {
        let upload = Upload::open(vec![text(3)]).expect("header rejected");
        let buffers = upload.finish().expect("stream reported as truncated");

        assert_eq!(buffers.len(), 1);
        assert_eq!(buffers[0].meta.dtype, Dtype::Str);
        assert!(buffers[0].data.is_empty());
        assert_eq!(buffers[0].strings, vec!["R0", "R1", "R2"]);
    }

    /// The shape is the count. A mismatch means the client's per-element index
    /// array and its side array disagree about how many groups there are, which
    /// silently mislabels everything if it gets through.
    #[test]
    fn the_string_count_must_match_the_shape() {
        let mut spec = text(3);
        spec.shape = vec![4];
        let error = Upload::open(vec![spec]).expect_err("count mismatch accepted");
        assert!(
            error.message().contains("needs 4 strings"),
            "unhelpful message: {}",
            error.message()
        );
    }

    /// An array is bytes or it is text. Both at once leaves two sources for the
    /// same element and no rule for which one wins.
    #[test]
    fn a_numeric_array_may_not_carry_strings() {
        let spec = BufferSpec {
            name: "positions".into(),
            dtype: ProtoDtype::Float32 as i32,
            shape: vec![1, 3],
            byte_length: 12,
            values: vec!["nope".into()],
        };
        let error = Upload::open(vec![spec]).expect_err("values accepted on a float array");
        assert!(
            error.message().contains("string arrays alone"),
            "unhelpful message: {}",
            error.message()
        );
    }

    /// A string array declares no bytes. Claiming otherwise is caught by the
    /// same equality check every buffer faces, with no special case for text.
    #[test]
    fn a_string_array_may_not_declare_bytes() {
        let mut spec = text(3);
        spec.byte_length = 12;
        let error = Upload::open(vec![spec]).expect_err("byte_length accepted on a string array");
        assert!(
            error.message().contains("byte_length is 12"),
            "unhelpful message: {}",
            error.message()
        );
    }

    /// Text is memory even though it declares no bytes, so it has to count
    /// against the same ceiling. Without this a header of nothing but strings
    /// passes every size check while asking for unbounded allocation.
    #[test]
    fn text_counts_against_the_size_ceiling() {
        // Three names of three characters each, declaring no bytes.
        assert_eq!(held_bytes(0, &text_of(&["ALA", "GLY", "HOH"])), 9);

        // A numeric buffer carries no text, so nothing changes for it.
        let numeric = BufferSpec {
            name: "positions".into(),
            dtype: ProtoDtype::Float32 as i32,
            shape: vec![1, 3],
            byte_length: 12,
            values: Vec::new(),
        };
        assert_eq!(held_bytes(12, &numeric), 12);
    }

    /// Text that outgrows the header is refused here, where the caller learns
    /// what to do about it, rather than at the transport, where it does not.
    /// This is the per-element name column a large system would otherwise send.
    #[test]
    fn text_past_the_ceiling_is_refused_with_an_answer() {
        let count = (MAX_TEXT_BYTES as usize / 4) + 1;
        let spec = BufferSpec {
            name: "atom_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![count as u64],
            byte_length: 0,
            values: vec!["CA12".into(); count],
        };
        let error = Upload::open(vec![spec]).expect_err("oversized text accepted");
        let message = error.message();
        assert!(message.contains("atom_name"), "unnamed array: {message}");
        assert!(
            message.contains("index per element"),
            "no remedy offered: {message}"
        );
    }

    /// A dictionary is what the ceiling asks for, and it fits with room to
    /// spare: the distinct values of even a large name column are a few
    /// hundred entries, whatever the element count.
    #[test]
    fn a_dictionary_of_distinct_values_fits() {
        let distinct: Vec<String> = (0..1000).map(|i| format!("name{i}")).collect();
        let spec = BufferSpec {
            name: "atom_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![distinct.len() as u64],
            byte_length: 0,
            values: distinct,
        };
        Upload::open(vec![spec]).expect("a dictionary was rejected");
    }

    /// A string array of exactly these values.
    fn text_of(values: &[&str]) -> BufferSpec {
        BufferSpec {
            name: "res_name".into(),
            dtype: ProtoDtype::String as i32,
            shape: vec![values.len() as u64],
            byte_length: 0,
            values: values.iter().map(|v| (*v).to_owned()).collect(),
        }
    }

    /// Never echoed: a listing describes an array, and re-sending every residue
    /// name on each ListData would make describing cost as much as uploading.
    #[test]
    fn a_listing_reports_the_count_and_not_the_text() {
        let meta = BufferMeta {
            name: "res_name".into(),
            dtype: Dtype::Str,
            shape: vec![3],
        };
        let spec = buffer_spec(&meta);
        assert_eq!(spec.dtype, ProtoDtype::String as i32);
        assert_eq!(spec.shape, vec![3]);
        assert_eq!(spec.byte_length, 0);
        assert!(spec.values.is_empty());
    }
}
