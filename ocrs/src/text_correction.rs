pub fn text_error_correct(ocr_text: &str, reference_text: &str) -> String {
    let ocr_bytes = ocr_text.as_bytes();
    let reference_bytes = reference_text.as_bytes();
    let ocr_len = ocr_bytes.len();
    let reference_len = reference_bytes.len();

    if ocr_len == 0 {
        return ocr_text.to_string();
    }

    let mut best_correction: Option<Vec<u8>> = None;
    let mut min_errors = usize::MAX;

    // Iterate over possible starting positions in the reference where the first byte matches
    for start in 0..reference_len {
        if reference_bytes[start] != ocr_bytes[0] {
            continue;
        }

        let mut corrected = Vec::with_capacity(ocr_len);
        let mut errors = 0;
        let mut ref_pos = start;
        let mut ocr_pos = 0;

        while ocr_pos < ocr_len && ref_pos < reference_len {
            if ocr_bytes[ocr_pos] == reference_bytes[ref_pos] {
                corrected.push(reference_bytes[ref_pos]);
            } else {
                corrected.push(reference_bytes[ref_pos]);
                errors += 1;
                if errors > 10 {
                    break;
                }
            }
            ocr_pos += 1;
            ref_pos += 1;
        }

        // Check if we've processed all OCR characters and if errors are within limit
        if ocr_pos == ocr_len && errors <= 10 {
            // If the reference is shorter than OCR after start, fill remaining with OCR bytes (but this case shouldn't happen due to loop)
            while ocr_pos < ocr_len {
                corrected.push(ocr_bytes[ocr_pos]);
                ocr_pos += 1;
            }

            // Update best correction if current is better
            if errors < min_errors {
                min_errors = errors;
                best_correction = Some(corrected);
            }
        }
    }

    // Convert the best correction to a string, fallback to original OCR if invalid UTF-8 or no correction found
    if let Some(corrected_bytes) = best_correction {
        String::from_utf8(corrected_bytes).unwrap_or_else(|_| ocr_text.to_string())
    } else {
        ocr_text.to_string()
    }
}
