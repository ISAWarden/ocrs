use std::collections::HashSet;

fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let v1: Vec<char> = s1.chars().collect();
    let v2: Vec<char> = s2.chars().collect();
    let len1 = v1.len();
    let len2 = v2.len();
    let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        for j in 0..=len2 {
            if i == 0 {
                dp[i][j] = j;
            } else if j == 0 {
                dp[i][j] = i;
            } else if v1[i - 1] == v2[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + std::cmp::min(
                    dp[i - 1][j],
                    std::cmp::min(dp[i][j - 1], dp[i - 1][j - 1]),
                );
            }
        }
    }

    dp[len1][len2]
}

pub fn text_error_correct(ocr_text: &str, reference_text: &str) -> String {
    let threshold = 2; // Adjust this threshold as needed
    let ocr_words: Vec<&str> = ocr_text.split_whitespace().collect();
    let reference_words: Vec<&str> = reference_text.split_whitespace().collect();
    let reference_set: HashSet<&str> = reference_words.iter().cloned().collect();
    let mut corrected_text = Vec::new();
    let mut last_index = 0;

    for word in ocr_words {
        let mut min_distance = usize::MAX;
        let mut closest_word = word;

        for (i, ref_word) in reference_words[last_index..].iter().enumerate() {
            let distance = levenshtein_distance(word, ref_word);
            if distance < min_distance {
                min_distance = distance;
                closest_word = ref_word;
                if min_distance == 0 {
                    break;
                }
            }
        }

        if min_distance <= threshold && reference_set.contains(closest_word) {
            corrected_text.push(closest_word);
            last_index += min_distance; // Update last_index to avoid reprocessing
        } else {
            corrected_text.push(word);
        }
    }

    corrected_text.join(" ")
}
