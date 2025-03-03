use crate::resize_image_nearest::resize_image_nearest;
use crate::{TypedArea, MASK_KEYS};
use anyhow::anyhow;
use rten::{Dimension, FloatOperators, Model, Operators, RunOptions};
use rten_imageproc::{find_contours, min_area_rect, simplify_polygon, RetrievalMode, RotatedRect};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, Tensor};

/// Parameters that control post-processing of text detection model outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct TextDetectorParams {
    /// Threshold for minimum area of returned rectangles.
    ///
    /// This can be used to filter out rects created by small false positives in
    /// the mask, at the risk of filtering out true positives. The more accurate
    /// the model producing the mask is, the smaller this value can be.
    pub min_area: f32,

    /// Threshold for per-pixel scores in output segmentation mask for
    /// classifying a pixel as text.
    pub text_threshold: f32,
}

impl Default for TextDetectorParams {
    fn default() -> TextDetectorParams {
        // Empirically chosen parameters for the initial model release.
        TextDetectorParams {
            // This area is quite large and can prevent detection of small /
            // single letter words.
            min_area: 100.,

            // Ideally the threshold would be 0.5 as a neutral value.
            text_threshold: 0.2,
        }
    }
}

/// Find the minimum-area oriented rectangles containing each connected
/// component in the binary mask `mask`.
fn find_connected_component_rects(
    mask: NdTensorView<bool, 2>,
    expand_dist: f32,
    min_area: f32,
) -> Vec<RotatedRect> {
    find_contours(mask, RetrievalMode::External)
        .iter()
        .filter_map(|poly| {
            let float_points: Vec<_> = poly.iter().map(|p| p.to_f32()).collect();
            let simplified = simplify_polygon(&float_points, 2. /* epsilon */);

            min_area_rect(&simplified).map(|mut rect| {
                rect.resize(
                    rect.width() + 2. * expand_dist,
                    rect.height() + 2. * expand_dist,
                );
                rect
            })
        })
        .filter(|r| r.area() >= min_area)
        .collect()
}

/// Text detector which finds the oriented bounding boxes of words in an input
/// image.
pub struct TextDetector {
    model: Model,
    params: TextDetectorParams,
    input_shape: Vec<Dimension>,
}

impl TextDetector {
    /// Initializate a DetectionModel from a trained RTen model.
    ///
    /// This will fail if the model doesn't have the expected inputs or outputs.
    pub fn from_model(model: Model, params: TextDetectorParams) -> anyhow::Result<TextDetector> {
        let input_id = model
            .input_ids()
            .first()
            .copied()
            .ok_or(anyhow!("model has no inputs"))?;
        let input_shape = model
            .node_info(input_id)
            .and_then(|info| info.shape())
            .ok_or(anyhow!("model does not specify expected input shape"))?;

        Ok(TextDetector {
            model,
            params,
            input_shape,
        })
    }

    /// Return the confidence threshold used to determine whether a pixel is
    /// text or not.
    pub fn threshold(&self) -> f32 {
        self.params.text_threshold
    }

    /// Detect text words in a greyscale image.
    ///
    /// `image` is a greyscale CHW image with values in the range `ZERO_VALUE` to
    /// `ZERO_VALUE + 1`. `model` is a model which takes an NCHW input tensor and
    /// returns a binary segmentation mask predicting whether each pixel is part of
    /// a text word or not. The image is padded and resized to the model's expected
    /// input size before performing detection.
    ///
    /// The result is an unsorted list of the oriented bounding rectangles of
    /// connected components (ie. text words) in the mask.
    pub fn detect_words(
        &self,
        image: NdTensorView<f32, 3>,
        debug: bool,
    ) -> anyhow::Result<Vec<TypedArea>> {
        let text_mask = self.detect_text_pixels(image, debug)?;

        let word_rects = MASK_KEYS
            .iter()
            .enumerate()
            .map(|(idx, key)| {
                // Distance to expand bounding boxes by. This is useful when the model is
                // trained to assign a positive label to pixels in a smaller area than the
                // ground truth, which may be done to create separation between adjacent
                // objects.
                let expand_dist = 3.;
                let binary_mask = text_mask.map(|x| *x == ((idx + 1) as u8));
                let word_rects = find_connected_component_rects(
                    binary_mask.view(),
                    expand_dist,
                    self.params.min_area,
                );
                word_rects.into_iter().map(|x| TypedArea {
                    rect: x,
                    area_type: *key,
                })
            })
            .flatten()
            .collect();
        Ok(word_rects)
    }

    /// Detect text pixels in an image.
    ///
    /// Takes a greyscale (CHW) input image and returns a probability map
    /// indicating whether each pixel in the input is text.
    ///
    /// See [detect_words](TextDetector::detect_words) for more details of
    /// expected input.
    pub fn detect_text_pixels(
        &self,
        image: NdTensorView<f32, 3>,
        debug: bool,
    ) -> anyhow::Result<NdTensor<u8, 2>> {
        let [img_chans, img_height, img_width] = image.shape();

        // Add batch dim
        let image = image.reshaped([1, img_chans, img_height, img_width]);

        let [_, _, Dimension::Fixed(in_height), Dimension::Fixed(in_width)] = self.input_shape[..]
        else {
            return Err(anyhow!("failed to get model dims"));
        };

        // Resize images to the text detection model's input size.
        let image = (image.size(2) != in_height || image.size(3) != in_width)
            .then(|| image.resize_image([in_height, in_width]))
            .transpose()?
            .map(|t| t.into_cow())
            .unwrap_or(image.as_dyn().as_cow());

        // Run text detection model to compute a probability mask indicating whether
        // each pixel is part of a text word or not.
        let text_mask: Tensor<i32> = self
            .model
            .run_one(
                image.view().into(),
                if debug {
                    Some(RunOptions {
                        timing: true,
                        verbose: false,
                        ..Default::default()
                    })
                } else {
                    None
                },
            )?
            .try_into()?;

        // Resize the mask to be compatible with the image input. Since resize_image can only
        // resize non-binary masks, we will convert to F32 then to the final 8-bit output
        let dequantized_mask = text_mask
            .slice((.., ..in_height, ..in_width))
            .map(|x| *x as f32)
            .into_shape([1, 1, in_height, in_width]);
        let resized_quantized_mask =
            resize_image_nearest(dequantized_mask.view(), [img_height, img_width])?
                .map(|x| x.round() as u8);
        Ok(resized_quantized_mask.into_shape([img_height, img_width]))
    }
}

#[cfg(test)]
mod tests {
    use rten_imageproc::{fill_rect, Point};
    use rten_tensor::prelude::*;
    use rten_tensor::NdTensor;

    use super::find_connected_component_rects;
    use crate::test_util::gen_rect_grid;

    #[test]
    fn test_find_connected_component_rects() {
        let mut mask = NdTensor::zeros([400, 400]);
        let (grid_h, grid_w) = (5, 5);
        let (rect_h, rect_w) = (10, 50);
        let rects = gen_rect_grid(
            Point::from_yx(10, 10),
            (grid_h, grid_w), /* grid_shape */
            (rect_h, rect_w), /* rect_size */
            (10, 5),          /* gap_size */
        );
        for r in rects.iter() {
            // Expand `r` because `fill_rect` does not set points along the
            // right/bottom boundary.
            let expanded = r.adjust_tlbr(0, 0, 1, 1);
            fill_rect(mask.view_mut(), expanded, true);
        }

        let min_area = 100.;
        let components = find_connected_component_rects(mask.view(), 0., min_area);
        assert_eq!(components.len() as i32, grid_h * grid_w);

        for c in components.iter() {
            let mut shape = [c.height().round() as i32, c.width().round() as i32];
            shape.sort();

            // We sort the dimensions before comparison here to be invariant to
            // different rotations of the connected component that cover the
            // same pixels.
            let mut expected_shape = [rect_h, rect_w];
            expected_shape.sort();

            assert_eq!(shape, expected_shape);
        }
    }
}
