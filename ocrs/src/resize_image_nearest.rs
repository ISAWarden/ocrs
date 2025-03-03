use rten::{
    ops::{resize, CoordTransformMode, NearestMode, OpError, ResizeMode, ResizeTarget},
    TensorPool,
};
use rten_tensor::{AsView, Layout, NdTensor, NdTensorView, Tensor, TensorView, ViewData};

pub fn resize_image_nearest(
    input: NdTensorView<f32, 4>,
    size: [usize; 2],
) -> Result<Tensor, OpError> {
    let [batch, chans, _height, _width] = input.shape();
    let [out_height, out_width] = size;
    let out_shape = [batch, chans, out_height, out_width].map(|x| x as i32);
    resize(
        &TensorPool::new(),
        input.as_dyn(),
        ResizeTarget::Sizes(out_shape.as_slice().into()),
        ResizeMode::Nearest,
        CoordTransformMode::default(),
        NearestMode::default(),
    )
}
