use image::{
    load_from_memory, load_from_memory_with_format, DynamicImage, GenericImageView, GrayImage,
};
use ndarray::{Array, Axis, Dim, Shape};
use wasm_bindgen::prelude::*;

const MEANS: [f32; 3] = [0.485, 0.456, 0.406];
const STDS: [f32; 3] = [0.229, 0.224, 0.225];
#[wasm_bindgen]
pub fn u2net_preprocess(img_buf: Vec<u8>) -> Vec<f32> {
    let Ok(rgba_image) = load_from_memory_with_format(&img_buf, image::ImageFormat::Png) else {
        return vec![];
    };
    let dyn_img = DynamicImage::from(rgba_image);
    let resized_img = dyn_img.resize_exact(320, 320, image::imageops::FilterType::Lanczos3);
    let mut f32_rgb_buffer = Array::from_shape_vec(
        Shape::from(Dim([1, 320, 320, 3])),
        resized_img.to_rgb32f().to_vec(),
    )
    .unwrap();

    //Normalize Image
    f32_rgb_buffer
        .axis_iter_mut(Axis(3))
        .enumerate()
        .for_each(|(i, mut x)| {
            x -= MEANS[i];
            x /= STDS[i];
        });

    f32_rgb_buffer.swap_axes(1, 3);
    f32_rgb_buffer.swap_axes(2, 3);
    f32_rgb_buffer
        .as_standard_layout()
        .to_owned()
        .into_raw_vec()
}

#[wasm_bindgen]
pub fn u2net_postprocess(img_buf: Vec<u8>, mask_buf: Vec<f32>) -> Vec<u8> {
    let Ok(rgba_img) = load_from_memory(&img_buf) else {
        return vec![];
    };

    let max = mask_buf
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap()
        .to_owned();
    let min = mask_buf
        .iter()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap()
        .to_owned();
    let mask_u8: Vec<u8> = mask_buf
        .into_iter()
        .map(|x| {
            let val = ((x - min) / (max - min) * 255.).round() as u8;
            val
        })
        .collect();
    let Some(f32_mask) = GrayImage::from_vec(320, 320, mask_u8) else {
        return vec![];
    };
    let (o_w, o_h) = rgba_img.dimensions();
    let resized_mask = DynamicImage::from(f32_mask)
        .resize_exact(o_w, o_h, image::imageops::FilterType::Lanczos3)
        .to_luma32f();
    let mut res_rgba = DynamicImage::from(rgba_img).to_rgba8();
    res_rgba.enumerate_pixels_mut().for_each(|(w, h, p)| {
        p.0[3] = (255. * resized_mask.get_pixel(w, h).0[0]).round() as u8;
    });
    res_rgba.to_vec()
}
