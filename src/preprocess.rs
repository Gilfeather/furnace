//! ImageNet preprocessing for `image/jpeg` and `image/png` request bodies.
//!
//! Matches the usual torchvision pipeline
//! (`Resize(256)`, `CenterCrop(224)`, `ToTensor()`, `Normalize(mean, std)`), scaled to the
//! model's `[3, H, W]` input: the shorter side is resized to `min(H, W) × 256 / 224` with a
//! bilinear filter, then the centre `H × W` is cropped and normalized per channel into CHW
//! `f32`.

use image::imageops::{self, FilterType};
use image::{ImageFormat, ImageReader, Limits, RgbImage};
use std::io::Cursor;

/// Per-channel ImageNet mean (RGB)
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// Per-channel ImageNet standard deviation (RGB)
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Largest accepted width or height. Bounds the memory a small, highly compressed body
/// can make the decoder allocate.
pub const MAX_IMAGE_DIMENSION: u32 = 16_384;
/// Largest buffer the decoder may allocate (about 44 megapixels of RGB)
const MAX_DECODE_ALLOC: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKind {
    Jpeg,
    Png,
}

impl ImageKind {
    /// Map a media type essence (`image/jpeg`, `image/png`) to an image kind
    pub fn from_media_type(essence: &str) -> Option<Self> {
        if essence.eq_ignore_ascii_case("image/jpeg") || essence.eq_ignore_ascii_case("image/jpg") {
            Some(Self::Jpeg)
        } else if essence.eq_ignore_ascii_case("image/png") {
            Some(Self::Png)
        } else {
            None
        }
    }

    fn format(self) -> ImageFormat {
        match self {
            Self::Jpeg => ImageFormat::Jpeg,
            Self::Png => ImageFormat::Png,
        }
    }
}

/// `(H, W)` if a model with this input shape takes an RGB image (`[3, H, W]`)
pub fn image_input_size(shape: &[usize]) -> Option<(u32, u32)> {
    match *shape {
        [3, h, w] if h > 0 && w > 0 => Some((u32::try_from(h).ok()?, u32::try_from(w).ok()?)),
        _ => None,
    }
}

/// Decode `bytes` and turn the image into a normalized `3 × height × width` CHW buffer
pub fn preprocess_image(
    bytes: &[u8],
    kind: ImageKind,
    height: u32,
    width: u32,
) -> Result<Vec<f32>, image::ImageError> {
    let mut reader = ImageReader::with_format(Cursor::new(bytes), kind.format());
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
    limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
    limits.max_alloc = Some(MAX_DECODE_ALLOC);
    reader.limits(limits);
    let image = reader.decode()?.into_rgb8();
    Ok(normalize(&resize_and_crop(&image, height, width)))
}

/// Resize so the shorter side is `min(height, width) × 256 / 224`, then centre-crop
fn resize_and_crop(image: &RgbImage, height: u32, width: u32) -> RgbImage {
    let (src_w, src_h) = image.dimensions();
    let (new_w, new_h) = resized_size(src_w, src_h, height, width);
    let resized = if (new_w, new_h) == (src_w, src_h) {
        image.clone()
    } else {
        imageops::resize(image, new_w, new_h, FilterType::Triangle)
    };
    // Same rounding as torchvision's CenterCrop
    let left = ((new_w - width) as f64 / 2.0).round() as u32;
    let top = ((new_h - height) as f64 / 2.0).round() as u32;
    imageops::crop_imm(&resized, left, top, width, height).to_image()
}

/// `(width, height)` after resizing a `src_w × src_h` image for a `height × width` crop
fn resized_size(src_w: u32, src_h: u32, height: u32, width: u32) -> (u32, u32) {
    let (src_w, src_h) = (u64::from(src_w), u64::from(src_h));
    let (width, height) = (u64::from(width), u64::from(height));
    let short = height.min(width) * 256 / 224;
    // torchvision truncates the longer side
    let (mut new_w, mut new_h) = if src_w <= src_h {
        (short, short * src_h / src_w)
    } else {
        (short * src_w / src_h, short)
    };
    // Non-square inputs or extreme aspect ratios: grow until the crop fits
    if new_w < width || new_h < height {
        let scale = f64::max(width as f64 / src_w as f64, height as f64 / src_h as f64);
        new_w = ((src_w as f64 * scale).ceil() as u64).max(width);
        new_h = ((src_h as f64 * scale).ceil() as u64).max(height);
    }
    // Bounded by MAX_IMAGE_DIMENSION × the crop size, so this only saturates on absurd inputs
    let clamp = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    (clamp(new_w), clamp(new_h))
}

/// `[0, 255]` HWC bytes to normalized CHW `f32`
fn normalize(image: &RgbImage) -> Vec<f32> {
    let plane = image.width() as usize * image.height() as usize;
    let mut out = vec![0.0f32; 3 * plane];
    for (i, pixel) in image.pixels().enumerate() {
        for c in 0..3 {
            out[c * plane + i] =
                (f32::from(pixel.0[c]) / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageEncoder, Rgb, codecs::png::PngEncoder};

    fn encode_png(image: &RgbImage) -> Vec<u8> {
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes)
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        bytes
    }

    #[test]
    fn test_image_input_size() {
        assert_eq!(image_input_size(&[3, 224, 224]), Some((224, 224)));
        assert_eq!(image_input_size(&[3, 32, 64]), Some((32, 64)));
        assert_eq!(image_input_size(&[784]), None);
        assert_eq!(image_input_size(&[1, 28, 28]), None);
        assert_eq!(image_input_size(&[3, 0, 224]), None);
    }

    #[test]
    fn test_media_types() {
        assert_eq!(
            ImageKind::from_media_type("image/jpeg"),
            Some(ImageKind::Jpeg)
        );
        assert_eq!(
            ImageKind::from_media_type("IMAGE/PNG"),
            Some(ImageKind::Png)
        );
        assert_eq!(ImageKind::from_media_type("image/gif"), None);
    }

    #[test]
    fn test_solid_color_normalization() {
        let image = RgbImage::from_pixel(300, 200, Rgb([255, 0, 128]));
        let out = preprocess_image(&encode_png(&image), ImageKind::Png, 224, 224).unwrap();
        assert_eq!(out.len(), 3 * 224 * 224);
        let plane = 224 * 224;
        let expected = [
            (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
            (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
            (128.0 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
        ];
        for (c, expected) in expected.iter().enumerate() {
            let channel = &out[c * plane..(c + 1) * plane];
            assert!(channel.iter().all(|v| (v - expected).abs() < 1e-5));
        }
    }

    #[test]
    fn test_resize_and_center_crop_geometry() {
        // 512 x 256 (W x H): resized to 512 x 256 is a no-op; centre 224 x 224 crop
        // starts at (144, 16). Encode the x coordinate in red to check the crop offset.
        let image = RgbImage::from_fn(512, 256, |x, _| Rgb([(x / 2) as u8, 0, 0]));
        let cropped = resize_and_crop(&image, 224, 224);
        assert_eq!(cropped.dimensions(), (224, 224));
        assert_eq!(cropped.get_pixel(0, 0).0[0], 72);
        assert_eq!(cropped.get_pixel(223, 0).0[0], 183); // (144 + 223) / 2

        // Shorter side scaled to 256, longer side truncated like torchvision
        assert_eq!(resized_size(1546, 1213, 224, 224), (326, 256));
        assert_eq!(resized_size(1213, 1546, 224, 224), (256, 326));
        assert_eq!(resized_size(100, 100, 224, 224), (256, 256));
        // Crop wider than the resized image: scaled up to cover it
        assert_eq!(resized_size(100, 100, 32, 64), (64, 64));
    }

    #[test]
    fn test_small_and_non_square_inputs() {
        let image = RgbImage::from_pixel(10, 1000, Rgb([10, 20, 30]));
        let out = preprocess_image(&encode_png(&image), ImageKind::Png, 64, 32).unwrap();
        assert_eq!(out.len(), 3 * 64 * 32);
    }

    #[test]
    fn test_rejects_garbage_and_wrong_format() {
        assert!(preprocess_image(b"not an image", ImageKind::Png, 224, 224).is_err());
        let png = encode_png(&RgbImage::new(8, 8));
        assert!(preprocess_image(&png, ImageKind::Jpeg, 224, 224).is_err());
    }
}
