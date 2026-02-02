use core::f32;
use std::collections::{HashMap, VecDeque};

use rayon::prelude::*;
use rten::ctc::{CtcDecoder, CtcHypothesis};
use rten::{thread_pool, Dimension, FloatOperators};
use rten_imageproc::{
    bounding_rect, BoundingRect, Line, Point, PointF, Polygon, Rect, RotatedRect,
};
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView, NdTensorViewMut, Tensor};

use crate::errors::ModelRunError;
use crate::geom_util::{downwards_line, leftmost_edge, rightmost_edge};
use crate::model::Model;
use crate::preprocess::BLACK_VALUE;
use crate::text_items::{TextChar, TextLine};

#[allow(dead_code)]
/// Result produced by [detect_table_lines].
#[derive(Debug, Clone)]
pub struct TableLines {
    pub horizontal: Vec<Line>,
    pub vertical: Vec<Line>,
}

impl TableLines {
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.horizontal.is_empty() && self.vertical.is_empty()
    }
}

#[allow(dead_code)]
/// Detect prominent horizontal and vertical guides (eg. table borders) using simple
/// histogram thresholds over the grayscale OCR input.
pub fn detect_table_lines(image: NdTensorView<f32, 3>) -> TableLines {
    const MIN_H_LINE_LENGTH_RATIO: f32 = 0.2;
    const MIN_V_LINE_LENGTH_RATIO: f32 = 0.05;
    const MAX_H_THICKNESS_RATIO: f32 = 0.05;
    const MAX_V_THICKNESS_RATIO: f32 = 0.05;
    const DILATE_RADIUS: usize = 2; // Radius for merging nearby line candidates
    const H_KERNEL_DIVISOR: usize = 24; // Larger -> smaller kernel
    const V_KERNEL_DIVISOR: usize = 24;
    const MIN_KERNEL: usize = 5;
    const MAX_KERNEL: usize = 200;

    let grey = image.slice([0]);
    let height = grey.size(0);
    let width = grey.size(1);

    let otsu_threshold = otsu_threshold(&grey);
    let (binary_mask, ridge_h, ridge_v) = binarize_and_compute_ridge(&grey, otsu_threshold);

    let horiz_kernel = compute_kernel_length(width, H_KERNEL_DIVISOR, MIN_KERNEL, MAX_KERNEL);
    let vert_kernel = compute_kernel_length(height, V_KERNEL_DIVISOR, MIN_KERNEL, MAX_KERNEL);

    let mut horizontal_mask = {
        let opened = morphological_open(&binary_mask, width, height, horiz_kernel, 1);
        morphological_close(&opened, width, height, 3, 1)
    };
    let mut vertical_mask = {
        let opened = morphological_open(&binary_mask, width, height, 1, vert_kernel);
        morphological_close(&opened, width, height, 1, 3)
    };

    // Kết hợp ridge mask với morphology mask:
    // Nếu pixel nằm trên ridge, đảm bảo nó được giữ trong mask (OR).
    for (i, px) in horizontal_mask.iter_mut().enumerate() {
        if ridge_h.get(i).copied().unwrap_or(0) != 0 {
            *px = 1;
        }
    }
    for (i, px) in vertical_mask.iter_mut().enumerate() {
        if ridge_v.get(i).copied().unwrap_or(0) != 0 {
            *px = 1;
        }
    }

    let min_horizontal_length = (width as f32 * MIN_H_LINE_LENGTH_RATIO).max(20.0) as i32;
    let max_h_thickness = (height as f32 * MAX_H_THICKNESS_RATIO).max(6.0) as i32;
    let min_vertical_length = (height as f32 * MIN_V_LINE_LENGTH_RATIO).max(20.0) as i32;
    let max_v_thickness = (width as f32 * MAX_V_THICKNESS_RATIO).max(6.0) as i32;

    let mut horizontals = trace_line_components(
        &horizontal_mask,
        width,
        height,
        true,
        min_horizontal_length,
        max_h_thickness,
    );
    horizontals = merge_close_lines(horizontals, 5, true);

    let mut verticals = trace_line_components(
        &vertical_mask,
        width,
        height,
        false,
        min_vertical_length,
        max_v_thickness,
    );
    verticals = merge_close_lines(verticals, 5, false);

    let (horizontals, verticals) =
        filter_lines_by_cells(horizontals, verticals, width as i32, height as i32);

    TableLines {
        horizontal: horizontals,
        vertical: verticals,
    }
}

/// Merge lines that are close to each other
fn merge_close_lines(
    mut lines: Vec<Line<i32>>,
    threshold: i32,
    is_horizontal: bool,
) -> Vec<Line<i32>> {
    if lines.is_empty() {
        return lines;
    }

    // Sort lines by their position
    if is_horizontal {
        lines.sort_by_key(|l| l.start.y);
    } else {
        lines.sort_by_key(|l| l.start.x);
    }

    let mut merged = Vec::new();
    let mut current = lines[0];

    for line in lines.into_iter().skip(1) {
        let current_pos = if is_horizontal {
            current.start.y
        } else {
            current.start.x
        };
        let line_pos = if is_horizontal {
            line.start.y
        } else {
            line.start.x
        };

        if (line_pos - current_pos).abs() <= threshold {
            // Merge: take the average position
            let avg_pos = ((current_pos + line_pos) as f32 / 2.0).round() as i32;
            if is_horizontal {
                let start_x = current.start.x.min(line.start.x);
                let end_x = current.end.x.max(line.end.x);
                current = Line::from_endpoints(
                    Point::from_yx(avg_pos, start_x),
                    Point::from_yx(avg_pos, end_x),
                );
            } else {
                let start_y = current.start.y.min(line.start.y);
                let end_y = current.end.y.max(line.end.y);
                current = Line::from_endpoints(
                    Point::from_yx(start_y, avg_pos),
                    Point::from_yx(end_y, avg_pos),
                );
            }
        } else {
            merged.push(current);
            current = line;
        }
    }
    merged.push(current);

    merged
}

fn trace_line_components(
    mask: &[u8],
    width: usize,
    height: usize,
    is_horizontal: bool,
    min_major_length: i32,
    max_minor_thickness: i32,
) -> Vec<Line<i32>> {
    let mut lines = Vec::new();
    if width == 0 || height == 0 || mask.is_empty() {
        return lines;
    }

    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    let neighbors = [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if mask[idx] == 0 || visited[idx] {
                continue;
            }

            visited[idx] = true;
            queue.push_back(idx);

            let mut min_x = x as i32;
            let mut max_x = x as i32;
            let mut min_y = y as i32;
            let mut max_y = y as i32;
            let mut sum_x = x as i64;
            let mut sum_y = y as i64;
            let mut count = 1i64;

            while let Some(current) = queue.pop_front() {
                let cx = (current % width) as i32;
                let cy = (current / width) as i32;

                for (dx, dy) in neighbors {
                    let nx = cx + dx;
                    let ny = cy + dy;
                    if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                        continue;
                    }
                    let n_idx = ny as usize * width + nx as usize;
                    if visited[n_idx] || mask[n_idx] == 0 {
                        continue;
                    }
                    visited[n_idx] = true;
                    queue.push_back(n_idx);

                    min_x = min_x.min(nx);
                    max_x = max_x.max(nx);
                    min_y = min_y.min(ny);
                    max_y = max_y.max(ny);
                    sum_x += nx as i64;
                    sum_y += ny as i64;
                    count += 1;
                }
            }

            let major = if is_horizontal {
                (max_x - min_x + 1).max(0)
            } else {
                (max_y - min_y + 1).max(0)
            };
            let minor = if is_horizontal {
                (max_y - min_y + 1).max(0)
            } else {
                (max_x - min_x + 1).max(0)
            };

            if major < min_major_length || minor > max_minor_thickness {
                continue;
            }
            if count < min_major_length as i64 {
                continue;
            }

            // Filter out text regions: table lines should have high aspect ratio
            // Text regions typically have aspect ratio < 5, while table lines have aspect ratio > 10
            let aspect_ratio = if minor > 0 {
                major as f32 / minor as f32
            } else {
                0.0
            };
            const MIN_ASPECT_RATIO: f32 = 8.0; // Minimum aspect ratio for table lines
            if aspect_ratio < MIN_ASPECT_RATIO {
                continue;
            }

            // Also filter out components that are too "square" (likely text regions)
            // Table lines should be much longer than they are thick
            let area = major * minor;
            if area > 0 {
                // If the component is too large relative to its dimensions, it's likely text
                let density = count as f32 / area as f32;
                // Text regions have higher density (more pixels per unit area)
                // Table lines have lower density (thin lines)
                if density > 0.6 && aspect_ratio < 15.0 {
                    continue;
                }
            }
            
            if is_horizontal {
                let avg_y = (sum_y / count).clamp(0, height as i64 - 1) as i32;
                lines.push(Line::from_endpoints(
                    Point::from_yx(avg_y, min_x),
                    Point::from_yx(avg_y, max_x),
                ));
            } else {
                let avg_x = (sum_x / count).clamp(0, width as i64 - 1) as i32;
                lines.push(Line::from_endpoints(
                    Point::from_yx(min_y, avg_x),
                    Point::from_yx(max_y, avg_x),
                ));
            }
        }
    }

    lines
}

fn filter_lines_by_cells(
    horizontals: Vec<Line<i32>>,
    verticals: Vec<Line<i32>>,
    width: i32,
    height: i32,
) -> (Vec<Line<i32>>, Vec<Line<i32>>) {
    if horizontals.len() < 2 || verticals.len() < 2 {
        return (horizontals, verticals);
    }

    let min_cell_height = ((height as f32) * 0.01).max(4.0) as i32;
    let min_cell_width = ((width as f32) * 0.012).max(3.0) as i32;

    let mut sorted_h = horizontals;
    sorted_h.sort_by_key(|l| l.start.y);
    let mut sorted_v = verticals;
    sorted_v.sort_by_key(|l| l.start.x);

    let mut horiz_counts = vec![0usize; sorted_h.len()];
    let mut vert_counts = vec![0usize; sorted_v.len()];

    for hi in 0..sorted_h.len().saturating_sub(1) {
        let gap_h = (sorted_h[hi + 1].start.y - sorted_h[hi].start.y).abs();
        if gap_h < min_cell_height {
            continue;
        }
        for vi in 0..sorted_v.len().saturating_sub(1) {
            let gap_v = (sorted_v[vi + 1].start.x - sorted_v[vi].start.x).abs();
            if gap_v < min_cell_width {
                continue;
            }
            horiz_counts[hi] += 1;
            horiz_counts[hi + 1] += 1;
            vert_counts[vi] += 1;
            vert_counts[vi + 1] += 1;
        }
    }

    let horiz_required = if sorted_v.len() >= 4 { 2 } else { 1 };
    let vert_required = if sorted_h.len() >= 4 { 2 } else { 1 };
    let filtered_h: Vec<Line<i32>> = sorted_h
        .into_iter()
        .zip(horiz_counts.into_iter())
        .filter_map(|(line, count)| (count >= horiz_required).then_some(line))
        .collect();

    let filtered_v: Vec<Line<i32>> = sorted_v
        .into_iter()
        .zip(vert_counts.into_iter())
        .filter_map(|(line, count)| (count >= vert_required).then_some(line))
        .collect();

    (filtered_h, filtered_v)
}

fn otsu_threshold(image: &NdTensorView<f32, 2>) -> f32 {
    const BINS: usize = 256;

    let height = image.size(0);
    let width = image.size(1);
    let mut hist = [0u32; BINS];

    for y in 0..height {
        for x in 0..width {
            let value = image[[y, x]];
            let scaled = ((value - BLACK_VALUE) * 255.0).clamp(0.0, 255.0);
            let idx = scaled as usize;
            hist[idx.min(BINS - 1)] += 1;
        }
    }

    let total = (height * width) as f32;
    if total == 0.0 {
        return BLACK_VALUE;
    }

    let mut sum_total = 0f32;
    for (i, &count) in hist.iter().enumerate() {
        sum_total += i as f32 * count as f32;
    }

    let mut sum_bg = 0f32;
    let mut weight_bg = 0f32;
    let mut max_variance = -1f32;
    let mut threshold = 0usize;

    for (i, &count) in hist.iter().enumerate() {
        weight_bg += count as f32;
        if weight_bg == 0.0 {
            continue;
        }

        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += i as f32 * count as f32;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;
        let between_var = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);

        if between_var > max_variance {
            max_variance = between_var;
            threshold = i;
        }
    }

    BLACK_VALUE + (threshold as f32 / (BINS as f32 - 1.0))
}

/// Binarize ảnh (giữ pixel "tối") và đồng thời tính ridge masks bằng Hessian eigenvalues
/// để tìm "xương sống" của line cong/mòn.
/// Trả về (binary_mask, ridge_h, ridge_v).
fn binarize_and_compute_ridge(
    image: &NdTensorView<f32, 2>,
    threshold: f32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let height = image.size(0);
    let width = image.size(1);
    let len = height * width;

    let mut mask = vec![0u8; len];
    let mut ridge_h = vec![0u8; len];
    let mut ridge_v = vec![0u8; len];

    // Ngưỡng thực nghiệm: với ảnh đã chuẩn hoá về 0..1
    let min_strength: f32 = 0.02;
    let anisotropy_ratio: f32 = 0.25; // |lambda_small| <= ratio * |lambda_large|

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let c = image[[y, x]];
            mask[idx] = if c <= threshold { 1 } else { 0 };

            if y == 0 || y == height - 1 || x == 0 || x == width - 1 {
                continue;
            }

            let l = image[[y, x - 1]];
            let r = image[[y, x + 1]];
            let u = image[[y - 1, x]];
            let d = image[[y + 1, x]];
            let ul = image[[y - 1, x - 1]];
            let ur = image[[y - 1, x + 1]];
            let dl = image[[y + 1, x - 1]];
            let dr = image[[y + 1, x + 1]];

            let ix = (r - l) * 0.5;
            let iy = (d - u) * 0.5;
            let ixx = r - 2.0 * c + l;
            let iyy = d - 2.0 * c + u;
            let ixy = (dr - dl - ur + ul) * 0.25;

            let tr = ixx + iyy;
            let det = ixx * iyy - ixy * ixy;
            let disc = tr * tr - 4.0 * det;
            if disc <= 0.0 {
                continue;
            }
            let sqrt_disc = disc.sqrt();
            let l1 = 0.5 * (tr + sqrt_disc);
            let l2 = 0.5 * (tr - sqrt_disc);
            let (lambda, lambda_other) = if l1.abs() >= l2.abs() {
                (l1, l2)
            } else {
                (l2, l1)
            };

            let strength = lambda.abs();
            if lambda >= 0.0 || strength < min_strength {
                continue;
            }
            if lambda_other.abs() > anisotropy_ratio * strength {
                continue;
            }

            if ix.abs() >= iy.abs() {
                ridge_v[idx] = 1;
            } else {
                ridge_h[idx] = 1;
            }
        }
    }

    (mask, ridge_h, ridge_v)
}

fn compute_kernel_length(length: usize, divisor: usize, min_size: usize, max_size: usize) -> usize {
    if length == 0 {
        return 1;
    }
    let mut size = length / divisor.max(1);
    if size < min_size {
        size = min_size;
    }
    if size > max_size {
        size = max_size;
    }
    size = size.min(length);
    size.max(1)
}

fn morphological_close(
    mask: &[u8],
    width: usize,
    height: usize,
    kernel_w: usize,
    kernel_h: usize,
) -> Vec<u8> {
    let kernel_w = kernel_w.max(1).min(width.max(1));
    let kernel_h = kernel_h.max(1).min(height.max(1));

    let dilated = dilate(mask, width, height, kernel_w, kernel_h);
    erode(&dilated, width, height, kernel_w, kernel_h)
}

fn morphological_open(
    mask: &[u8],
    width: usize,
    height: usize,
    kernel_w: usize,
    kernel_h: usize,
) -> Vec<u8> {
    let kernel_w = kernel_w.max(1).min(width.max(1));
    let kernel_h = kernel_h.max(1).min(height.max(1));

    let eroded = erode(mask, width, height, kernel_w, kernel_h);
    dilate(&eroded, width, height, kernel_w, kernel_h)
}

fn dilate(data: &[u8], width: usize, height: usize, kernel_w: usize, kernel_h: usize) -> Vec<u8> {
    let horizontal = if kernel_w > 1 {
        axis_window_op(data, width, height, kernel_w, true, false)
    } else {
        data.to_vec()
    };
    if kernel_h > 1 {
        axis_window_op(&horizontal, width, height, kernel_h, false, false)
    } else {
        horizontal
    }
}

fn erode(data: &[u8], width: usize, height: usize, kernel_w: usize, kernel_h: usize) -> Vec<u8> {
    let horizontal = if kernel_w > 1 {
        axis_window_op(data, width, height, kernel_w, true, true)
    } else {
        data.to_vec()
    };
    if kernel_h > 1 {
        axis_window_op(&horizontal, width, height, kernel_h, false, true)
    } else {
        horizontal
    }
}

fn axis_window_op(
    data: &[u8],
    width: usize,
    height: usize,
    kernel_len: usize,
    horizontal: bool,
    require_full: bool,
) -> Vec<u8> {
    if kernel_len <= 1 {
        return data.to_vec();
    }

    let mut output = vec![0u8; data.len()];
    let half = kernel_len / 2;

    if horizontal {
        for y in 0..height {
            let row = &data[y * width..(y + 1) * width];
            let mut prefix = vec![0u32; width + 1];
            for x in 0..width {
                prefix[x + 1] = prefix[x] + row[x] as u32;
            }
            for x in 0..width {
                let left = x.saturating_sub(half);
                let right = (x + kernel_len - half).min(width);
                let window_len = right - left;
                let sum = prefix[right] - prefix[left];
                let idx = y * width + x;
                output[idx] = if require_full {
                    (sum as usize == window_len) as u8
                } else {
                    (sum > 0) as u8
                };
            }
        }
    } else {
        for x in 0..width {
            let mut prefix = vec![0u32; height + 1];
            for y in 0..height {
                prefix[y + 1] = prefix[y] + data[y * width + x] as u32;
            }
            for y in 0..height {
                let top = y.saturating_sub(half);
                let bottom = (y + kernel_len - half).min(height);
                let window_len = bottom - top;
                let sum = prefix[bottom] - prefix[top];
                let idx = y * width + x;
                output[idx] = if require_full {
                    (sum as usize == window_len) as u8
                } else {
                    (sum > 0) as u8
                };
            }
        }
    }

    output
}

/// Return a polygon which contains all the rects in `words`.
///
/// `words` is assumed to be a series of disjoint rectangles ordered from left
/// to right. The returned points are arranged in clockwise order starting from
/// the top-left point.
///
/// There are several ways to compute a polygon for a line. The simplest is
/// to use [min_area_rect] on the union of the line's points. However the result
/// will not tightly fit curved lines. This function returns a polygon which
/// closely follows the edges of individual words.
fn line_polygon(words: &[RotatedRect]) -> Vec<Point> {
    let mut polygon = Vec::new();

    let floor_point = |p: PointF| Point::from_yx(p.y as i32, p.x as i32);

    // Add points from top edges, in left-to-right order.
    for word_rect in words.iter() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(left.start));
        polygon.push(floor_point(right.start));
    }

    // Add points from bottom edges, in right-to-left order.
    for word_rect in words.iter().rev() {
        let (left, right) = (
            downwards_line(leftmost_edge(word_rect)),
            downwards_line(rightmost_edge(word_rect)),
        );
        polygon.push(floor_point(right.end));
        polygon.push(floor_point(left.end));
    }

    polygon
}

/// Compute width to resize a text line image to, for a given height.
fn resized_line_width(orig_width: i32, orig_height: i32, height: i32) -> u32 {
    let min_width = 10.;

    // A larger maximum width avoids horizontally squashing long input lines,
    // affecting accuracy. However it also increases the processing time.
    //
    // The current value was chosen to be large enough to produce good results
    // on screenshots taken from the longest lines in English Wikipedia articles
    // (image size approx 1860x30, 150 characters).
    //
    // The widest image seen during training may be constrained to a shorter
    // value than this, but we rely on the model's ability to generalize to
    // longer sequences.
    let max_width = 2400.;

    let aspect_ratio = orig_width as f32 / orig_height as f32;
    (height as f32 * aspect_ratio).clamp(min_width, max_width) as u32
}

/// Details about a text line needed to prepare the input to the text
/// recognition model.
#[derive(Clone)]
struct TextRecLine {
    /// Index of this line in the list of lines found in the image.
    index: usize,

    /// Region of the image containing this line.
    region: Polygon,

    /// Width to resize this line to.
    resized_width: u32,
}

fn prepare_text_line(
    image: NdTensorView<f32, 3>,
    page_rect: Rect,
    line_region: &Polygon,
    resized_width: u32,
    output_height: usize,
) -> NdTensor<f32, 2> {
    // Page rect adjusted to only contain coordinates that are valid for
    // indexing into the input image.
    let page_index_rect = page_rect.adjust_tlbr(0, 0, -1, -1);

    let grey_chan = image.slice([0]);

    let line_rect = line_region.bounding_rect();
    
    // Mở rộng bounding rect một chút để đảm bảo bao phủ đủ các ký tự
    // Đặc biệt quan trọng cho các ký tự ở giữa line có thể bị polygon bỏ qua
    let expand_pixels = 2; // Mở rộng 2 pixels mỗi phía
    let expanded_rect = line_rect.adjust_tlbr(
        -expand_pixels,
        -expand_pixels,
        expand_pixels,
        expand_pixels,
    );
    
    // Clamp expanded rect vào page bounds
    let crop_top = expanded_rect.top().max(0) as usize;
    let crop_bottom = expanded_rect.bottom().min(page_rect.height() as i32) as usize;
    let crop_left = expanded_rect.left().max(0) as usize;
    let crop_right = expanded_rect.right().min(page_rect.width() as i32) as usize;
    
    let cropped_height = crop_bottom - crop_top;
    let cropped_width = crop_right - crop_left;
    
    let mut line_img = NdTensor::full(
        [cropped_height, cropped_width],
        BLACK_VALUE,
    );

    // Copy pixels từ original image: chỉ copy pixels trong polygon
    let offset_x = crop_left as i32;
    let offset_y = crop_top as i32;
    
    for in_p in line_region.fill_iter() {
        // Map từ original image coordinates sang cropped image coordinates
        let crop_x = in_p.x - offset_x;
        let crop_y = in_p.y - offset_y;
        
        if !page_index_rect.contains_point(in_p) {
            continue;
        }
        
        if crop_y >= 0 && crop_y < cropped_height as i32 &&
           crop_x >= 0 && crop_x < cropped_width as i32 {
            line_img[[crop_y as usize, crop_x as usize]] =
                grey_chan[[in_p.y as usize, in_p.x as usize]];
        }
    }

    let resized_line_img = line_img
        .reshaped([1, 1, line_img.size(0), line_img.size(1)])
        .resize_image([output_height, resized_width as usize])
        .unwrap();

    let out_shape = [resized_line_img.size(2), resized_line_img.size(3)];
    resized_line_img.into_shape(out_shape)
}

/// Prepare an NCHW tensor containing a batch of text line images, for input
/// into the text recognition model.
///
/// For each line in `lines`, the line region is extracted from `image`, resized
/// to a fixed `output_height` and a line-specific width, then copied to the
/// output tensor. Lines in the batch can have different widths, so the output
/// is padded on the right side to a common width of `output_width`.
fn prepare_text_line_batch(
    image: &NdTensorView<f32, 3>,
    lines: &[TextRecLine],
    page_rect: Rect,
    output_height: usize,
    output_width: usize,
) -> NdTensor<f32, 4> {
    let mut output = NdTensor::full([lines.len(), 1, output_height, output_width], BLACK_VALUE);

    for (group_line_index, line) in lines.iter().enumerate() {
        let resized_line_img = prepare_text_line(
            image.view(),
            page_rect,
            &line.region,
            line.resized_width,
            output_height,
        );
        output
            .slice_mut((group_line_index, 0, .., ..(line.resized_width as usize)))
            .copy_from(&resized_line_img);
    }

    output
}

/// Return the bounding rectangle of the slice of a polygon with X coordinates
/// between `min_x` and `max_x` inclusive.
fn polygon_slice_bounding_rect(
    poly: Polygon<i32, &[Point]>,
    min_x: i32,
    max_x: i32,
) -> Option<Rect> {
    poly.edges()
        .filter_map(|e| {
            let e = e.rightwards();

            // Filter out edges that don't overlap [min_x, max_x].
            if (e.start.x < min_x && e.end.x < min_x) || (e.start.x > max_x && e.end.x > max_x) {
                return None;
            }

            // Truncate edge to [min_x, max_x].
            let trunc_edge_start = e
                .to_f32()
                .y_for_x(min_x as f32)
                .map_or(e.start, |y| Point::from_yx(y.round() as i32, min_x));

            let trunc_edge_end = e
                .to_f32()
                .y_for_x(max_x as f32)
                .map_or(e.end, |y| Point::from_yx(y.round() as i32, max_x));

            Some(Line::from_endpoints(trunc_edge_start, trunc_edge_end))
        })
        .fold(None, |bounding_rect, e| {
            let edge_br = e.bounding_rect();
            bounding_rect.map(|br| br.union(edge_br)).or(Some(edge_br))
        })
}

/// Method used to decode sequence model outputs to a sequence of labels.
///
/// See [CtcDecoder] for more details.
#[derive(Copy, Clone, Default)]
pub enum DecodeMethod {
    #[default]
    Greedy,
    BeamSearch {
        width: u32,
    },
}

#[derive(Clone, Default)]
pub struct RecognitionOpt<'a> {
    pub debug: bool,

    /// Method used to decode character sequence outputs to character values.
    pub decode_method: DecodeMethod,

    pub alphabet: &'a str,

    pub excluded_char_labels: Option<&'a [usize]>,
}

/// Input and output from recognition for a single text line.
struct LineRecResult {
    /// Input to the recognition model.
    line: TextRecLine,

    /// Length of input sequences to recognition model, padded so that all
    /// lines in batch have the same length.
    rec_input_len: usize,

    /// Length of output sequences from recognition model, used as input to
    /// CTC decoding.
    ctc_input_len: usize,

    /// Output label sequence produced by CTC decoding.
    ctc_output: CtcHypothesis,
}

/// Combine information from the input and output of text line recognition
/// to produce [TextLine]s containing character sequences and bounding boxes
/// for each line.
///
/// Entries in the result may be `None` if no text was recognized for a line.
fn text_lines_from_recognition_results(
    results: &[LineRecResult],
    alphabet: &str,
) -> Vec<Option<TextLine>> {
    results
        .iter()
        .map(|result| {
            let line_rect = result.line.region.bounding_rect();
            let x_scale_factor = (line_rect.width() as f32) / (result.line.resized_width as f32);
            // Calculate how much the recognition model downscales the image
            // width. We assume this will be an integer factor, or close to it
            // if the input width is not an exact multiple of the downscaling
            // factor.
            let downsample_factor =
                (result.rec_input_len as f32 / result.ctc_input_len as f32).round() as u32;

            let steps = result.ctc_output.steps();
            let text_line: Vec<TextChar> = steps
                .iter()
                .enumerate()
                .filter_map(|(i, step)| {
                    // X coord range of character in line recognition input image.
                    let start_x = step.pos * downsample_factor;
                    let end_x = if let Some(next_step) = steps.get(i + 1) {
                        next_step.pos * downsample_factor
                    } else {
                        result.line.resized_width
                    };

                    // Map X coords to those of the input image.
                    let [start_x, end_x] = [start_x, end_x]
                        .map(|x| line_rect.left() + (x as f32 * x_scale_factor) as i32);

                    // Since the recognition input is padded, it is possible to
                    // get predicted characters in the output with positions
                    // that correspond to the padding region, and thus are
                    // outside the bounds of the original line. Ignore these.
                    if start_x >= line_rect.right() {
                        return None;
                    }

                    let char = alphabet
                        .chars()
                        // Index `0` is reserved for blank character and `i + 1` is used as training
                        // label for character at index `i` of `alphabet` string.  Here we're
                        // subtracting 1 to get the actual index from the output label
                        //
                        // See https://github.com/robertknight/ocrs-models/blob/3d98fc655d6fd4acddc06e7f5d60a55b55748a48/ocrs_models/datasets/util.py#L113
                        .nth((step.label - 1) as usize)
                        .unwrap_or('?');

                    Some(TextChar {
                        char,
                        rect: polygon_slice_bounding_rect(
                            result.line.region.borrow(),
                            start_x,
                            end_x,
                        )
                        .expect("invalid X coords"),
                    })
                })
                .collect();

            let text_line = insert_missing_spaces(text_line);

            if text_line.is_empty() {
                None
            } else {
                Some(TextLine::new(text_line))
            }
        })
        .collect()
}

fn insert_missing_spaces(chars: Vec<TextChar>) -> Vec<TextChar> {
    if chars.len() < 2 {
        return chars;
    }

    // Helper: determine if a character should be treated as lowercase for width calculation
    fn is_lowercase_for_width(ch: char, idx: usize, chars: &[TextChar]) -> bool {
        if ch.is_lowercase() {
            return true;
        }
        if ch == '?' {
            // '?' is treated as lowercase if surrounded by lowercase characters
            let prev_is_lower = idx > 0 && chars[idx - 1].char.is_lowercase();
            let next_is_lower = idx + 1 < chars.len() && chars[idx + 1].char.is_lowercase();
            return prev_is_lower || next_is_lower;
        }
        false
    }

    // Calculate normalized character widths
    // Rule: uppercase width = 2 * lowercase width
    // So if we see an uppercase char with width W, its normalized width is W/2
    let char_types: Vec<bool> = chars
        .iter()
        .enumerate()
        .map(|(idx, c)| is_lowercase_for_width(c.char, idx, &chars))
        .collect();

    let normalized_widths: Vec<i32> = chars
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let raw_width = c.rect.width();
            if char_types[idx] {
                raw_width
            } else if c.char.is_uppercase() {
                // Uppercase: normalize by dividing by 2
                (raw_width as f32 / 2.0).round() as i32
            } else {
                // Other characters (punctuation, etc.): use raw width
                raw_width
            }
        })
        .collect();

    // Calculate word-based average character width
    // x = word_width / (char_count + uppercase_count)
    // We'll calculate this for each "word" (sequence of non-space chars)
    let mut word_widths: Vec<f32> = Vec::new();
    let mut current_word_start: Option<usize> = None;

    for (idx, ch) in chars.iter().enumerate() {
        if ch.char != ' ' {
            if current_word_start.is_none() {
                current_word_start = Some(idx);
            }
        } else {
            // End of word
            if let Some(start) = current_word_start {
                let word_chars = &chars[start..idx];
                if !word_chars.is_empty() {
                    let word_width = word_chars.last().unwrap().rect.right()
                        - word_chars.first().unwrap().rect.left();
                    let char_count = word_chars.len();
                    let uppercase_count: usize =
                        word_chars.iter().filter(|c| c.char.is_uppercase()).count();
                    let normalized_char_count = char_count + uppercase_count;
                    if normalized_char_count > 0 {
                        let avg_char_width = word_width as f32 / normalized_char_count as f32;
                        word_widths.push(avg_char_width);
                    }
                }
                current_word_start = None;
            }
        }
    }

    // Handle last word if exists
    if let Some(start) = current_word_start {
        let word_chars = &chars[start..];
        if !word_chars.is_empty() {
            let word_width =
                word_chars.last().unwrap().rect.right() - word_chars.first().unwrap().rect.left();
            let char_count = word_chars.len();
            let uppercase_count: usize =
                word_chars.iter().filter(|c| c.char.is_uppercase()).count();
            let normalized_char_count = char_count + uppercase_count;
            if normalized_char_count > 0 {
                let avg_char_width = word_width as f32 / normalized_char_count as f32;
                word_widths.push(avg_char_width);
            }
        }
    }

    // Calculate median based on normalized widths and word-based widths
    let mut sorted_normalized = normalized_widths.clone();
    sorted_normalized.sort_unstable();
    let median_normalized = sorted_normalized[sorted_normalized.len() / 2].max(1);

    // Use word-based average if available, otherwise use normalized median
    let median_width = if !word_widths.is_empty() {
        let mut sorted_word_widths = word_widths.clone();
        sorted_word_widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (sorted_word_widths[sorted_word_widths.len() / 2] as f32).round() as i32
    } else {
        median_normalized
    }
    .max(1);

    // Thresholds for space detection (based on normalized lowercase width)
    let min_space_width = (median_width / 2).max(2);
    // Gap threshold: if gap between consecutive chars > this, insert space
    // Use 1.5x normalized median width as threshold
    let gap_threshold = ((median_width as f32) * 1.5).round() as i32;

    // Store char info before moving chars
    let char_chars: Vec<char> = chars.iter().map(|c| c.char).collect();
    let char_rects: Vec<Rect> = chars.iter().map(|c| c.rect).collect();

    // Helper function to log a word
    fn log_word(word_chars: &[TextChar], line_text: &str) {
        let word_text: String = word_chars.iter().map(|c| c.char).collect();
        println!(
            "[recognition-debug] word \"{}\" in line \"{}\" has {} chars",
            word_text,
            line_text,
            word_chars.len()
        );
        for (idx, ch) in word_chars.iter().enumerate() {
            let rect = ch.rect;
            let width = rect.width();
            let gap_to_next = word_chars
                .get(idx + 1)
                .map(|next| next.rect.left() - rect.right())
                .unwrap_or(0);
            println!(
                "  idx {:02} char '{}' width {:3} left {:4} right {:4} gap_to_next {:4}",
                idx,
                ch.char,
                width,
                rect.left(),
                rect.right(),
                gap_to_next
            );
        }
    }

    // Get line text for logging
    let line_text: String = chars.iter().map(|c| c.char).collect();

    // Pass 1: Handle wide characters (may contain embedded space)
    // Only insert space when comparing with next character in the same word
    let mut pass1_result = Vec::with_capacity(chars.len() * 2);
    let mut word_start: Option<usize> = None;

    for (idx, mut ch) in chars.into_iter().enumerate() {
        // Track word boundaries (sequences of non-space chars)
        if ch.char != ' ' {
            if word_start.is_none() {
                word_start = Some(idx);
            }
        } else {
            // Space character - end of word
            word_start = None;
            pass1_result.push(ch);
            continue;
        }

        // Check if we're in a word
        if let Some(start) = word_start {
            // Calculate normal character width (x) for this word
            // x = word_width / (char_count + uppercase_count)
            let word_end = {
                let mut end = idx + 1;
                while end < char_chars.len() && char_chars[end] != ' ' {
                    end += 1;
                }
                end
            };

            let word_chars_count = word_end - start;
            if word_chars_count > 0 {
                let word_width = char_rects[word_end - 1].right() - char_rects[start].left();
                let uppercase_count: usize = (start..word_end)
                    .filter(|&i| char_chars[i].is_uppercase())
                    .count();
                let normalized_char_count = word_chars_count + uppercase_count;

                let normal_char_width = if normalized_char_count > 0 {
                    word_width as f32 / normalized_char_count as f32
                } else {
                    0.0
                };

                // Only proceed if we have a valid normal character width
                if normal_char_width > 0.0 {
                    let current_width = ch.rect.width();

                    // Determine if we should insert space based on character type
                    // Compare with normal character width (x) instead of next_width
                    let should_insert = if ch.char.is_uppercase() {
                        // Uppercase: must be > 3 * x
                        current_width as f32 > 3.0 * normal_char_width
                    } else if char_types[idx] {
                        // Lowercase (including '?' treated as lowercase): must be > 2 * x
                        current_width as f32 > 2.0 * normal_char_width
                    } else {
                        // Other characters: use lowercase rule
                        current_width as f32 > 2.0 * normal_char_width
                    };

                    if should_insert && current_width > min_space_width {
                        // Split the character: keep expected width, rest becomes space
                        let expected_width = if char_types[idx] {
                            median_width
                        } else if ch.char.is_uppercase() {
                            median_width * 2
                        } else {
                            median_width
                        };

                        let char_width = expected_width.min(current_width - min_space_width);
                        let space_width = current_width - char_width;

                        if char_width > 0 && space_width >= min_space_width {
                            // Log the word before inserting space
                            if let Some(start) = word_start {
                                // Reconstruct word chars from stored info for logging
                                let word_chars: Vec<TextChar> = (start..=idx)
                                    .map(|i| TextChar {
                                        char: char_chars[i],
                                        rect: char_rects[i],
                                    })
                                    .collect();
                                //log_word(&word_chars, &line_text);
                                /*println!(
                                "[insert_missing_spaces-P1] Inserting space after '{}' (idx {}): current_width={}, normal_char_width={:.2}, char_width={}, space_width={}",
                                ch.char, idx, current_width, normal_char_width, char_width, space_width
                                );*/
                            }

                            let rect = ch.rect;
                            let adjusted_char_rect =
                                Rect::from_tlhw(rect.top(), rect.left(), rect.height(), char_width);
                            ch.rect = adjusted_char_rect;

                            let space_left = rect.left() + char_width;
                            let space_rect =
                                Rect::from_tlhw(rect.top(), space_left, rect.height(), space_width);

                            pass1_result.push(ch);
                            pass1_result.push(TextChar {
                                char: ' ',
                                rect: space_rect,
                            });
                            continue;
                        }
                    }
                }
            }
        }

        pass1_result.push(ch);
    }

    // Pass 2: Insert spaces based on gaps between consecutive characters
    // Only insert if gap is significantly large (not normal character spacing)
    let pass1_line_text: String = pass1_result.iter().map(|c| c.char).collect();
    let mut final_result = Vec::with_capacity(pass1_result.len() * 2);
    for (idx, ch) in pass1_result.iter().enumerate() {
        final_result.push(ch.clone());

        // Skip if current char is already a space
        if ch.char == ' ' {
            continue;
        }

        // Check gap to next character
        if let Some(next_ch) = pass1_result.get(idx + 1) {
            // Skip if next char is already a space
            if next_ch.char == ' ' {
                continue;
            }

            let gap = (next_ch.rect.left() - ch.rect.right()).max(0);

            // Skip if gap is too small (likely normal character spacing)
            // Require at least 5 pixels to avoid inserting in normal text
            if gap < 5 {
                continue;
            }

            // Only insert space if gap is significantly large
            // Use a higher threshold (2x median width) to avoid inserting in normal text
            let large_gap_threshold = (median_width * 2).max(10);

            // Check if the gap suggests missing space (e.g., in tables with dots)
            let is_likely_missing_space = gap > large_gap_threshold;

            // For dots/special chars, be more lenient but still require substantial gap
            let is_dot_or_special = ch.char == '.'
                || ch.char == '·'
                || ch.char == '…'
                || next_ch.char == '.'
                || next_ch.char == '·'
                || next_ch.char == '…';

            // Only insert if gap is truly large (2x median width or more)
            // OR if it's dots/special chars with gap > 1.5x median width
            if is_likely_missing_space {
                // Find word containing current character for logging
                let mut word_start = idx;
                while word_start > 0 && pass1_result[word_start - 1].char != ' ' {
                    word_start -= 1;
                }
                let mut word_end = idx + 1;
                while word_end < pass1_result.len() && pass1_result[word_end].char != ' ' {
                    word_end += 1;
                }
                if word_start < word_end {
                    let word_chars: Vec<TextChar> = pass1_result[word_start..word_end].to_vec();
                    //log_word(&word_chars, &pass1_line_text);
                    /*println!(
                        "[insert_missing_spaces-P2] Inserting space after '{}' (idx {}): gap={}, large_gap_threshold={}",
                        ch.char, idx, gap, large_gap_threshold
                    );*/
                }

                // Create a space character with reasonable width
                let space_left = ch.rect.right();
                // Use a reasonable space width (median width or gap, whichever is smaller)
                let space_width = gap.min(median_width * 3).max(min_space_width);
                let space_rect =
                    Rect::from_tlhw(ch.rect.top(), space_left, ch.rect.height(), space_width);

                final_result.push(TextChar {
                    char: ' ',
                    rect: space_rect,
                });
            } else if is_dot_or_special && gap > gap_threshold {
                // Find word containing current character for logging
                let mut word_start = idx;
                while word_start > 0 && pass1_result[word_start - 1].char != ' ' {
                    word_start -= 1;
                }
                let mut word_end = idx + 1;
                while word_end < pass1_result.len() && pass1_result[word_end].char != ' ' {
                    word_end += 1;
                }
                if word_start < word_end {
                    let word_chars: Vec<TextChar> = pass1_result[word_start..word_end].to_vec();
                    //log_word(&word_chars, &pass1_line_text);
                    /*println!(
                        "[insert_missing_spaces-P2] Inserting space for dots/special after '{}' (idx {}): gap={}, gap_threshold={}",
                        ch.char, idx, gap, gap_threshold
                    );*/
                }

                // For dots/special chars with moderate gap, insert space
                // This handles cases like "word.....word" where dots are recognized
                let space_left = ch.rect.right();
                let space_width = gap.min(median_width * 2).max(min_space_width);
                let space_rect =
                    Rect::from_tlhw(ch.rect.top(), space_left, ch.rect.height(), space_width);

                final_result.push(TextChar {
                    char: ' ',
                    rect: space_rect,
                });
            }
        }
    }

    final_result
}

/// Extracts character sequences and coordinates from text lines detected in
/// an image.
pub struct TextRecognizer {
    model: Box<dyn Model + Send + Sync>,
    input_shape: Vec<Dimension>,
}

impl TextRecognizer {
    /// Initialize a text recognizer from a trained RTen model. Fails if the
    /// model does not have the expected inputs or outputs.
    pub fn from_model(model: impl Model + Send + Sync + 'static) -> anyhow::Result<TextRecognizer> {
        let input_shape = model.input_shape()?;
        Ok(TextRecognizer {
            model: Box::new(model),
            input_shape,
        })
    }

    /// Return the expected height of input line images.
    fn input_height(&self) -> u32 {
        match self.input_shape[2] {
            Dimension::Fixed(size) => size.try_into().unwrap(),
            Dimension::Symbolic(_) => 50,
        }
    }

    /// Run text recognition on an NCHW batch of text line images, and return
    /// a `[batch, seq, label]` tensor of class probabilities.
    fn run(&self, input: NdTensor<f32, 4>) -> Result<NdTensor<f32, 3>, ModelRunError> {
        let input: Tensor<f32> = input.into();
        let output = self
            .model
            .run(input.view(), None)
            .map_err(|err| ModelRunError::RunFailed(err.into()))?;

        let output_ndim = output.ndim();
        let mut rec_sequence: NdTensor<f32, 3> = output.try_into().map_err(|_| {
            ModelRunError::WrongOutput(format!(
                "expected recognition output to have 3 dims but it has {}",
                output_ndim
            ))
        })?;

        // Transpose from [seq, batch, class] => [batch, seq, class]
        rec_sequence.permute([1, 0, 2]);

        Ok(rec_sequence)
    }

    /// Prepare a text line for input into the recognition model.
    ///
    /// This method exists for model debugging purposes to expose the
    /// preprocessing that [TextRecognizer::recognize_text_lines] does.
    pub fn prepare_input(
        &self,
        image: NdTensorView<f32, 3>,
        line: &[RotatedRect],
    ) -> NdTensor<f32, 2> {
        // These lines should match corresponding code in
        // `recognize_text_lines`.
        let [_, img_height, img_width] = image.shape();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);

        let line_rect = bounding_rect(line.iter())
            .expect("line has no words")
            .integral_bounding_rect();

        let line_poly = Polygon::new(line_polygon(line));
        let rec_img_height = self.input_height();
        let resized_width =
            resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);

        prepare_text_line(
            image,
            page_rect,
            &line_poly,
            resized_width,
            rec_img_height as usize,
        )
    }

    /// Recognize text lines in an image.
    ///
    /// `image` is a CHW greyscale image with values in the range `ZERO_VALUE` to
    /// `ZERO_VALUE + 1`. `lines` is a list of detected text lines, where each line
    /// is a sequence of word rects. `model` is a recognition model which accepts an
    /// NCHW tensor of greyscale line images and outputs a `[sequence, batch, label]`
    /// tensor of log probabilities of character classes, which must be converted to
    /// a character sequence using CTC decoding.
    ///
    /// Entries in the result can be `None` if no text was found in a line.
    pub fn recognize_text_lines(
        &self,
        image: NdTensorView<f32, 3>,
        lines: &[Vec<RotatedRect>],
        opts: RecognitionOpt,
    ) -> anyhow::Result<Vec<Option<TextLine>>> {
        let RecognitionOpt {
            debug,
            decode_method,
            alphabet,
            excluded_char_labels,
        } = opts;

        let [_, img_height, img_width] = image.shape();
        let page_rect = Rect::from_hw(img_height as i32, img_width as i32);

        // Group lines into batches which will have similar widths after resizing
        // to a fixed height.
        //
        // It is more efficient to run recognition on multiple lines at once, but
        // all line images in a batch must be padded to an equal length. Some
        // computation is wasted on shorter lines in the batch. Choosing batches
        // such that all line images have a similar width reduces this wastage.
        // There is a trade-off between maximizing the batch size and minimizing
        // the variance in width of images in the batch.
        let rec_img_height = self.input_height();
        let mut line_groups: HashMap<i32, Vec<TextRecLine>> = HashMap::new();
        for (line_index, word_rects) in lines.iter().enumerate() {
            let line_rect = bounding_rect(word_rects.iter())
                .expect("line has no words")
                .integral_bounding_rect();
            let resized_width =
                resized_line_width(line_rect.width(), line_rect.height(), rec_img_height as i32);
            let group_width = resized_width.next_multiple_of(50);
            line_groups
                .entry(group_width as i32)
                .or_default()
                .push(TextRecLine {
                    index: line_index,
                    region: Polygon::new(line_polygon(word_rects)),
                    resized_width,
                });
        }

        // Split large line groups up into smaller batches that can be processed
        // in parallel.
        let max_lines_per_group = 20;
        let line_groups: Vec<(i32, Vec<TextRecLine>)> = line_groups
            .into_iter()
            .flat_map(|(group_width, lines)| {
                lines
                    .chunks(max_lines_per_group)
                    .map(|chunk| (group_width, chunk.to_vec()))
                    .collect::<Vec<_>>()
            })
            .collect();

        let alphabet_len = alphabet.chars().count();

        // Run text recognition on batches of lines.
        let batch_rec_results: Result<Vec<Vec<LineRecResult>>, ModelRunError> =
            thread_pool().run(|| {
                line_groups
                    .into_par_iter()
                    .map(|(group_width, lines)| {
                        if debug {
                            println!(
                                "Processing group of {} lines of width {}",
                                lines.len(),
                                group_width,
                            );
                        }

                        let rec_input = prepare_text_line_batch(
                            &image,
                            &lines,
                            page_rect,
                            rec_img_height as usize,
                            group_width as usize,
                        );

                        let mut rec_output = self.run(rec_input)?;

                        if alphabet_len + 1 != rec_output.size(2) {
                            return Err(ModelRunError::WrongOutput(format!(
                                "output column count ({}) does not match alphabet size ({})",
                                rec_output.size(2),
                                alphabet_len + 1
                            )));
                        }

                        let ctc_input_len = rec_output.shape()[1];

                        // Apply CTC decoding to get the label sequence for each line.
                        let line_rec_results = lines
                            .into_iter()
                            .enumerate()
                            .map(|(group_line_index, line)| {
                                let decoder = CtcDecoder::new();

                                let mut input_seq_slice = rec_output.slice_mut([group_line_index]);
                                let input_seq = Self::filter_excluded_char_labels(
                                    excluded_char_labels,
                                    &mut input_seq_slice,
                                );

                                let ctc_output = match decode_method {
                                    DecodeMethod::Greedy => decoder.decode_greedy(input_seq),
                                    DecodeMethod::BeamSearch { width } => {
                                        decoder.decode_beam(input_seq, width)
                                    }
                                };
                                LineRecResult {
                                    line,
                                    rec_input_len: group_width as usize,
                                    ctc_input_len,
                                    ctc_output,
                                }
                            })
                            .collect::<Vec<_>>();

                        Ok(line_rec_results)
                    })
                    .collect()
            });

        let mut line_rec_results: Vec<LineRecResult> =
            batch_rec_results?.into_iter().flatten().collect();

        // The recognition outputs are in a different order than the inputs due to
        // batching and parallel processing. Re-sort them into input order.
        line_rec_results.sort_by_key(|result| result.line.index);

        Ok(text_lines_from_recognition_results(
            &line_rec_results,
            alphabet,
        ))
    }

    /// Post-process recognition model outputs to filter excluded characters.
    ///
    /// `input_seq_slice` is a (seq, char_prob) matrix of log probabilities for
    /// characters. `excluded_char_labels` specifies indices of characters that
    /// should be excluded, by setting the log probability to -Inf.
    fn filter_excluded_char_labels<'a>(
        excluded_char_labels: Option<&[usize]>,
        input_seq_slice: &'a mut NdTensorViewMut<'_, f32, 2>,
    ) -> NdTensorView<'a, f32, 2> {
        if let Some(excluded_char_labels) = excluded_char_labels {
            for row in 0..input_seq_slice.size(0) {
                for &excluded_char_label in excluded_char_labels.iter() {
                    // Setting the output value of excluded char to -Inf causes the
                    // `decode_method` to favour chars other than the excluded char.
                    (*input_seq_slice)[[row, excluded_char_label]] = f32::NEG_INFINITY;
                }
            }
        }
        input_seq_slice.view()
    }
}

