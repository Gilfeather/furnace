//! Winograd F(2x2, 3x3) convolution for `f32` on x86_64 (AVX-512 or AVX2 + FMA): 3x3 kernels,
//! stride 1, no dilation, `groups == 1`.
//!
//! Each 2x2 block of outputs (a tile) is computed from a 4x4 block of input as
//! `Y = A^T [U . V] A` with `U = G g G^T` (the filter) and `V = B^T d B` (the input), which
//! takes 16 multiplications per input channel instead of 36. Summed over input channels, the
//! element-wise products become 16 independent GEMMs, one per element `xi` of the 4x4 tile:
//! `M[xi] = U[xi] (c_out x c_in) x V[xi] (c_in x tiles)`, run on the direct convolution's
//! microkernel.
//!
//! The transformed filters `U` are computed once per weight tensor and cached
//! (see [`super::weight_cache`]).

use alloc::vec;
use alloc::vec::Vec;

use super::conv_direct::{Args, Isa, Tile, run_kernel};

/// Filters in the Winograd domain: `u[(xi * c_out + co) * c_in + ci]`
pub(crate) struct Filters {
    u: Vec<f32>,
}

impl Filters {
    /// `U = G g G^T` for every OIHW 3x3 filter in `w`
    pub(crate) fn new(w: &[f32], c_out: usize, c_in: usize) -> Self {
        let mut u = vec![0.0f32; 16 * c_out * c_in];
        for co in 0..c_out {
            for ci in 0..c_in {
                let g = &w[(co * c_in + ci) * 9..][..9];
                // G g: rows g0, (g0 + g1 + g2) / 2, (g0 - g1 + g2) / 2, g2 of the filter rows
                let mut gg = [[0.0f32; 3]; 4];
                for j in 0..3 {
                    let (a, b, c) = (g[j], g[3 + j], g[6 + j]);
                    gg[0][j] = a;
                    gg[1][j] = 0.5 * (a + b + c);
                    gg[2][j] = 0.5 * (a - b + c);
                    gg[3][j] = c;
                }
                // (G g) G^T: the same combination over columns
                for (i, &[a, b, c]) in gg.iter().enumerate() {
                    let row = [a, 0.5 * (a + b + c), 0.5 * (a - b + c), c];
                    for (j, &value) in row.iter().enumerate() {
                        u[((i * 4 + j) * c_out + co) * c_in + ci] = value;
                    }
                }
            }
        }
        Self { u }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.u.len() * 4
    }
}

/// `f32` 3x3 stride-1 convolution, or `None` when this CPU has neither AVX-512 nor AVX2 + FMA.
/// The output is NCHW and includes the bias.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_winograd_f32(
    x: &[f32],
    filters: &Filters,
    bias: Option<&[f32]>,
    [batch, c_in, in_h, in_w]: [usize; 4],
    c_out: usize,
    [out_h, out_w]: [usize; 2],
    [p_h, p_w]: [usize; 2],
    threads: usize,
) -> Option<Vec<f32>> {
    let isa = Isa::detect()?;
    Some(conv2d_winograd_with(
        isa,
        x,
        filters,
        bias,
        [batch, c_in, in_h, in_w],
        c_out,
        [out_h, out_w],
        [p_h, p_w],
        threads,
    ))
}

#[allow(clippy::too_many_arguments)]
fn conv2d_winograd_with(
    isa: Isa,
    x: &[f32],
    filters: &Filters,
    bias: Option<&[f32]>,
    [batch, c_in, in_h, in_w]: [usize; 4],
    c_out: usize,
    [out_h, out_w]: [usize; 2],
    [p_h, p_w]: [usize; 2],
    threads: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * c_out * out_h * out_w];
    if output.is_empty() || c_in == 0 {
        if let Some(bias) = bias
            && !output.is_empty()
        {
            for (plane, &b) in output.chunks_mut(out_h * out_w).zip(bias.iter().cycle()) {
                plane.fill(b);
            }
        }
        return output;
    }

    let (tiles_h, tiles_w) = (out_h.div_ceil(2), out_w.div_ceil(2));
    let tiles = tiles_h * tiles_w;
    let tile = Tile::choose(isa, tiles);
    let nb = tile.nb();
    // Tile columns padded to whole microkernel blocks; the extra columns are zero
    let tiles_pad = tiles.div_ceil(nb) * nb;

    // V[b][xi][ci][t], computed one (image, input channel) at a time
    let image_v = 16 * c_in * tiles_pad;
    // Columns past the last tile are never written: they only feed discarded lanes
    let mut v = super::scratch::Scratch::take(batch * image_v);
    let geometry = InputGeometry {
        in_h,
        in_w,
        p_h,
        p_w,
        tiles_h,
        tiles_w,
        c_in,
        tiles_pad,
    };
    let transform = |bc: usize, v_ptr: *mut f32| {
        let (b, ci) = (bc / c_in, bc % c_in);
        let src = &x[bc * in_h * in_w..][..in_h * in_w];
        // SAFETY: each (image, channel) writes its own rows of V
        unsafe { input_transform(isa, &geometry, src, v_ptr.add(b * image_v), ci) };
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let v_ptr = crate::ops::SendMutPtr::new(v.as_mut_ptr());
        (0..batch * c_in)
            .into_par_iter()
            .for_each(|bc| transform(bc, unsafe { v_ptr.ptr_add(0) }));
    }
    #[cfg(not(feature = "rayon"))]
    for bc in 0..batch * c_in {
        transform(bc, v.as_mut_ptr());
    }

    // Tasks: (image, range of output channel blocks, range of tile blocks). Within a task the
    // tile blocks are the outer loop, so one block of V (16 x c_in x nb) stays in cache while
    // every channel block of the task multiplies it; channel ranges are sized to keep their
    // filters (16 x channels x c_in) within 256 KiB.
    let mr = tile.mr;
    let mr_blocks = c_out.div_ceil(mr);
    let tile_blocks = tiles_pad / nb;
    let co_per_task = (256 * 1024 / (16 * mr * c_in * 4)).clamp(1, mr_blocks);
    let co_chunks = mr_blocks.div_ceil(co_per_task);
    let tile_chunks = (4 * threads.max(1))
        .div_ceil(batch * co_chunks)
        .clamp(1, tile_blocks);
    let tiles_per_task = tile_blocks.div_ceil(tile_chunks);
    let tile_chunks = tile_blocks.div_ceil(tiles_per_task);
    let tasks = batch * co_chunks * tile_chunks;
    let out_hw = out_h * out_w;

    let run_task = |task: usize, out: *mut f32| {
        let b = task / (co_chunks * tile_chunks);
        let co_chunk = task / tile_chunks % co_chunks;
        let tile_chunk = task % tile_chunks;
        let v_image = &v[b * image_v..][..image_v];
        // M[xi][row][t] for one block of output channels and tiles
        let mut m = vec![0.0f32; 16 * mr * nb];
        let first_tile = tile_chunk * tiles_per_task;
        let first_block = co_chunk * co_per_task;
        for tb in first_tile..(first_tile + tiles_per_task).min(tile_blocks) {
            let t0 = tb * nb;
            for block in first_block..(first_block + co_per_task).min(mr_blocks) {
                let co = block * mr;
                let rows = mr.min(c_out - co);
                let m_ptr = m.as_mut_ptr();
                for xi in 0..16 {
                    let args = |row: usize| Args {
                        b: unsafe { v_image.as_ptr().add(xi * c_in * tiles_pad + t0) },
                        w: unsafe { filters.u.as_ptr().add((xi * c_out + co + row) * c_in) },
                        k_total: c_in,
                        c_in,
                        cs: tiles_pad,
                        taps: &[0],
                        out: unsafe { m_ptr.add((xi * mr + row) * nb) },
                    };
                    // SAFETY: V rows have `tiles_pad` columns, a whole number of blocks
                    unsafe {
                        if rows == mr {
                            run_kernel(isa, mr, tile.nv, args(0));
                        } else {
                            for row in 0..rows {
                                run_kernel(isa, 1, tile.nv, args(row));
                            }
                        }
                    }
                }
                // SAFETY: tasks write disjoint (channel, tile) outputs of their own image
                unsafe {
                    output_transform(
                        isa,
                        &m,
                        [mr, nb],
                        rows,
                        t0,
                        [tiles, tiles_w],
                        [out_h, out_w],
                        bias.map(|bias| &bias[co..co + rows]),
                        out.add((b * c_out + co) * out_hw),
                    );
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let out = crate::ops::SendMutPtr::new(output.as_mut_ptr());
        (0..tasks)
            .into_par_iter()
            .for_each(|task| run_task(task, unsafe { out.ptr_add(0) }));
    }
    #[cfg(not(feature = "rayon"))]
    for task in 0..tasks {
        run_task(task, output.as_mut_ptr());
    }
    output
}

struct InputGeometry {
    in_h: usize,
    in_w: usize,
    p_h: usize,
    p_w: usize,
    tiles_h: usize,
    tiles_w: usize,
    c_in: usize,
    tiles_pad: usize,
}

/// Write `V = B^T d B` for every tile of input channel `ci` (plane `src`) into
/// `v[xi][ci][t]`.
///
/// # Safety
/// The CPU must support `isa`; `v` must hold `16 * c_in * tiles_pad` elements.
unsafe fn input_transform(isa: Isa, g: &InputGeometry, src: &[f32], v: *mut f32, ci: usize) {
    unsafe {
        match isa {
            Isa::Avx512 => input_transform_avx512(g, src, v, ci),
            Isa::Avx2 => input_transform_avx2(g, src, v, ci),
        }
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn input_transform_avx512(g: &InputGeometry, src: &[f32], v: *mut f32, ci: usize) {
    unsafe { input_transform_body(g, src, v, ci) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn input_transform_avx2(g: &InputGeometry, src: &[f32], v: *mut f32, ci: usize) {
    unsafe { input_transform_body(g, src, v, ci) }
}

/// Inlined into each target-feature wrapper so the loops over tiles vectorize for it.
#[inline(always)]
unsafe fn input_transform_body(g: &InputGeometry, src: &[f32], v: *mut f32, ci: usize) {
    let (th, tw) = (g.tiles_h, g.tiles_w);
    // Zero-padded input split into even and odd columns: tile (ty, tx) reads rows
    // 2ty..2ty+4, and in each of them even[tx], odd[tx], even[tx + 1], odd[tx + 1]
    let (hp, wh) = (2 * th + 2, tw + 1);
    let mut even = vec![0.0f32; hp * wh];
    let mut odd = vec![0.0f32; hp * wh];
    for r in 0..hp {
        let ih = r as isize - g.p_h as isize;
        if ih < 0 || ih >= g.in_h as isize {
            continue;
        }
        let src_row = &src[ih as usize * g.in_w..][..g.in_w];
        let (e, o) = (&mut even[r * wh..][..wh], &mut odd[r * wh..][..wh]);
        for (iw, &value) in src_row.iter().enumerate() {
            let col = iw + g.p_w;
            if col < 2 * wh {
                if col.is_multiple_of(2) {
                    e[col / 2] = value;
                } else {
                    o[col / 2] = value;
                }
            }
        }
    }

    // d B per input row of a tile row: db[i * 4 + j][tx]
    let mut db = vec![0.0f32; 16 * tw];
    for ty in 0..th {
        for i in 0..4 {
            let r = 2 * ty + i;
            let e = &even[r * wh..][..wh];
            let o = &odd[r * wh..][..wh];
            let (e0, e1, o0, o1) = (&e[..tw], &e[1..], &o[..tw], &o[1..]);
            let (d0, rest) = db[i * 4 * tw..][..4 * tw].split_at_mut(tw);
            let (d1, rest) = rest.split_at_mut(tw);
            let (d2, d3) = rest.split_at_mut(tw);
            // Columns d0 - d2, d1 + d2, d2 - d1, d1 - d3 with d = (e0, o0, e1, o1)
            for t in 0..tw {
                d0[t] = e0[t] - e1[t];
                d1[t] = o0[t] + e1[t];
                d2[t] = e1[t] - o0[t];
                d3[t] = o0[t] - o1[t];
            }
        }
        // B^T (d B): rows r0 - r2, r1 + r2, r2 - r1, r1 - r3
        let t0 = ty * tw;
        for j in 0..4 {
            let row = |i: usize| &db[(i * 4 + j) * tw..][..tw];
            let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
            let dst = |i: usize| unsafe {
                core::slice::from_raw_parts_mut(
                    v.add(((i * 4 + j) * g.c_in + ci) * g.tiles_pad + t0),
                    tw,
                )
            };
            let (v0, v1, v2, v3) = (dst(0), dst(1), dst(2), dst(3));
            for t in 0..tw {
                v0[t] = r0[t] - r2[t];
                v1[t] = r1[t] + r2[t];
                v2[t] = r2[t] - r1[t];
                v3[t] = r1[t] - r3[t];
            }
        }
    }
}

/// `Y = A^T M A` for `rows` output channels and the `nb` tiles from `t0`, plus bias, written
/// into the NCHW planes at `out` (the first channel's plane).
///
/// # Safety
/// The CPU must support `isa`; `out` must hold `rows` planes of `out_h * out_w`.
#[allow(clippy::too_many_arguments)]
unsafe fn output_transform(
    isa: Isa,
    m: &[f32],
    dims: [usize; 2],
    rows: usize,
    t0: usize,
    tiles: [usize; 2],
    out_size: [usize; 2],
    bias: Option<&[f32]>,
    out: *mut f32,
) {
    unsafe {
        match isa {
            Isa::Avx512 => output_transform_avx512(m, dims, rows, t0, tiles, out_size, bias, out),
            Isa::Avx2 => output_transform_avx2(m, dims, rows, t0, tiles, out_size, bias, out),
        }
    }
}

#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn output_transform_avx512(
    m: &[f32],
    dims: [usize; 2],
    rows: usize,
    t0: usize,
    tiles: [usize; 2],
    out_size: [usize; 2],
    bias: Option<&[f32]>,
    out: *mut f32,
) {
    unsafe { output_transform_body(m, dims, rows, t0, tiles, out_size, bias, out) }
}

#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn output_transform_avx2(
    m: &[f32],
    dims: [usize; 2],
    rows: usize,
    t0: usize,
    tiles: [usize; 2],
    out_size: [usize; 2],
    bias: Option<&[f32]>,
    out: *mut f32,
) {
    unsafe { output_transform_body(m, dims, rows, t0, tiles, out_size, bias, out) }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn output_transform_body(
    m: &[f32],
    [mr, nb]: [usize; 2],
    rows: usize,
    t0: usize,
    [tiles, tiles_w]: [usize; 2],
    [out_h, out_w]: [usize; 2],
    bias: Option<&[f32]>,
    out: *mut f32,
) {
    const MAX_NB: usize = 64;
    let count = nb.min(tiles.saturating_sub(t0));
    debug_assert!(nb <= MAX_NB);
    // Tiles whose two output columns are both inside the output
    let full_w = out_w / 2;
    let mut y = [[0.0f32; MAX_NB]; 4];
    for row in 0..rows {
        let add = bias.map_or(0.0, |bias| bias[row]);
        let p: [&[f32]; 16] = core::array::from_fn(|xi| &m[(xi * mr + row) * nb..][..nb]);
        // A^T M A: rows m0 + m1 + m2 and m1 - m2 - m3, then the same over columns
        for t in 0..count {
            let r0: [f32; 4] = core::array::from_fn(|j| p[j][t] + p[4 + j][t] + p[8 + j][t]);
            let r1: [f32; 4] = core::array::from_fn(|j| p[4 + j][t] - p[8 + j][t] - p[12 + j][t]);
            y[0][t] = r0[0] + r0[1] + r0[2] + add;
            y[1][t] = r0[1] - r0[2] - r0[3] + add;
            y[2][t] = r1[0] + r1[1] + r1[2] + add;
            y[3][t] = r1[1] - r1[2] - r1[3] + add;
        }

        let plane =
            unsafe { core::slice::from_raw_parts_mut(out.add(row * out_h * out_w), out_h * out_w) };
        // Walk the block one tile row at a time
        let mut t = 0;
        while t < count {
            let tile = t0 + t;
            let (ty, tx0) = (tile / tiles_w, tile % tiles_w);
            let run = (tiles_w - tx0).min(count - t);
            for (dy, (ya, yb)) in [(0, (&y[0], &y[1])), (1, (&y[2], &y[3]))] {
                let oy = 2 * ty + dy;
                if oy >= out_h {
                    continue;
                }
                let out_row = &mut plane[oy * out_w..][..out_w];
                let full = run.min(full_w.saturating_sub(tx0));
                for k in 0..full {
                    let ox = 2 * (tx0 + k);
                    out_row[ox] = ya[t + k];
                    out_row[ox + 1] = yb[t + k];
                }
                // A last tile column with only its left output inside
                if full < run {
                    out_row[2 * (tx0 + full)] = ya[t + full];
                }
            }
            t += run;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        [batch, c_in, in_h, in_w]: [usize; 4],
        c_out: usize,
        [out_h, out_w]: [usize; 2],
        [p_h, p_w]: [usize; 2],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * c_out * out_h * out_w];
        for b in 0..batch {
            for co in 0..c_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias.map_or(0.0, |bias| bias[co]);
                        for ci in 0..c_in {
                            for r in 0..3 {
                                for c in 0..3 {
                                    let ih = (oh + r) as isize - p_h as isize;
                                    let iw = (ow + c) as isize - p_w as isize;
                                    if ih >= 0
                                        && iw >= 0
                                        && (ih as usize) < in_h
                                        && (iw as usize) < in_w
                                    {
                                        sum += x[((b * c_in + ci) * in_h + ih as usize) * in_w
                                            + iw as usize]
                                            * w[((co * c_in + ci) * 3 + r) * 3 + c];
                                    }
                                }
                            }
                        }
                        out[((b * c_out + co) * out_h + oh) * out_w + ow] = sum;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn matches_reference() {
        let mut isas = Vec::new();
        if std::is_x86_feature_detected!("avx512f") {
            isas.push(Isa::Avx512);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            isas.push(Isa::Avx2);
        }
        // (batch, c_in, in_h, in_w, c_out, pad, bias)
        let cases = [
            (1, 4, 8, 8, 8, 1, true),
            (2, 3, 7, 9, 5, 1, false),
            (1, 16, 7, 7, 17, 1, true),
            (1, 8, 14, 14, 24, 1, true),
            (1, 5, 6, 5, 3, 0, true),
            (1, 2, 5, 11, 9, 2, false),
            (1, 64, 56, 56, 16, 1, true),
        ];
        for (batch, c_in, in_h, in_w, c_out, p, with_bias) in cases {
            let out_h = in_h + 2 * p - 2;
            let out_w = in_w + 2 * p - 2;
            let x = values(batch * c_in * in_h * in_w, 1);
            let w = values(c_out * c_in * 9, 2);
            let bias = values(c_out, 3);
            let bias = with_bias.then_some(bias.as_slice());
            let expected = reference(
                &x,
                &w,
                bias,
                [batch, c_in, in_h, in_w],
                c_out,
                [out_h, out_w],
                [p, p],
            );
            let filters = Filters::new(&w, c_out, c_in);
            for &isa in &isas {
                for threads in [1, 5] {
                    let got = conv2d_winograd_with(
                        isa,
                        &x,
                        &filters,
                        bias,
                        [batch, c_in, in_h, in_w],
                        c_out,
                        [out_h, out_w],
                        [p, p],
                        threads,
                    );
                    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                        assert!(
                            (g - e).abs() <= 1e-4 * (1.0 + e.abs()),
                            "case {:?} {} threads {threads}: [{i}] {g} vs {e}",
                            (batch, c_in, in_h, in_w, c_out, p),
                            isa.name()
                        );
                    }
                }
            }
        }
    }

    /// `cargo test --release --lib winograd::tests::throughput -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn throughput() {
        let shapes = [
            ("layer1 3x3 64", 64, 56, 64),
            ("layer2 3x3 128", 128, 28, 128),
            ("layer3 3x3 256", 256, 14, 256),
            ("layer4 3x3 512", 512, 7, 512),
        ];
        let Some(isa) = Isa::detect() else { return };
        for (name, c, size, c_out) in shapes {
            let x = values(c * size * size, 1);
            let w = values(c_out * c * 9, 2);
            let filters = Filters::new(&w, c_out, c);
            let run = || {
                conv2d_winograd_with(
                    isa,
                    &x,
                    &filters,
                    None,
                    [1, c, size, size],
                    c_out,
                    [size, size],
                    [1, 1],
                    rayon::current_num_threads(),
                )
            };
            for _ in 0..3 {
                core::hint::black_box(run());
            }
            let flops = 2.0 * (c_out * size * size * c * 9) as f64;
            let iters = ((2e9 / flops) as usize).clamp(3, 2000);
            let start = std::time::Instant::now();
            for _ in 0..iters {
                core::hint::black_box(run());
            }
            let secs = start.elapsed().as_secs_f64() / iters as f64;
            std::println!(
                "{} {name:16} {:8.3} ms {:6.1} effective GFLOP/s",
                isa.name(),
                secs * 1e3,
                flops / secs / 1e9
            );
        }
    }
}
