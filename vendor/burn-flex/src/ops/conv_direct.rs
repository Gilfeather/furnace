//! Direct 2D convolution for `f32` on x86_64 (AVX-512 or AVX2 + FMA), without im2col.
//!
//! The input is copied once into a zero-padded buffer. With stride `s > 1` the buffer is split
//! into phase planes by row and column remainder (`s_h * s_w` of them at most), which turns the
//! convolution into a stride-1 one over the phases. Output position `p = oh * wq + ow` of an
//! image then reads kernel tap `(ci, kh, kw)` at a fixed offset plus `p`, so each GEMM column is
//! a plain vector load from the padded buffer: an implicit GEMM with no im2col copy. Columns with
//! `ow >= out_w` fall on the padding and are computed and dropped, which costs
//! `(k_w - 1) / s_w` columns per output row.
//!
//! The microkernel keeps an `MR x (NV * lanes)` block of outputs in registers and broadcasts
//! weights from their OIHW layout, so the weights need no repacking either. Bias is added when
//! the block is written out.

use alloc::vec;
use alloc::vec::Vec;

/// Largest stride handled here; larger ones use the im2col path.
const MAX_STRIDE: usize = 4;

#[derive(Clone, Copy)]
pub(super) enum Isa {
    Avx512,
    Avx2,
}

impl Isa {
    pub(super) fn detect() -> Option<Self> {
        if std::is_x86_feature_detected!("avx512f") {
            Some(Self::Avx512)
        } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            Some(Self::Avx2)
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Avx512 => "avx512",
            Self::Avx2 => "avx2",
        }
    }

    /// `f32` lanes per vector register
    pub(super) fn lanes(self) -> usize {
        match self {
            Self::Avx512 => 16,
            Self::Avx2 => 8,
        }
    }

    /// Microkernel shapes as (output channels, vectors of positions), widest first. Each keeps
    /// its accumulators, input vectors and broadcast within the vector registers.
    fn shapes(self) -> &'static [(usize, usize)] {
        match self {
            Self::Avx512 => &[(8, 3), (12, 2), (16, 1)],
            Self::Avx2 => &[(6, 2), (12, 1)],
        }
    }
}

/// Microkernel shape chosen for one convolution
#[derive(Clone, Copy)]
pub(super) struct Tile {
    pub(super) isa: Isa,
    /// Output channels per block
    pub(super) mr: usize,
    /// Position vectors per block
    pub(super) nv: usize,
}

impl Tile {
    /// Output positions per block
    pub(super) fn nb(self) -> usize {
        self.nv * self.isa.lanes()
    }

    /// The shape that wastes the fewest lanes on the last, partial position block of an
    /// image with `positions` outputs, preferring wider blocks on ties.
    pub(super) fn choose(isa: Isa, positions: usize) -> Self {
        let mut best = None;
        for &(mr, nv) in isa.shapes() {
            let nb = nv * isa.lanes();
            let computed = positions.div_ceil(nb) * nb;
            if best.is_none_or(|(_, c)| computed < c) {
                best = Some(((mr, nv), computed));
            }
        }
        let ((mr, nv), _) = best.expect("at least one shape");
        Self { isa, mr, nv }
    }
}

/// Geometry of the padded, phase-split input of one convolution.
pub(super) struct Geometry {
    pub(super) c_in: usize,
    pub(super) c_out: usize,
    pub(super) out_h: usize,
    pub(super) out_w: usize,
    /// Rows and columns of each phase plane
    pub(super) hq: usize,
    pub(super) wq: usize,
    /// Phase planes per input channel
    pub(super) phases: usize,
    /// Offset of each kernel tap `(kh, kw)` within one input channel's planes
    pub(super) taps: Vec<usize>,
}

impl Geometry {
    pub(super) fn plane(&self) -> usize {
        self.hq * self.wq
    }

    pub(super) fn channel_stride(&self) -> usize {
        self.phases * self.plane()
    }

    pub(super) fn image_len(&self) -> usize {
        self.c_in * self.channel_stride()
    }

    /// Output positions of one image, including the dropped padding columns
    pub(super) fn positions(&self) -> usize {
        self.out_h * self.wq
    }
}

/// `f32` 2D convolution with `groups == 1` and no dilation, or `None` when this CPU has neither
/// AVX-512 nor AVX2 + FMA or the stride is larger than [`MAX_STRIDE`]. The output is NCHW and
/// includes the bias.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_direct_f32(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    [batch, c_in, in_h, in_w]: [usize; 4],
    [c_out, k_h, k_w]: [usize; 3],
    [out_h, out_w]: [usize; 2],
    [s_h, s_w]: [usize; 2],
    [p_h, p_w]: [usize; 2],
    threads: usize,
) -> Option<Vec<f32>> {
    if s_h == 0 || s_w == 0 || s_h > MAX_STRIDE || s_w > MAX_STRIDE {
        return None;
    }
    let isa = Isa::detect()?;
    Some(conv2d_direct_with(
        isa,
        x,
        w,
        bias,
        [batch, c_in, in_h, in_w],
        [c_out, k_h, k_w],
        [out_h, out_w],
        [s_h, s_w],
        [p_h, p_w],
        threads,
        None,
    ))
}

/// [`conv2d_direct_f32`] with a given instruction set, which the CPU must support, and
/// optionally a fixed microkernel shape (output channels, position vectors) from
/// [`Isa::shapes`].
#[allow(clippy::too_many_arguments)]
fn conv2d_direct_with(
    isa: Isa,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    [batch, c_in, in_h, in_w]: [usize; 4],
    [c_out, k_h, k_w]: [usize; 3],
    [out_h, out_w]: [usize; 2],
    [s_h, s_w]: [usize; 2],
    [p_h, p_w]: [usize; 2],
    threads: usize,
    shape: Option<(usize, usize)>,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * c_out * out_h * out_w];
    if output.is_empty() {
        return output;
    }
    if c_in == 0 || k_h == 0 || k_w == 0 {
        if let Some(bias) = bias {
            for (plane, &b) in output.chunks_mut(out_h * out_w).zip(bias.iter().cycle()) {
                plane.fill(b);
            }
        }
        return output;
    }

    let tile = shape.map_or_else(
        || {
            let wq = out_w + (k_w - 1) / s_w;
            Tile::choose(isa, out_h * wq)
        },
        |(mr, nv)| Tile { isa, mr, nv },
    );
    let (geo, padded) = prepare_input(
        x,
        [batch, c_in, in_h, in_w],
        [c_out, k_h, k_w],
        [out_h, out_w],
        [s_h, s_w],
        [p_h, p_w],
        tile.nb(),
    );
    let image_len = geo.image_len();

    // Tasks: (image, block of output channels, range of position blocks)
    let k_total = c_in * k_h * k_w;
    let mr = tile.mr;
    let nb = tile.nb();
    let mr_blocks = c_out.div_ceil(mr);
    // Output channels per task: as many MR blocks as keep their weights within 32 KiB
    let co_blocks = (32 * 1024 / (mr * k_total * 4)).clamp(1, mr_blocks);
    let co_chunks = mr_blocks.div_ceil(co_blocks);
    let pos_blocks = geo.positions().div_ceil(nb);
    let units = batch * co_chunks * pos_blocks;
    let pos_per_task = (units / (4 * threads.max(1))).clamp(1, pos_blocks);
    let pos_chunks = pos_blocks.div_ceil(pos_per_task);
    let tasks = batch * co_chunks * pos_chunks;

    let run_task = |task: usize, out: *mut f32| {
        let b = task / (co_chunks * pos_chunks);
        let co_chunk = task / pos_chunks % co_chunks;
        let pos_chunk = task % pos_chunks;
        let image = &padded[b * image_len..];
        let out_image = unsafe { out.add(b * c_out * out_h * out_w) };
        let blocks = co_chunk * co_blocks..((co_chunk + 1) * co_blocks).min(mr_blocks);
        let positions = pos_chunk * pos_per_task..((pos_chunk + 1) * pos_per_task).min(pos_blocks);
        for block in blocks {
            let co = block * mr;
            for pb in positions.clone() {
                // SAFETY: the padded buffer has SLACK elements past the last image, enough for
                // the widest block starting at any position; tasks write disjoint outputs.
                unsafe {
                    compute_block(tile, &geo, image, w, k_total, bias, co, pb * nb, out_image);
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

/// Copy every image into a zero-padded, phase-split buffer (see the module docs): per image,
/// `c_in` channels of `phases` planes of `hq x wq`, followed by `slack + wq` elements that only
/// feed discarded lanes (a block of up to `slack` positions starting near the end of the last
/// plane).
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_input(
    x: &[f32],
    [batch, c_in, in_h, in_w]: [usize; 4],
    [c_out, k_h, k_w]: [usize; 3],
    [out_h, out_w]: [usize; 2],
    [s_h, s_w]: [usize; 2],
    [p_h, p_w]: [usize; 2],
    slack: usize,
) -> (Geometry, super::scratch::Scratch) {
    // Phase planes the kernel taps actually read: (row remainder, column remainder)
    let mut phase_of = [[usize::MAX; MAX_STRIDE]; MAX_STRIDE];
    let mut phase_list = Vec::new();
    for (a, row) in phase_of.iter_mut().enumerate().take(k_h.min(s_h)) {
        for (c, slot) in row.iter_mut().enumerate().take(k_w.min(s_w)) {
            *slot = phase_list.len();
            phase_list.push((a, c));
        }
    }
    let hq = out_h + (k_h - 1) / s_h;
    let wq = out_w + (k_w - 1) / s_w;
    let plane = hq * wq;
    let mut taps = Vec::with_capacity(k_h * k_w);
    for r in 0..k_h {
        for c in 0..k_w {
            taps.push(phase_of[r % s_h][c % s_w] * plane + (r / s_h) * wq + c / s_w);
        }
    }
    let geo = Geometry {
        c_in,
        c_out,
        out_h,
        out_w,
        hq,
        wq,
        phases: phase_list.len(),
        taps,
    };

    let image_len = geo.image_len();
    // Every plane is written in full; the slack only feeds discarded lanes
    let mut padded = super::scratch::Scratch::take(batch * image_len + slack + wq);
    let fill = |(bc, dst): (usize, &mut [f32])| {
        let src = &x[bc * in_h * in_w..][..in_h * in_w];
        for (&(a, c), dst) in phase_list.iter().zip(dst.chunks_mut(plane)) {
            fill_phase(
                dst,
                src,
                [in_h, in_w],
                [hq, wq],
                [a, c],
                [s_h, s_w],
                [p_h, p_w],
            );
        }
    };
    let channels = &mut padded[..batch * image_len];
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        channels
            .par_chunks_mut(geo.channel_stride())
            .enumerate()
            .for_each(fill);
    }
    #[cfg(not(feature = "rayon"))]
    channels
        .chunks_mut(geo.channel_stride())
        .enumerate()
        .for_each(fill);
    (geo, padded)
}

/// Copy phase `(a, c)` of one zero-padded input channel into `dst` (`hq x wq`): element
/// `[i][j]` is input `[s_h * i + a - p_h][s_w * j + c - p_w]`, or zero outside the input.
#[allow(clippy::too_many_arguments)]
fn fill_phase(
    dst: &mut [f32],
    src: &[f32],
    [in_h, in_w]: [usize; 2],
    [hq, wq]: [usize; 2],
    [a, c]: [usize; 2],
    [s_h, s_w]: [usize; 2],
    [p_h, p_w]: [usize; 2],
) {
    for (i, row) in dst.chunks_mut(wq).take(hq).enumerate() {
        let ih = (s_h * i + a) as isize - p_h as isize;
        if ih < 0 || ih >= in_h as isize {
            row.fill(0.0);
            continue;
        }
        let src_row = &src[ih as usize * in_w..][..in_w];
        // Columns [lo, hi) map onto the input: 0 <= s_w * j + first < in_w
        let first = c as isize - p_w as isize;
        let lo = if first >= 0 {
            0
        } else {
            ((-first) as usize).div_ceil(s_w)
        }
        .min(wq);
        let last = in_w as isize - 1 - first;
        let hi = if last < 0 {
            lo
        } else {
            (last as usize / s_w + 1).clamp(lo, wq)
        };
        row[..lo].fill(0.0);
        if s_w == 1 {
            let start = (lo as isize + first) as usize;
            row[lo..hi].copy_from_slice(&src_row[start..start + (hi - lo)]);
        } else {
            for (j, value) in row[lo..hi].iter_mut().enumerate() {
                *value = src_row[((s_w * (lo + j)) as isize + first) as usize];
            }
        }
        row[hi..].fill(0.0);
    }
}

/// Compute output channels `co..co + mr` (fewer at the end) for positions `p0..p0 + nb` of one
/// image and write them, plus bias, into `out_image` (NCHW, dropping padding columns).
///
/// # Safety
/// `image` must have at least `p0 + nb + max tap offset` readable elements past each channel
/// start, and no other thread may write the same outputs.
#[allow(clippy::too_many_arguments)]
unsafe fn compute_block(
    shape: Tile,
    geo: &Geometry,
    image: &[f32],
    w: &[f32],
    k_total: usize,
    bias: Option<&[f32]>,
    co: usize,
    p0: usize,
    out_image: *mut f32,
) {
    const MAX_TILE: usize = 384;
    let mut tile = [0.0f32; MAX_TILE];
    let nb = shape.nb();
    debug_assert!(shape.mr * nb <= MAX_TILE);
    let mr = shape.mr.min(geo.c_out - co);
    let b = unsafe { image.as_ptr().add(p0) };
    let cs = geo.channel_stride();
    let out = tile.as_mut_ptr();
    let args = |row: usize| Args {
        b,
        w: unsafe { w.as_ptr().add((co + row) * k_total) },
        k_total,
        c_in: geo.c_in,
        cs,
        taps: &geo.taps,
        out: unsafe { out.add(row * nb) },
    };
    unsafe {
        if mr == shape.mr {
            run_kernel(shape.isa, mr, shape.nv, args(0));
        } else {
            // Last, partial block of output channels: one row at a time
            for row in 0..mr {
                run_kernel(shape.isa, 1, shape.nv, args(row));
            }
        }
    }

    // Scatter the valid positions: p = oh * wq + ow with ow < out_w and oh < out_h
    let out_hw = geo.out_h * geo.out_w;
    let end = (p0 + nb).min(geo.positions());
    for r in 0..mr {
        let row = &tile[r * nb..][..nb];
        let add = bias.map_or(0.0, |bias| bias[co + r]);
        let out_plane = unsafe { out_image.add((co + r) * out_hw) };
        let mut p = p0;
        while p < end {
            let (oh, ow) = (p / geo.wq, p % geo.wq);
            let run_end = (p - ow + geo.wq).min(end);
            if ow < geo.out_w {
                let valid = (run_end - p).min(geo.out_w - ow);
                let dst = unsafe { out_plane.add(oh * geo.out_w + ow) };
                for (i, &v) in row[p - p0..][..valid].iter().enumerate() {
                    unsafe { dst.add(i).write(v + add) };
                }
            }
            p = run_end;
        }
    }
}

/// Pointers and sizes for one microkernel call
pub(super) struct Args<'a> {
    /// Padded input at the block's first position
    pub(super) b: *const f32,
    /// OIHW weights of the block's first output channel
    pub(super) w: *const f32,
    pub(super) k_total: usize,
    pub(super) c_in: usize,
    /// Distance between input channels in the padded buffer
    pub(super) cs: usize,
    pub(super) taps: &'a [usize],
    /// `mr x nb` output tile, row-major
    pub(super) out: *mut f32,
}

/// Run the `mr x nv` microkernel of `isa`.
///
/// # Safety
/// The CPU must support `isa`, `(mr, nv)` must be one of its shapes or `(1, nv)` for one of
/// them, and every access must be in bounds.
pub(super) unsafe fn run_kernel(isa: Isa, mr: usize, nv: usize, a: Args) {
    macro_rules! call {
        ($m:ident, $($mr:literal x $nv:literal),*) => {
            match (mr, nv) {
                $(($mr, $nv) => unsafe {
                    $m::kernel::<$mr, $nv>(a.b, a.w, a.k_total, a.c_in, a.cs, a.taps, a.out)
                },)*
                _ => unreachable!("no {mr}x{nv} microkernel"),
            }
        };
    }
    match isa {
        Isa::Avx512 => call!(avx512, 8 x 3, 12 x 2, 16 x 1, 1 x 3, 1 x 2, 1 x 1),
        Isa::Avx2 => call!(avx2, 6 x 2, 12 x 1, 1 x 2, 1 x 1),
    }
}

macro_rules! microkernel {
    ($module:ident, $feature:literal, $lanes:literal, $zero:ident, $load:ident, $store:ident, $set1:ident, $fma:ident) => {
        mod $module {
            use core::arch::x86_64::*;

            /// `out[r][v * lanes..]` = sum over `ci` and taps of `w[r * k_total + k]` times the
            /// input vectors at `b + ci * cs + tap + v * lanes`, for `MR` rows (row stride
            /// `NV * lanes`).
            ///
            /// # Safety
            /// Requires the target feature; every load and store must be in bounds.
            #[target_feature(enable = $feature)]
            #[allow(clippy::too_many_arguments)]
            pub(super) unsafe fn kernel<const MR: usize, const NV: usize>(
                b: *const f32,
                w: *const f32,
                k_total: usize,
                c_in: usize,
                cs: usize,
                taps: &[usize],
                out: *mut f32,
            ) {
                unsafe {
                    let mut acc = [[$zero(); NV]; MR];
                    let mut k = 0;
                    for ci in 0..c_in {
                        let bc = b.add(ci * cs);
                        for &tap in taps {
                            let bp = bc.add(tap);
                            let mut bv = [$zero(); NV];
                            for v in 0..NV {
                                bv[v] = $load(bp.add(v * $lanes));
                            }
                            for r in 0..MR {
                                let wr = $set1(*w.add(r * k_total + k));
                                for v in 0..NV {
                                    acc[r][v] = $fma(wr, bv[v], acc[r][v]);
                                }
                            }
                            k += 1;
                        }
                    }
                    for r in 0..MR {
                        for v in 0..NV {
                            $store(out.add((r * NV + v) * $lanes), acc[r][v]);
                        }
                    }
                }
            }
        }
    };
}

microkernel!(
    avx512,
    "avx512f",
    16,
    _mm512_setzero_ps,
    _mm512_loadu_ps,
    _mm512_storeu_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps
);
microkernel!(
    avx2,
    "avx2,fma",
    8,
    _mm256_setzero_ps,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Plain nested-loop convolution as the reference
    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        [batch, c_in, in_h, in_w]: [usize; 4],
        [c_out, k_h, k_w]: [usize; 3],
        [out_h, out_w]: [usize; 2],
        [s_h, s_w]: [usize; 2],
        [p_h, p_w]: [usize; 2],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * c_out * out_h * out_w];
        for b in 0..batch {
            for co in 0..c_out {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias.map_or(0.0, |bias| bias[co]);
                        for ci in 0..c_in {
                            for r in 0..k_h {
                                for c in 0..k_w {
                                    let ih = (oh * s_h + r) as isize - p_h as isize;
                                    let iw = (ow * s_w + c) as isize - p_w as isize;
                                    if ih >= 0
                                        && iw >= 0
                                        && (ih as usize) < in_h
                                        && (iw as usize) < in_w
                                    {
                                        sum += x[((b * c_in + ci) * in_h + ih as usize) * in_w
                                            + iw as usize]
                                            * w[((co * c_in + ci) * k_h + r) * k_w + c];
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

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    /// Instruction sets this CPU can run, so every microkernel is checked where possible
    fn supported() -> Vec<Isa> {
        let mut isas = Vec::new();
        if std::is_x86_feature_detected!("avx512f") {
            isas.push(Isa::Avx512);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            isas.push(Isa::Avx2);
        }
        isas
    }

    #[test]
    fn matches_reference() {
        // (batch, c_in, in_h, in_w, c_out, k, stride, pad, bias)
        let cases = [
            (1, 3, 15, 13, 5, 7, 2, 3, true),
            (2, 4, 9, 9, 9, 3, 1, 1, true),
            (1, 16, 7, 7, 17, 3, 1, 1, false),
            (1, 8, 14, 14, 16, 3, 2, 1, true),
            (1, 8, 14, 14, 13, 1, 2, 0, true),
            (1, 5, 6, 11, 8, 1, 1, 0, false),
            (2, 3, 5, 5, 7, 3, 1, 0, true),
            (1, 2, 4, 4, 3, 5, 1, 2, true),
            (1, 6, 12, 10, 24, 3, 3, 2, true),
        ];
        for (batch, c_in, in_h, in_w, c_out, k, s, p, with_bias) in cases {
            let out_h = (in_h + 2 * p - k) / s + 1;
            let out_w = (in_w + 2 * p - k) / s + 1;
            let x = values(batch * c_in * in_h * in_w, 1);
            let w = values(c_out * c_in * k * k, 2);
            let bias = values(c_out, 3);
            let bias = with_bias.then_some(bias.as_slice());
            let args = (
                [batch, c_in, in_h, in_w],
                [c_out, k, k],
                [out_h, out_w],
                [s, s],
                [p, p],
            );
            let expected = reference(&x, &w, bias, args.0, args.1, args.2, args.3, args.4);
            let runs = supported().into_iter().flat_map(|isa| {
                isa.shapes()
                    .iter()
                    .flat_map(move |&shape| [(isa, shape, 1), (isa, shape, 7)])
            });
            for (isa, shape, threads) in runs {
                let got = conv2d_direct_with(
                    isa,
                    &x,
                    &w,
                    bias,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    threads,
                    Some(shape),
                );
                assert_eq!(got.len(), expected.len());
                for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        (g - e).abs() <= 1e-4 * (1.0 + e.abs()),
                        "case {:?} {} {shape:?} threads {threads}: [{i}] {g} vs {e}",
                        (batch, c_in, in_h, in_w, c_out, k, s, p),
                        isa.name()
                    );
                }
            }
        }
    }

    /// Single-threaded throughput on ResNet-18 shapes:
    /// `cargo test --release --lib conv_direct::tests::throughput -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn throughput() {
        // (name, c_in, size, c_out, k, stride, pad)
        let shapes = [
            ("hot 16x3x3 @16", 16, 16, 16, 3, 1, 1),
            ("conv1 7x7/2", 3, 224, 64, 7, 2, 3),
            ("layer1 3x3 64", 64, 56, 64, 3, 1, 1),
            ("layer2 3x3 128", 128, 28, 128, 3, 1, 1),
            ("layer3 3x3 256", 256, 14, 256, 3, 1, 1),
            ("layer4 3x3 512", 512, 7, 512, 3, 1, 1),
            ("down 1x1/2 256", 256, 14, 512, 1, 2, 0),
        ];
        let runs = supported()
            .into_iter()
            .flat_map(|isa| isa.shapes().iter().map(move |&shape| (isa, Some(shape))))
            .chain(supported().into_iter().map(|isa| (isa, None)));
        for (isa, shape) in runs {
            for (name, c_in, size, c_out, k, s, p) in shapes {
                let out = (size + 2 * p - k) / s + 1;
                let x = values(c_in * size * size, 1);
                let w = values(c_out * c_in * k * k, 2);
                let run = || {
                    conv2d_direct_with(
                        isa,
                        &x,
                        &w,
                        None,
                        [1, c_in, size, size],
                        [c_out, k, k],
                        [out, out],
                        [s, s],
                        [p, p],
                        rayon::current_num_threads(),
                        shape,
                    )
                };
                for _ in 0..3 {
                    core::hint::black_box(run());
                }
                let flops = 2.0 * (c_out * out * out * c_in * k * k) as f64;
                let iters = ((2e9 / flops) as usize).clamp(3, 2000);
                let start = std::time::Instant::now();
                for _ in 0..iters {
                    core::hint::black_box(run());
                }
                let secs = start.elapsed().as_secs_f64() / iters as f64;
                std::println!(
                    "{:6} {:8} {name:16} {:8.3} ms {:6.1} GFLOP/s",
                    isa.name(),
                    shape.map_or("auto".into(), |(m, n)| std::format!("{m}x{n}")),
                    secs * 1e3,
                    flops / secs / 1e9
                );
            }
        }
    }

    /// Microkernel alone on cache-resident data, short and long K
    #[test]
    #[ignore]
    fn kernel_alone() {
        let Some(isa) = Isa::detect() else { return };
        for &(mr, nv) in isa.shapes() {
            for (k, taps) in [(64usize, alloc::vec![0usize]), (576, alloc::vec![0usize])] {
                let nb = nv * isa.lanes();
                let cs = 816;
                let b = values(k * cs + 64, 1);
                let w = values(mr * k, 2);
                let mut out = alloc::vec![0.0f32; mr * nb];
                let calls = 200_000usize;
                let start = std::time::Instant::now();
                for i in 0..calls {
                    let a = Args {
                        b: unsafe { b.as_ptr().add((i % 17) * nb % (cs - nb)) },
                        w: w.as_ptr(),
                        k_total: k,
                        c_in: k,
                        cs,
                        taps: &taps,
                        out: out.as_mut_ptr(),
                    };
                    unsafe { run_kernel(isa, mr, nv, a) };
                }
                core::hint::black_box(&out);
                let secs = start.elapsed().as_secs_f64();
                std::println!(
                    "{mr}x{nv} K={k}: {:.1} GFLOP/s",
                    (2 * mr * nb * k * calls) as f64 / secs / 1e9
                );
            }
        }
    }
}
