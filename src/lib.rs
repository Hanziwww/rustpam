use ndarray::ArrayView2;
use ndarray::Axis;
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// 初始化每个样本的最近和次近 medoid
#[inline(always)]
fn init_for_sample(
    j: usize,
    _k: usize,
    dist: &ArrayView2<f32>,
    medoids: &[usize],
    first_dist_init: f32,
    sec_dist_init: f32,
) -> (f32, f32, usize, usize) {
    let mut idx1 = 0;
    let mut idx2 = 0;
    let mut first_dist = first_dist_init;
    let mut sec_dist = sec_dist_init;

    // NOTE: This helper is no longer used in the initialization hot path, but kept for potential reuse.
    let col_j = dist.index_axis(Axis(1), j);
    for (kk, &med_idx) in medoids.iter().enumerate() {
        let d = unsafe { *col_j.uget(med_idx) };
        if d < sec_dist {
            if d <= first_dist {
                idx2 = idx1;
                idx1 = kk;
                sec_dist = first_dist;
                first_dist = d;
            } else {
                idx2 = kk;
                sec_dist = d;
            }
        }
    }

    (first_dist, sec_dist, idx1, idx2)
}

/// 评估候选点 i 的 swap 增益
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn evaluate_candidate(
    i: usize,
    k: usize,
    b: usize,
    is_medoid: &[bool],
    dist: &ArrayView2<f32>,
    min_dist_to_med: &[f32],
    second_min_dist_to_med: &[f32],
    nearest: &[usize],
    swap_gains_k: &[f32],
    delta_k: &mut [f32],
) -> (f32, usize) {
    if is_medoid[i] {
        return (f32::NEG_INFINITY, 0);
    }

    let mut swap_gain_add_i = 0.0;
    // delta_k is a thread-local reused buffer provided by the caller

    let row_i = dist.index_axis(Axis(0), i);
    for j in 0..b {
        let d = unsafe { *row_i.uget(j) };
        let mn = min_dist_to_med[j];
        let sc = second_min_dist_to_med[j];
        let kn = nearest[j];

        if d < mn {
            swap_gain_add_i += mn - d;
            delta_k[kn] += sc - mn;
        } else if d < sc {
            delta_k[kn] += sc - d;
        }
    }

    // 找到最佳的 k 来移除
    let mut k_best = 0;
    let mut over_k = swap_gains_k[0] + delta_k[0];
    for kk in 1..k {
        let v = swap_gains_k[kk] + delta_k[kk];
        if v > over_k || (v == over_k && kk < k_best) {
            over_k = v;
            k_best = kk;
        }
    }

    let gain_i = swap_gain_add_i + over_k;
    (gain_i, k_best)
}

/// 在 swap 后重新计算样本 j 的最近和次近 medoid
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn recompute_after_swap(
    j: usize,
    k: usize,
    best_k: usize,
    best_i: usize,
    dist: &ArrayView2<f32>,
    medoids: &[usize],
    first_dist_init: f32,
    sec_dist_init: f32,
    nearest_j: usize,
    second_j: usize,
    second_min_dist: f32,
) -> Option<(f32, f32, usize, usize, f32)> {
    // 只在需要时重新计算
    let row_best_i = dist.index_axis(Axis(0), best_i);
    if nearest_j == best_k
        || second_j == best_k
        || unsafe { *row_best_i.uget(j) } <= second_min_dist
    {
        let old_nearest = nearest_j;
        let delta_if_changed = if old_nearest != best_k {
            let row_old = dist.index_axis(Axis(0), medoids[old_nearest]);
            second_min_dist - unsafe { *row_old.uget(j) }
        } else {
            0.0
        };

        let (first_dist, sec_dist, idx1, idx2) =
            init_for_sample(j, k, dist, medoids, first_dist_init, sec_dist_init);

        Some((first_dist, sec_dist, idx1, idx2, delta_if_changed))
    } else {
        None
    }
}

/// Rust 实现的 PAM swap_eager 算法，使用 Rayon 并行化
#[pyfunction]
#[pyo3(signature = (dist, medoids_init, k, max_iter, n, b, tol, n_threads=0))]
#[allow(clippy::too_many_arguments)]
fn swap_eager<'py>(
    py: Python<'py>,
    dist: PyReadonlyArray2<f32>,
    medoids_init: Vec<usize>,
    k: usize,
    max_iter: usize,
    n: usize,
    b: usize,
    tol: f32,
    n_threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let dist = dist.as_array();
    let first_dist_init = f32::INFINITY;
    let sec_dist_init = f32::INFINITY;

    // 设置线程数
    if n_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .ok();
    }

    // 初始化工作数组
    let mut medoids: Vec<usize> = medoids_init.clone();
    let mut is_medoid = vec![false; n];
    for &m in &medoids {
        is_medoid[m] = true;
    }

    let mut min_dist_to_med = vec![0.0; b];
    let mut second_min_dist_to_med = vec![0.0; b];
    let mut nearest = vec![0; b];
    let mut second = vec![0; b];
    let mut swap_gains_k = vec![0.0; k];

    // 将核心计算放入无 GIL 环境
    let (medoids, nearest, min_dist_to_med, loss, steps) = py.allow_threads(|| {
        // Step 1: Row-major initialization over medoid rows to improve cache locality
        // Initialize arrays
        for j in 0..b {
            min_dist_to_med[j] = first_dist_init;
            second_min_dist_to_med[j] = sec_dist_init;
            nearest[j] = 0;
            second[j] = 0;
        }

        // Scan each medoid row and update per-sample min/second distances and indices
        for (kk, &med_idx) in medoids.iter().enumerate() {
            let row_m = dist.index_axis(Axis(0), med_idx);
            for j in 0..b {
                let d = unsafe { *row_m.uget(j) };
                let first = min_dist_to_med[j];
                let second_d = second_min_dist_to_med[j];
                if d < second_d {
                    if d <= first {
                        second[j] = nearest[j];
                        second_min_dist_to_med[j] = first;
                        nearest[j] = kk;
                        min_dist_to_med[j] = d;
                    } else {
                        second[j] = kk;
                        second_min_dist_to_med[j] = d;
                    }
                }
            }
        }

        // Compute initial loss and swap gains per medoid
        let mut loss = 0.0;
        for j in 0..b {
            swap_gains_k[nearest[j]] += min_dist_to_med[j] - second_min_dist_to_med[j];
            loss += min_dist_to_med[j];
        }

        let tol_abs = tol * loss;
        let mut steps = 0;

        // 主循环: 每轮找到最佳的 swap
        for s in 0..max_iter {
            steps = s;

            // Build non-medoid candidate list
            let non_medoids: Vec<usize> = (0..n).filter(|&i| !is_medoid[i]).collect();

            // Parallel evaluation of candidates with thread-local reusable delta_k buffers
            let (best_gain, best_k, best_i_idx) = non_medoids
                .into_par_iter()
                .map_init(
                    || vec![0.0f32; k],
                    |delta_k, i| {
                        // Reuse buffer by zero-filling once per candidate
                        for v in delta_k.iter_mut() {
                            *v = 0.0;
                        }
                        let (gain, k_best) = evaluate_candidate(
                            i,
                            k,
                            b,
                            &is_medoid,
                            &dist,
                            &min_dist_to_med,
                            &second_min_dist_to_med,
                            &nearest,
                            &swap_gains_k,
                            delta_k,
                        );
                        (gain, k_best, i)
                    },
                )
                .reduce(
                    || (f32::NEG_INFINITY, 0usize, usize::MAX),
                    |a, b| {
                        if a.0 > b.0 || (a.0 == b.0 && a.2 < b.2) {
                            a
                        } else {
                            b
                        }
                    },
                );

            if best_gain <= tol_abs || best_i_idx == usize::MAX {
                break;
            }

            let best_i = best_i_idx;

            // 应用 swap
            let old_medoid = medoids[best_k];
            medoids[best_k] = best_i;
            is_medoid[old_medoid] = false;
            is_medoid[best_i] = true;
            loss -= best_gain;
            swap_gains_k[best_k] = 0.0;

            // Parallel in-place recomputation for affected samples with thread-local reduction
            let chunk_size = 1024usize;
            use rayon::prelude::*;
            let deltas: Vec<Vec<f32>> = min_dist_to_med
                .par_chunks_mut(chunk_size)
                .zip(second_min_dist_to_med.par_chunks_mut(chunk_size))
                .zip(nearest.par_chunks_mut(chunk_size))
                .zip(second.par_chunks_mut(chunk_size))
                .enumerate()
                .map(
                    |(chunk_idx, (((min_chunk, sec_chunk), nearest_chunk), second_chunk))| {
                        let mut local_delta = vec![0.0f32; k];
                        let start = chunk_idx * chunk_size;
                        for offset in 0..min_chunk.len() {
                            let j = start + offset;
                            let nearest_j = nearest_chunk[offset];
                            let second_j = second_chunk[offset];
                            let second_min_d = sec_chunk[offset];
                            if let Some((first_dist, sec_dist, idx1, idx2, delta)) =
                                recompute_after_swap(
                                    j,
                                    k,
                                    best_k,
                                    best_i,
                                    &dist,
                                    &medoids,
                                    first_dist_init,
                                    sec_dist_init,
                                    nearest_j,
                                    second_j,
                                    second_min_d,
                                )
                            {
                                if nearest_j != best_k {
                                    local_delta[nearest_j] += delta;
                                }
                                min_chunk[offset] = first_dist;
                                sec_chunk[offset] = sec_dist;
                                nearest_chunk[offset] = idx1;
                                second_chunk[offset] = idx2;
                                local_delta[idx1] += first_dist - sec_dist;
                            }
                        }
                        local_delta
                    },
                )
                .collect();

            // Merge thread-local deltas to global swap_gains_k
            for local in deltas.iter() {
                for kk in 0..k {
                    swap_gains_k[kk] += local[kk];
                }
            }
        }

        (medoids, nearest, min_dist_to_med, loss, steps)
    });

    // 构建结果字典
    let result = PyDict::new(py);
    result.set_item("medoids", medoids.into_pyarray(py))?;
    result.set_item("nearest", nearest.into_pyarray(py))?;
    result.set_item("dist_to_nearest", min_dist_to_med.into_pyarray(py))?;
    result.set_item("loss", loss)?;
    result.set_item("steps", steps)?;

    Ok(result.clone())
}

/// Python module definition
#[pymodule]
fn _rustpam(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(swap_eager, m)?)?;
    Ok(())
}
