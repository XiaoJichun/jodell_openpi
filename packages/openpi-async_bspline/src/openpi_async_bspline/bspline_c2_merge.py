"""
C² B-spline merge (Li et al. 2022, Appendix A).

Ported from ominf/src/ominf/trajectory/bspline_c2_merge.py.
See the package README for integration and continuity limitations.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import time

import numpy as np

# 用于B-Spline曲线的手动定义节点、系数、阶数来创建样条对象(BSpline)、插值(make_interp_spline)和拟合(make_lsq_spline)
from scipy.interpolate import BSpline
from scipy.interpolate import make_interp_spline
from scipy.interpolate import make_lsq_spline


def fit_interpolate_bspline(x: np.ndarray, y: np.ndarray, *, degree: int) -> BSpline:
    """插值样条，曲线严格穿过所有数据点."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} points for degree-{degree} interpolation")
    return make_interp_spline(x, y, k=degree)


def fit_lsq_bspline(
    x: np.ndarray,
    y: np.ndarray,
    knots_interior: int = 6,
    *,
    degree: int,
) -> BSpline:
    """
    输入:
        一组轨迹点 (x=时间 t, y=关节 q)
        n_interior:内部节点数
        degree:B样条阶数
    输出: 一元夹紧型(clamped)最小二乘 B 样条 q(t)；自变量即逻辑时间。
        clamped: 端点节点重复 degree + 1 次；最小二乘拟合不保证端点严格插值。
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    t_min, t_max = float(x[0]), float(x[-1])

    interior = np.linspace(t_min, t_max, knots_interior + 2)[1:-1]

    knots = np.concatenate(([t_min] * (degree + 1), interior, [t_max] * (degree + 1)))

    return make_lsq_spline(x, y, knots, k=degree)


@contextmanager
def _timed(section: str, timing: dict[str, float]) -> Iterator[None]:
    """
    用于记录各子步骤代码耗时
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        # 记录各子步骤代码耗时
        timing[section] = timing.get(section, 0.0) + (time.perf_counter() - t0)


def compute_must_pass_t(window_t_start: float, must_pass_offset: float) -> float:
    return float(window_t_start) + float(must_pass_offset)


def split_scatter_for_merge(
    prev_all_timesteps: np.ndarray,
    prev_all_actions: np.ndarray,
    cur_all_timesteps: np.ndarray,
    cur_all_actions: np.ndarray,
    must_pass_t: float,
    *,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (切割散点)
    把前后两段轨迹(prev、cur)在 must_pass_t 时刻切开：
        左边只保留: t ≤ must_pass_t 的点（来自前一段）
        右边只保留: t > must_pass_t 的点（来自后一段）
        并且保证两边至少有degree+1个点，够拟合 B 样条
    """
    prev_all_timesteps = np.asarray(prev_all_timesteps, dtype=float)
    prev_all_actions = np.asarray(prev_all_actions, dtype=float)
    cur_all_timesteps = np.asarray(cur_all_timesteps, dtype=float)
    cur_all_actions = np.asarray(cur_all_actions, dtype=float)

    # 对prev_all_timesteps数组里每一个元素做判断，获得一个布尔值数组，用于后续的索引操作(索引为True的元素保留，索引为False的元素丢弃)
    m1 = prev_all_timesteps <= must_pass_t
    m2 = cur_all_timesteps > must_pass_t
    min_pts = degree + 1
    n1, n2 = int(m1.sum()), int(m2.sum())
    if n1 < min_pts or n2 < min_pts:
        raise ValueError(f"merge split at must_pass_t={must_pass_t}: left={n1}, right={n2} (need >={min_pts} each)")

    prev_move_scatter_timesteps = prev_all_timesteps[m1]
    prev_move_scatter_actions = prev_all_actions[m1]
    cur_move_scatter_timesteps = cur_all_timesteps[m2]
    cur_move_scatter_actions = cur_all_actions[m2]

    return prev_move_scatter_timesteps, prev_move_scatter_actions, cur_move_scatter_timesteps, cur_move_scatter_actions


def cur_add_must_pass_point(
    cur_move_scatter_timesteps: np.ndarray,
    cur_move_scatter_actions: np.ndarray,
    must_pass_t: float,
    must_pass_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为保证融合后的曲线在must_pass_t处连续，需要在右段散点最前面补一个「必过锚点」，再对补全后的散点做 LSQ 拟合，得到融合用的 spl2。
    输入:
        cur_move_scatter_timesteps:右段去除时间步小于must_pass_t的散点后的时间步
        cur_move_scatter_actions:右段去除时间步小于must_pass_t的散点后的关节位置
        must_pass_t:接缝时间
        must_pass_y:接缝关节位置
    输出:
        补全后的右段散点的时间步
        补全后的右段散点的关节位置
    """
    cur_move_scatter_timesteps = np.asarray(cur_move_scatter_timesteps, dtype=float)
    cur_move_scatter_actions = np.asarray(cur_move_scatter_actions, dtype=float)
    if len(cur_move_scatter_timesteps) > 0 and abs(cur_move_scatter_timesteps[0] - must_pass_t) < 1e-9:
        return cur_move_scatter_timesteps, cur_move_scatter_actions

    cur_move_scatter_timesteps_add_must = np.concatenate([[must_pass_t], cur_move_scatter_timesteps])
    cur_move_scatter_actions_add_must = np.concatenate([[must_pass_y], cur_move_scatter_actions])
    return (cur_move_scatter_timesteps_add_must, cur_move_scatter_actions_add_must)


def fit_merge_splines(
    prev_move_scatter_timesteps: np.ndarray,
    prev_move_scatter_actions: np.ndarray,
    cur_move_scatter_timesteps: np.ndarray,
    cur_move_scatter_actions: np.ndarray,
    must_pass_t: float,
    must_pass_y: float,
    *,
    knots_interior_merge: int,
    knots_interior: int,
    degree: int,
    spl1_lsq_fit: BSpline | None = None,
    max_deriv: int = 2,
) -> tuple[BSpline, BSpline]:
    """对 prev 和 current 两段轨迹散点分别拟合 spl1、spl2。

    一元 q(t) 约定：
        - 左段优先从 spl1_lsq_fit 在 must_pass_t 处裁剪（与已发布轨迹一致）；
        - 右段在 t > must_pass_t 散点前补必过锚点，使接缝参数与spl1对齐。
    """
    if spl1_lsq_fit is not None:
        spl1, _ = crop_merge_splines(
            spl1_lsq_fit,
            spl1_lsq_fit,
            must_pass_t,
            must_pass_t,
            max_deriv=max_deriv,
        )
    else:
        spl1 = fit_lsq_bspline(
            prev_move_scatter_timesteps, prev_move_scatter_actions, knots_interior=knots_interior_merge, degree=degree
        )

    # 在右段散点最前面补一个「必过锚点」，再对补全后的散点做 LSQ 拟合，得到融合用的 spl2。
    cur_move_scatter_timesteps_add_must, cur_move_scatter_actions_add_must = cur_add_must_pass_point(
        cur_move_scatter_timesteps, cur_move_scatter_actions, must_pass_t, must_pass_y
    )

    spl2 = fit_lsq_bspline(
        cur_move_scatter_timesteps_add_must,
        cur_move_scatter_actions_add_must,
        knots_interior=knots_interior,
        degree=degree,
    )
    return spl1, spl2


def split_bspline(spl: BSpline, u: float) -> tuple[BSpline, BSpline]:
    """
    (基础工具:切割样条)
    在参数 u 处将 B 样条分为左右两段(重复插入knots至重数 k+1)。

    几何不变，仅改变控制点 / 节点划分。

    输入
        spl:B样条曲线对象
        u:切割节点向量值的位置
    输出
        spl_left:左段B样条曲线对象
        spl_right:右段B样条曲线对象
    """

    # 样条的阶次
    k = spl.k

    spl_work = spl.insert_knot(float(u), m=k + 1)
    # 获取插入节点后的节点向量
    t = np.asarray(spl_work.t, dtype=float)
    # 获取插入节点后的控制点向量
    c = np.asarray(spl_work.c, dtype=float)
    # 获取插入节点后的节点向量中，u的位置
    eq = np.where(np.abs(t - u) < 1e-9)[0]
    if len(eq) == 0:
        raise ValueError(f"split_bspline: u={u} not found in knot vector")

    # 取最后节点的下标，并检查左右控制点数量是否足够
    # 一条合法 B 样条至少需要 k+1 个控制点，不能切太靠边
    i = int(eq[-1])
    n_left = i - k
    n_right = len(c) - n_left
    min_cp = k + 1
    if n_left < min_cp or n_right < min_cp:
        raise ValueError(
            f"split_bspline: u={u} too close to domain end (left_cp={n_left}, right_cp={n_right}, need >={min_cp})"
        )
    # B 样条切割的标准分法:切开左右对应的节点和控制点
    t_l, c_l = t[: i + 1], c[:n_left]
    t_r, c_r = t[i - k :], c[i - k :]
    if len(c_l) != len(t_l) - k - 1 or len(c_r) != len(t_r) - k - 1:
        raise ValueError("split_bspline: invalid knot/control sizing after split")
    return BSpline(t_l, c_l, k), BSpline(t_r, c_r, k)


def _interior_split_u(spl: BSpline, u: float) -> float:
    """将切分参数限制在样条域内部，避免在端点切分导致一侧无控制点。"""
    t_lo, t_hi = curve_timestep_range(spl)
    u = float(u)
    if u < t_lo or u > t_hi:
        raise ValueError(f"split u={u} outside domain [{t_lo}, {t_hi}]")
    span = max(t_hi - t_lo, 1e-12)
    u_try = u
    for i in range(64):
        try:
            split_bspline(spl, u_try)
            return u_try
        except ValueError:
            delta = span * min(0.05 * (i + 1), 0.45)
            if u_try >= t_hi - 1e-9:
                u_try = t_hi - delta
            elif u_try <= t_lo + 1e-9:
                u_try = t_lo + delta
            else:
                u_try = (u_try + (t_lo + t_hi) * 0.5) * 0.5
            u_try = float(np.clip(u_try, t_lo + 1e-9, t_hi - 1e-9))
    raise ValueError(f"no valid split point near u={u} on domain [{t_lo}, {t_hi}]")


def _crop_spline_left(spl: BSpline, u: float, *, max_deriv: int) -> BSpline:
    """保留 t ≤ u 段；u 在右端点时返回整条样条（无需切分）。"""
    t_lo, t_hi = curve_timestep_range(spl)
    u = float(u)
    if not (t_lo <= u <= t_hi):
        raise ValueError(f"crop left: u={u} outside domain [{t_lo}, {t_hi}]")
    min_cp = max_deriv + 1
    if u >= t_hi - 1e-9:
        if len(spl.c) < min_cp:
            raise ValueError(
                f"crop left: control points insufficient for C^{max_deriv} (n={len(spl.c)}, need >={min_cp})"
            )
        return spl
    u_split = _interior_split_u(spl, u)
    left, _ = split_bspline(spl, u_split)
    if len(left.c) < min_cp:
        raise ValueError(f"crop left: control points insufficient for C^{max_deriv} (n={len(left.c)}, need >={min_cp})")
    return left


def _crop_spline_right(spl: BSpline, u: float, *, max_deriv: int) -> BSpline:
    """保留 t ≥ u 段；u 在左端点时返回整条样条（新窗起点即 must_pass_t 时常见）。"""
    t_lo, t_hi = curve_timestep_range(spl)
    u = float(u)
    if not (t_lo <= u <= t_hi):
        raise ValueError(f"crop right: u={u} outside domain [{t_lo}, {t_hi}]")
    min_cp = max_deriv + 1
    if u <= t_lo + 1e-9:
        if len(spl.c) < min_cp:
            raise ValueError(
                f"crop right: control points insufficient for C^{max_deriv} (n={len(spl.c)}, need >={min_cp})"
            )
        return spl
    u_split = _interior_split_u(spl, u)
    _, right = split_bspline(spl, u_split)
    if len(right.c) < min_cp:
        raise ValueError(
            f"crop right: control points insufficient for C^{max_deriv} (n={len(right.c)}, need >={min_cp})"
        )
    return right


def crop_merge_splines(
    spl1_lsq_fit: BSpline,
    spl2_lsq_fit: BSpline,
    must_pass_t: float,
    ts_cur_right_start: float,
    *,
    max_deriv: int,
) -> tuple[BSpline, BSpline]:
    """
    (裁剪样条,针对左右两段样条)
    版本 crop: 从整个样条在must_pass处裁剪,不做第二次LSQ。

    把两条已经拟合好的完整 B 样条，分别裁剪成：
        左边只留到 must_pass_t(左段终点 t),
        右边从 must_pass_t 起（一元 q(t) 接缝与参数域一致；ts_cur_right_start 保留兼容）。
    """

    # 最少控制点要求
    min_cp = max_deriv + 1

    # 找到左段终点和右段起点的参数u，并切割两条曲线，得到可拼接版本的曲线
    u_left = _interior_split_u(spl1_lsq_fit, float(must_pass_t))
    t_lo1, t_hi1 = curve_timestep_range(spl1_lsq_fit)
    if not (t_lo1 <= u_left <= t_hi1):
        raise ValueError(f"crop: must_pass_t={must_pass_t} outside spl1 domain [{t_lo1}, {t_hi1}]")
    spl1_left, _ = split_bspline(spl1_lsq_fit, u_left)

    u_right = _interior_split_u(spl2_lsq_fit, float(must_pass_t))
    t_lo2, t_hi2 = curve_timestep_range(spl2_lsq_fit)
    if not (t_lo2 <= u_right <= t_hi2):
        raise ValueError(f"crop: must_pass_t={u_right} outside spl2 domain [{t_lo2}, {t_hi2}]")
    _, spl2_right = split_bspline(spl2_lsq_fit, u_right)
    if len(spl1_left.c) < min_cp or len(spl2_right.c) < min_cp:
        raise ValueError(
            f"crop: control points insufficient for C^{max_deriv} "
            f"(left={len(spl1_left.c)}, right={len(spl2_right.c)}, need >={min_cp})"
        )
    return spl1_left, spl2_right


def merge_window_trajectories(
    prev_all_timesteps: np.ndarray,
    prev_all_actions: np.ndarray,
    cur_all_timesteps: np.ndarray,
    cur_all_actions: np.ndarray,
    must_pass_t: float,
    must_pass_y: float,
    *,
    knots_interior_merge: int,
    knots_interior: int,
    degree: int,
    max_deriv: int,
    merge_input_mode: str,
    spl1_lsq_fit: BSpline | None = None,
    spl2_lsq_fit: BSpline | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[BSpline, BSpline, BSpline, BSpline]:
    """
    把前后两段轨迹，在 must_pass_t 这个点平滑融合成一条 C² 连续、且强制经过指定点的完整轨迹。
    支持两种融合模式：

        refit:切散点 → 重新拟合样条
        crop:直接裁剪已有样条（更快，不重新拟合）


    输入
        prev_all_timesteps:前一段轨迹的完整时间步数据
        prev_all_actions:前一段轨迹的完整关节位置数据
        cur_all_timesteps:后一段轨迹的完整时间步数据
        cur_all_actions:后一段轨迹的完整关节位置数据
        must_pass_t:融合时刻
        must_pass_y:融合时刻的关节位置
        n_interior_merge:左段B样条曲线内部节点数
        n_interior:右段B样条曲线内部节点数
        degree:B样条阶数
        max_deriv:最大导数阶数
        merge_input_mode:融合模式
        spl1_lsq_fit:左段B样条曲线对象
        spl2_lsq_fit:右段B样条曲线对象
        timing: 可选 dict,写入各子步骤耗时(秒)。
    输出
        spl1_move_scatter_spline:移除未参与融合的散点后的左段样条
        spl2_move_scatter_spline:移除未参与融合的散点后的右段样条
        merged_c2:融合后的C²连续B样条曲线对象
        merged_c2_must_pass:融合后的C²连续B样条曲线对象，且强制经过指定点
    """

    # 获取融合模式，并转换为小写
    mode = merge_input_mode.strip().lower()

    log: dict[str, float] = timing if timing is not None else {}

    with _timed("merge.prepare_inputs", log):
        if mode == "refit":
            with _timed("merge.slice", log):
                (
                    prev_move_scatter_timesteps,
                    prev_move_scatter_actions,
                    cur_move_scatter_timesteps,
                    cur_move_scatter_actions,
                ) = split_scatter_for_merge(
                    prev_all_timesteps, prev_all_actions, cur_all_timesteps, cur_all_actions, must_pass_t, degree=degree
                )
            spl1_move_scatter_spline, spl2_move_scatter_spline = fit_merge_splines(
                prev_move_scatter_timesteps,
                prev_move_scatter_actions,
                cur_move_scatter_timesteps,
                cur_move_scatter_actions,
                must_pass_t,
                must_pass_y,
                knots_interior_merge=knots_interior_merge,
                knots_interior=knots_interior,
                degree=degree,
                spl1_lsq_fit=spl1_lsq_fit,
                max_deriv=max_deriv,
            )
        elif mode == "crop":
            if spl1_lsq_fit is None or spl2_lsq_fit is None:
                raise ValueError("crop mode requires spl1_lsq_fit and spl2_full_bspline")
            ts_right = np.asarray(cur_all_timesteps, dtype=float)
            right_mask = ts_right > must_pass_t
            if not np.any(right_mask):
                raise ValueError(f"crop: no scatter points with t > must_pass_t={must_pass_t}")
            spl1_move_scatter_spline, spl2_move_scatter_spline = crop_merge_splines(
                spl1_lsq_fit,
                spl2_lsq_fit,
                must_pass_t,
                float(ts_right[right_mask][0]),
                max_deriv=max_deriv,
            )
        else:
            raise ValueError(f"merge_input_mode must be refit or crop, got {merge_input_mode!r}")

    with _timed("merge.c2", log):
        merged_c2 = merge_c2(spl1_move_scatter_spline, spl2_move_scatter_spline, max_deriv=max_deriv)

    with _timed("merge.merged_c2_must_pass", log):
        merged_c2_must_pass = enforce_must_pass_point(merged_c2, must_pass_t, must_pass_y)

    return spl1_move_scatter_spline, spl2_move_scatter_spline, merged_c2, merged_c2_must_pass


def _basis_derivs(
    knots: np.ndarray,
    degree: int,
    indices: list[int],
    u: float,
    *,
    max_deriv: int,
) -> np.ndarray:
    """
    用于merge_c2拼接，求出B样条曲线在u处的值和导数，以便计算最小扰动
        只需要拼接点附近几个控制点
        需要位置 + 速度 + 加速度连续
        解方程用
    """

    # 获取控制点数量
    n_ctrl = len(knots) - degree - 1
    out = np.zeros((max_deriv + 1, len(indices)))
    for col, i in enumerate(indices):
        c = np.zeros(n_ctrl)
        c[i] = 1.0
        spl = BSpline(knots, c, degree)
        out[0, col] = float(spl(u))
        for d in range(1, max_deriv + 1):
            out[d, col] = float(spl.derivative(d)(u))

    return out


def _basis_values(knots: np.ndarray, degree: int, u: float) -> np.ndarray:
    """
    用于强制过点/插值约束，求出B样条曲线在u处的值
        需要知道所有基函数在u处的权重
        只关心位置，不关心导数
        比如让曲线必须经过某个关键点位
    """

    # 计算控制点数量
    n_ctrl = len(knots) - degree - 1
    b = np.zeros(n_ctrl)
    for i in range(n_ctrl):
        c = np.zeros(n_ctrl)
        c[i] = 1.0
        spl = BSpline(knots, c, degree)
        b[i] = float(spl(u))
    return b


def _transform_knots(knots2: np.ndarray, u_joint: float, u_prev: float, degree: int) -> np.ndarray:
    """
    因为两条样条曲线要融合，但是曲线的刻度不一致、终点和起点也不一致，无法直接拼接：
    将第二段曲线的节点向量进行线性变换(把第二条样条“节点向量”，做一次【平移+缩放】仿射变换)，使两段样条的拼接点参数坐标对齐
    平移：将第二条样条的拼接点坐标平移到第一条样条的拼接点坐标
    缩放：将第二条样条的节点向量缩放到与第一条样条的节点向量相同
    """

    # degree=3  因此，叫s3，s4
    # 此外，对于clamped型p次B样条，有效参数区间从第p个节点开始 ---> knots[3]
    # 前p+1个节点重复
    # knots = [s0, s1, s2, s3, s4, s5, s6, ...]
    #    = [0,  0,  0,  0,  2,  4,  6, ...]
    s3 = knots2[degree]
    s4 = knots2[degree + 1]

    if abs(s3 - u_joint) < 1e-9:
        return np.asarray(knots2, dtype=float)
    scale = (u_joint - u_prev) / (s4 - s3)
    return u_joint + (knots2 - s3) * scale


def _prev_distinct_knot(knots: np.ndarray, u_end: float, degree: int) -> float:
    """
    在节点向量knots中，找到严格小于u_end的最后一个节点值---> 找到 spl1 最后一段的起点。
    """
    # 从后向前遍历节点向量，找到严格小于u_end的最后一个节点值
    for i in range(len(knots) - 2, -1, -1):
        # 如果节点值小于u_end，则返回该节点值
        if knots[i] < u_end - 1e-9:
            return float(knots[i])
    return float(knots[degree])


def merge_c2(spl1: BSpline, spl2: BSpline, *, max_deriv: int) -> BSpline:
    """
    实现两条B样条曲线spl1、spl2在拼接点u_joint处C^n连续,只微调连接处附近的少量控制点，不改动整条曲线
    """

    # 获取B样条曲线的阶数，保证两条曲线阶数一致，才可融合
    k = spl1.k
    if spl1.k != spl2.k:
        raise ValueError("Both curves must have the same B-spline degree")
    if max_deriv < 1:
        raise ValueError(f"max_deriv must be >= 1, got {max_deriv}")
    if k < max_deriv + 1:
        raise ValueError(f"degree k={k} too small for max_deriv={max_deriv} (need k >= max_deriv + 1)")

    # 获取控制点(决定曲线形状)和节点向量(决定分段区间)
    c1 = np.asarray(spl1.c, dtype=float)
    c2 = np.asarray(spl2.c, dtype=float)
    t1 = np.asarray(spl1.t, dtype=float)
    t2 = np.asarray(spl2.t, dtype=float)

    n = len(c1) - 1
    m = len(c2) - 1

    # 找到两段曲线的公共连接点u_joint:第一段曲线在这里结束，第二段曲线必须从这里开始
    # spl1的有效参数区间是[t1[k], t1[len(c1)]]，t1[len(c1)] 就是 spl1 真正的终点，即公共点
    u_joint = float(t1[len(c1)])
    # 找到第一段曲线中，最后一个严格小于u_joint的节点
    # u_prev 是节点最终重复值，前面那个不一样的节点
    u_prev = _prev_distinct_knot(t1, u_joint, k)
    # 对第二段曲线进行线性变换，使拼接点对齐，保证两条曲线参数域严丝合缝
    tilde_t2 = _transform_knots(t2, u_joint, u_prev, k)

    # 拼接邻域:左末 (max_deriv+1) 个、右首 (max_deriv+1) 个；k=3 且 max_deriv=2 时为 [n-2,n-1,n] 与 [0,1,2]
    # 求取使用哪一些控制点来微调融合曲线，使得两条曲线在拼接点处C^n连续
    # B样条基函数是局部支撑的！
    #     在接缝 u_joint 处：
    #         只有 spl1 最后 max_deriv+1 个控制点 有影响
    #         只有 spl2 最前 max_deriv+1 个控制点 有影响
    #         其它控制点离得远，基函数 = 0，完全没用
    #     例子：max_deriv=2（C² 连续）
    #         n_junc = 3
    #         idx1 = 最后 3 个控制点id
    #         idx2 = 最前 3 个控制点id
    n_junc = max_deriv + 1
    idx1 = list(range(n - max_deriv, n + 1))
    idx2 = list(range(n_junc))

    # n1：spl1 3个基函数N_i(u)在接缝处的 0/1/2 阶导数
    # n2：spl2 3个基函数N_i(u)在接缝处的 0/1/2 阶导数
    n1 = _basis_derivs(t1, k, idx1, u_joint, max_deriv=max_deriv)
    n2 = _basis_derivs(tilde_t2, k, idx2, u_joint, max_deriv=max_deriv)

    # 取出要微调的控制点
    p1 = c1[idx1]
    r2 = c2[idx2]

    # 解方程：最小扰动实现 C⁰ C¹ C² 连续
    # @：点乘
    # gap：两段曲线在接缝处的 0/1/2阶导数 的误差，要通过调整 eps、delta 把这个gap消掉

    # 使用拉格朗日乘子法求解最小扰动 eps、delta
    gap = n2 @ r2 - n1 @ p1
    n1_inv_t = np.linalg.inv(n1).T
    a_mat = n1 + n2 @ n2.T @ n1_inv_t
    # x = np.linalg.solve(A, b)
    # 对应数学式： Ax=b
    #     已知矩阵 A、向量 b
    #     求未知向量 x
    eps = np.linalg.solve(a_mat, gap)
    delta = -np.linalg.solve(n1, n2).T @ eps

    c1_new = c1.copy()
    c2_new = c2.copy()
    c1_new[idx1] = p1 + eps
    c2_new[idx2] = r2 + delta

    # 拼接两条曲线的控制点，生成新的控制点向量，以便生成新的B样条曲线
    merged_c = np.concatenate([c1_new[: n + 1], c2_new[n_junc : m + 1]])
    # 拼接两条曲线的节点向量，生成新的节点向量，以便生成新的B样条曲线
    merged_knots = np.concatenate([t1[: n + 1], tilde_t2[k:]])

    return BSpline(merged_knots, merged_c, k)


def find_u_at_timestep(spl: BSpline, timestep: float) -> float:
    """
    一元样条 q(t)：有效参数即逻辑时间，域内直接返回 t。

    保留此函数名以兼容调用方；不再做 brentq 反解。
    """
    t_lo, t_hi = curve_timestep_range(spl)
    if not (t_lo <= timestep <= t_hi):
        raise ValueError(f"timestep {timestep} outside domain [{t_lo}, {t_hi}]")
    return float(timestep)


def enforce_must_pass_point(spl: BSpline, timestep: float, curve_y: float) -> BSpline:
    """
    最小扰动修正，使 q(timestep) = curve_y。
    """
    u_star = find_u_at_timestep(spl, timestep)
    b = _basis_values(spl.t, spl.k, u_star)
    active = np.where(b > 1e-12)[0]
    b_a = b[active]

    denom = float(b_a @ b_a)
    if denom < 1e-15:
        return spl

    c_new = np.asarray(spl.c, dtype=float).copy()
    residual_y = float(curve_y) - float(spl(u_star))
    c_new[active] += b_a * residual_y / denom
    return BSpline(spl.t, c_new, spl.k)


def merge_c2_with_must_pass(
    spl1: BSpline,
    spl2: BSpline,
    must_pass_timestep: float,
    must_pass_curve_y: float,
    *,
    max_deriv: int,
) -> BSpline:
    """
    第一步：merged = merge_c2(...)

    输入：两条分段轨迹 spl1、spl2
    动作：求解连接处控制点修正量，实现 C⁰~C^max_deriv 连续
    输出：一条整体平滑、无突变的拼接曲线
    特点：只保证平滑，不保证经过任何特定点

    第二步：enforce_must_pass_point(...)

    输入：上一步得到的平滑曲线 + 必须经过的时间点
    动作：
        找到时间对应的参数 u_star
        提取局部有效控制点
        用拉格朗日最小范数修正，让曲线精确过点
    输出：既平滑、又过指定点的最终曲线
    """
    merged = merge_c2(spl1, spl2, max_deriv=max_deriv)
    return enforce_must_pass_point(merged, must_pass_timestep, must_pass_curve_y)


def curve_timestep_range(spl: BSpline) -> tuple[float, float]:
    """

    输入: spl B样条曲线
        spl.t → 节点向量
        spl.k → 阶数
        spl.c → 控制点
    输出: B样条曲线真正有效的参数区间[t_start, t_end]

    节点向量 = [ 0.  0.  0.  0.  7. 14. 21. 28. 35. 42. 49. 49. 49. 49.]
    左边界: spl.t[k]=spl.t[3] = 0    右边界: spl.t[len(c)]spl.t[8] = 49
    """
    return float(spl.t[spl.k]), float(spl.t[len(spl.c)])


def trajectory_p_v_a_at_timestep(spl: BSpline, timestep: float) -> tuple[float, float, float]:
    """一元 q(t)：直接求值与对 t 的导数。"""
    t = find_u_at_timestep(spl, timestep)
    return (
        float(spl(t)),
        float(spl.derivative(1)(t)),
        float(spl.derivative(2)(t)),
    )


def control_point_abscissae(spl: BSpline) -> np.ndarray:
    """Greville 横坐标，用于绘制一元样条控制点折线。"""
    k = spl.k
    t = spl.t
    n = len(spl.c)
    return np.array([float(np.mean(t[i + 1 : i + k + 1])) for i in range(n)])


# @dataclass(frozen=True)
# class KinematicsOptions:
#     """控制是否计算 / 缓存 / 发布位置、速度、加速度。"""

#     position: bool = True
#     velocity: bool = True
#     acceleration: bool = True

#     @classmethod
#     def from_client_config(cls, cli: dict) -> KinematicsOptions:
#         return cls(
#             position=bool(cli.get("publish_position", True)),
#             velocity=bool(cli.get("publish_velocity", True)),
#             acceleration=bool(cli.get("publish_acceleration", True)),
#         )

#     def needs_cache_kinematics(self) -> bool:
#         return self.position or self.velocity or self.acceleration


@dataclass(frozen=True)
class KinematicsOptions:
    """控制是否计算 / 缓存 / 发布位置、速度、加速度。"""

    position: bool = True
    velocity: bool = True
    acceleration: bool = True

    @classmethod
    def from_client_config(
        cls,
        *,
        publish_position: bool,
        publish_velocity: bool,
        publish_acceleration: bool,
    ) -> KinematicsOptions:
        return cls(
            position=publish_position,
            velocity=publish_velocity,
            acceleration=publish_acceleration,
        )

    def needs_cache_kinematics(self) -> bool:
        return self.position or self.velocity or self.acceleration


@dataclass(frozen=True)
class DofKinematicsCache:
    """单关节在 timestep 轴上的查表；未启用的运动学分量为 None。"""

    timestep: np.ndarray
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None

    def lookup(self, t: float, opts: KinematicsOptions) -> tuple[float | None, float | None, float | None]:
        pos = vel = acc = None
        if opts.position and self.position is not None:
            pos = float(np.interp(t, self.timestep, self.position))
        if opts.velocity and self.velocity is not None:
            vel = float(np.interp(t, self.timestep, self.velocity))
        if opts.acceleration and self.acceleration is not None:
            acc = float(np.interp(t, self.timestep, self.acceleration))
        return pos, vel, acc


def build_kinematics_cache(
    spl: BSpline,
    n_samples: int,
    opts: KinematicsOptions,
    *,
    t_start: float | None = None,
    t_end: float | None = None,
) -> DofKinematicsCache:
    """在逻辑时间轴 [t_start, t_end] 上均匀采样；默认整条样条有效区间。"""
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    k = spl.k
    spl_lo = float(spl.t[k])
    spl_hi = float(spl.t[len(spl.c)])
    lo = max(spl_lo, float(t_start)) if t_start is not None else spl_lo
    hi = min(spl_hi, float(t_end)) if t_end is not None else spl_hi
    if hi < lo:
        lo, hi = spl_lo, spl_hi
    ts = np.linspace(lo, hi, n_samples)
    pos = np.asarray(spl(ts), dtype=float) if opts.position else None
    vel = np.asarray(spl.derivative(1)(ts), dtype=float) if opts.velocity else None
    acc = np.asarray(spl.derivative(2)(ts), dtype=float) if opts.acceleration else None
    return DofKinematicsCache(ts, pos, vel, acc)


def trajectory_position_at_timestep(spl: BSpline, timestep: float) -> float:
    """仅求位置 q(t)，不算导数。"""
    t = find_u_at_timestep(spl, timestep)
    return float(spl(t))


@dataclass
class ActiveTrajectory:
    """
    激活轨迹：一条完整机器人轨迹的封装
        内部：所有关节的样条曲线列表
        可选：开启缓存加速（DofKinematicsCache）
        对外：提供位置 / 速度 / 加速度查询接口
        上层（控制 / 发布线程）只和它打交道，不用管内部样条细节。

        splines：轨迹的真值来源
        cache：预计算加速表（可选）
            有 cache → 超快查表
            无 cache → 实时算样条

    """

    splines: tuple[BSpline, ...]
    cache: tuple[DofKinematicsCache, ...] | None = None
    kinematics: KinematicsOptions = KinematicsOptions()

    @classmethod
    def from_splines(
        cls,
        splines: list[BSpline],
        *,
        cache_samples: int = 0,
        kinematics: KinematicsOptions | None = None,
        cache_t_start: float | None = None,
        cache_t_end: float | None = None,
    ) -> ActiveTrajectory:
        """
        输入:
            splines: 所有关节的样条曲线列表
            cache_samples: 缓存样本数
            cache_t_start / cache_t_end: cache 采样逻辑时间范围（默认整条样条）
        输出:
            ActiveTrajectory对象
        """
        spl_tuple = tuple(splines)
        kin = kinematics if kinematics is not None else KinematicsOptions()
        cache: tuple[DofKinematicsCache, ...] | None = None
        if cache_samples > 0:
            cache = tuple(
                build_kinematics_cache(
                    s,
                    cache_samples,
                    kin,
                    t_start=cache_t_start,
                    t_end=cache_t_end,
                )
                for s in spl_tuple
            )

        print(f"[cache_samples]: {cache_samples}")
        print(f"[cache_t_start]: {cache_t_start}")
        print(f"[cache_t_end]: {cache_t_end}")
        return cls(spl_tuple, cache, kin)

    @property
    def n_dof(self) -> int:
        """
        获取关节数,即样条曲线列表的长度
        """
        return len(self.splines)

    @property
    def t_range(self) -> tuple[float, float]:
        """
        获取轨迹的时间范围,即第一条样条曲线的时间范围
        """
        return curve_timestep_range(self.splines[0])

    def all_trajectory_p_at_timestep(self, t: float, dof: int = 0) -> float:
        """融合必过点等：仅求关节位置，不算速度/加速度。"""
        if self.cache is not None:
            c = self.cache[dof]
            if c.position is not None:
                return float(np.interp(t, c.timestep, c.position))
        return trajectory_position_at_timestep(self.splines[dof], t)

    def all_trajectory_p_v_a_at_timestep(
        self, t: float
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """按 kinematics 开关仅计算启用的分量。"""
        opts = self.kinematics
        n = self.n_dof
        pos = np.empty(n) if opts.position else None
        vel = np.empty(n) if opts.velocity else None
        acc = np.empty(n) if opts.acceleration else None
        if self.cache is not None:
            for d, c in enumerate(self.cache):
                p, v, a = c.lookup(t, opts)
                if pos is not None and p is not None:
                    pos[d] = p
                if vel is not None and v is not None:
                    vel[d] = v
                if acc is not None and a is not None:
                    acc[d] = a
        else:
            for d, spl in enumerate(self.splines):
                u = find_u_at_timestep(spl, t)
                if pos is not None:
                    pos[d] = float(spl(u))
                if vel is not None:
                    vel[d] = float(spl.derivative(1)(u))
                if acc is not None:
                    acc[d] = float(spl.derivative(2)(u))
        return pos, vel, acc

    def sample_scatter(self, n_points: int) -> dict[str, np.ndarray]:
        """在逻辑时间轴上均匀取散点；仅包含 kinematics 启用的分量。"""
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        opts = self.kinematics
        t_lo, t_hi = self.t_range
        ts = np.linspace(t_lo, t_hi, n_points)
        out: dict[str, np.ndarray] = {"timestep": ts}
        if not opts.needs_cache_kinematics():
            return out
        pos_arr = np.empty((n_points, self.n_dof)) if opts.position else None
        vel_arr = np.empty((n_points, self.n_dof)) if opts.velocity else None
        acc_arr = np.empty((n_points, self.n_dof)) if opts.acceleration else None
        for i, t in enumerate(ts):
            p, v, a = self.all_trajectory_p_v_a_at_timestep(float(t))
            if pos_arr is not None and p is not None:
                pos_arr[i] = p
            if vel_arr is not None and v is not None:
                vel_arr[i] = v
            if acc_arr is not None and a is not None:
                acc_arr[i] = a
        if pos_arr is not None:
            out["position"] = pos_arr
        if vel_arr is not None:
            out["velocity"] = vel_arr
        if acc_arr is not None:
            out["acceleration"] = acc_arr
        return out


def evaluate_curve(spl: BSpline, n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """采样一元样条 q(t)，返回 (timestep, joint)。"""
    k = spl.k
    t = np.linspace(float(spl.t[k]), float(spl.t[len(spl.c)]), n_samples)
    return t, np.asarray(spl(t), dtype=float)


def must_pass_error(spl: BSpline, timestep: float, curve_y: float) -> tuple[float, float]:
    """必过点误差：一元样条时间维无残差，仅报告关节误差。"""
    t = find_u_at_timestep(spl, timestep)
    return 0.0, float(abs(float(spl(t)) - curve_y))


def junction_errors(spl1: BSpline, spl2: BSpline, merged: BSpline) -> dict[str, float]:
    """拼接点 t_joint 处 C0/C1/C2 融合前后误差（一元 q(t)）。"""
    u = float(spl1.t[len(spl1.c)])
    h = 1e-5
    errs: dict[str, float] = {}

    v1_0, v2_0 = float(spl1(u)), float(spl2(u))
    errs["C0_before"] = abs(v1_0 - v2_0)
    errs["C0_after"] = abs(float(merged(u - h)) - float(merged(u + h)))

    v1_1 = float(spl1.derivative(1)(u - h))
    v2_1 = float(spl2.derivative(1)(u + h))
    errs["C1_before"] = abs(v1_1 - v2_1)
    dm1 = merged.derivative(1)
    errs["C1_after"] = abs(float(dm1(u - h)) - float(dm1(u + h)))

    v1_2 = float(spl1.derivative(2)(u - h))
    v2_2 = float(spl2.derivative(2)(u + h))
    errs["C2_before"] = abs(v1_2 - v2_2)
    dm2 = merged.derivative(2)
    errs["C2_after"] = abs(float(dm2(u - h)) - float(dm2(u + h)))

    return errs
