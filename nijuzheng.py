

import math
import re
import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# 第一部分：自定义异常
# ============================================================

class MatrixError(Exception):
    """所有矩阵相关错误的基类"""
    pass


class NonSquareMatrixError(MatrixError):
    """不是方阵"""
    pass


class SingularMatrixError(MatrixError):
    """奇异矩阵，没有逆矩阵"""
    pass


class InvalidElementError(MatrixError):
    """元素不是合法数字"""
    pass


# ============================================================
# 第二部分：输入校验
# ============================================================

def validate_matrix(A):
    """
    检查 A 是不是一个合法的方阵。
    合法：返回矩阵大小 n
    不合法：抛出对应异常
    """
    if not isinstance(A, (list, tuple)):
        raise MatrixError("矩阵必须是一个二维列表，例如 [[1,2],[3,4]]")

    n = len(A)
    if n == 0:
        raise MatrixError("矩阵不能为空")

    # 每行必须是列表，且列数等于行数
    for i, row in enumerate(A):
        if not isinstance(row, (list, tuple)):
            raise MatrixError(f"第 {i + 1} 行不是列表：{row!r}")
        if len(row) != n:
            raise NonSquareMatrixError(
                f"矩阵必须是方阵。\n"
                f"当前第 {i + 1} 行有 {len(row)} 列，"
                f"但总共有 {n} 行，行列不相等，不能求逆。"
            )

    # 每个元素必须是有限数字
    for i, row in enumerate(A):
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise InvalidElementError(
                    f"第 {i + 1} 行第 {j + 1} 列不是数字：{v!r}"
                )
            if not math.isfinite(v):
                raise InvalidElementError(
                    f"第 {i + 1} 行第 {j + 1} 列不是有限数字：{v}"
                )

    return n


# ============================================================
# 第三部分：核心算法
# ============================================================

def _snapshot(matrix):
    """复制一份矩阵，用于保存历史状态"""
    return [row[:] for row in matrix]


def inverse_matrix(A, tol=None):
    """
    用带部分主元的高斯-约当消元法求逆矩阵。

    返回：
        (A_inv, steps)
        A_inv : 逆矩阵
        steps : 每一步的过程记录，用于图形化展示
    """
    # 1) 校验
    n = validate_matrix(A)

    # 2) 转成浮点数，避免整数除法问题
    M = [[float(v) for v in row] for row in A]

    # 3) 构造增广矩阵 [A | I]
    aug = []
    for i in range(n):
        identity_row = [1.0 if i == j else 0.0 for j in range(n)]
        aug.append(M[i][:] + identity_row)

    # 4) 计算主元阈值（和矩阵量级、规模有关）
    if tol is None:
        max_abs = max(abs(v) for row in M for v in row)
        tol = 1e-12 * max(1.0, max_abs) * n

    # 5) 记录步骤
    steps = []

    def record(description, highlight=None):
        steps.append({
            "matrix": _snapshot(aug),
            "description": description,
            "highlight": highlight or [],
        })

    record("初始状态：左边是原矩阵 A，右边是单位矩阵 I。"
           "目标是通过行变换，把左边变成单位矩阵。")

    # 6) 逐列消元
    for k in range(n):
        # 6.1 选主元：当前列 k 中绝对值最大的行
        pivot_row = max(range(k, n), key=lambda r: abs(aug[r][k]))

        # 6.2 如果最大都接近 0，矩阵奇异
        if abs(aug[pivot_row][k]) < tol:
            raise SingularMatrixError(
                f"矩阵是奇异矩阵，没有逆矩阵。\n\n"
                f"原因：处理到第 {k + 1} 列时，找不到足够大的主元。\n"
                f"最大绝对值只有 {abs(aug[pivot_row][k]):.3e}，"
                f"小于阈值 {tol:.3e}。\n"
                f"这通常意味着矩阵的行之间线性相关。"
            )

        # 6.3 交换行
        if pivot_row != k:
            aug[k], aug[pivot_row] = aug[pivot_row], aug[k]
            record(
                f"【换行】交换第 {k + 1} 行和第 {pivot_row + 1} 行。\n"
                f"目的：把绝对值最大的数放到主元位置，减少计算误差。",
                [k, pivot_row]
            )

        # 6.4 归一化：主元变成 1
        pivot = aug[k][k]
        if abs(pivot - 1.0) > 1e-15:
            aug[k] = [v / pivot for v in aug[k]]
            record(
                f"【归一】第 {k + 1} 行整体除以 {pivot:.4f}，"
                f"使主元变成 1。",
                [k]
            )

        # 6.5 消元：其他行第 k 列变成 0
        for r in range(n):
            if r == k:
                continue
            factor = aug[r][k]
            if abs(factor) < 1e-15:
                continue
            aug[r] = [aug[r][c] - factor * aug[k][c] for c in range(2 * n)]
            record(
                f"【消元】第 {r + 1} 行 减去 ({factor:.4f}) × 第 {k + 1} 行，"
                f"使第 {k + 1} 列变成 0。",
                [r, k]
            )

    record("完成！左边已经是单位矩阵，右边就是逆矩阵 A⁻¹。")

    # 7) 取出右半部分
    A_inv = [row[n:] for row in aug]
    return A_inv, steps


def verify_inverse(A, A_inv, tol=1e-6):
    """验证 A × A_inv 是否接近单位矩阵，返回 (最大误差, 是否通过)"""
    n = len(A)
    max_err = 0.0
    for i in range(n):
        for j in range(n):
            s = sum(A[i][k] * A_inv[k][j] for k in range(n))
            target = 1.0 if i == j else 0.0
            max_err = max(max_err, abs(s - target))
    return max_err, max_err < tol


# ============================================================
# 第四部分：内置自测
# ============================================================

def _allclose(A, B, tol=1e-9):
    """比较两个矩阵是否几乎相等"""
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def run_self_tests():
    """
    跑一组内置测试。
    返回 (通过数, 总数, 详细列表)
    """
    results = []

    def case(name, func):
        try:
            func()
            results.append((name, True, ""))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))

    # ---- 1. 2×2 已知答案 ----
    def t1():
        A = [[2, 1], [1, 1]]
        inv, _ = inverse_matrix(A)
        expect = [[1.0, -1.0], [-1.0, 2.0]]
        assert _allclose(inv, expect, 1e-9), f"结果不对：{inv}"
    case("2×2 已知答案", t1)

    # ---- 2. 3×3 验证残差 ----
    def t2():
        A = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
        inv, _ = inverse_matrix(A)
        err, ok = verify_inverse(A, inv)
        assert ok, f"残差过大：{err}"
    case("3×3 验证残差", t2)

    # ---- 3. 单位矩阵 ----
    def t3():
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        inv, _ = inverse_matrix(A)
        assert _allclose(inv, A, 1e-9), f"结果不对：{inv}"
    case("单位矩阵", t3)

    # ---- 4. 对角矩阵 ----
    def t4():
        A = [[2, 0, 0], [0, 4, 0], [0, 0, 5]]
        inv, _ = inverse_matrix(A)
        expect = [[0.5, 0, 0], [0, 0.25, 0], [0, 0, 0.2]]
        assert _allclose(inv, expect, 1e-9), f"结果不对：{inv}"
    case("对角矩阵", t4)

    # ---- 5. 需要换行 ----
    def t5():
        A = [[0, 1], [1, 0]]
        inv, _ = inverse_matrix(A)
        err, ok = verify_inverse(A, inv)
        assert ok, f"残差过大：{err}"
    case("需要换行的情况", t5)

    # ---- 6. 非方阵应报错 ----
    def t6():
        try:
            inverse_matrix([[1, 2, 3], [4, 5, 6]])
        except NonSquareMatrixError:
            return
        raise AssertionError("应该报 NonSquareMatrixError，但没有")
    case("非方阵 → 报错", t6)

    # ---- 7. 奇异矩阵应报错 ----
    def t7():
        try:
            inverse_matrix([[1, 2], [2, 4]])
        except SingularMatrixError:
            return
        raise AssertionError("应该报 SingularMatrixError，但没有")
    case("奇异矩阵 → 报错", t7)

    # ---- 8. 字符串元素应报错 ----
    def t8():
        try:
            inverse_matrix([[1, "a"], [2, 3]])
        except InvalidElementError:
            return
        raise AssertionError("应该报 InvalidElementError，但没有")
    case("非数字元素 → 报错", t8)

    # ---- 9. NaN 应报错 ----
    def t9():
        try:
            inverse_matrix([[1, float("nan")], [2, 3]])
        except InvalidElementError:
            return
        raise AssertionError("应该报 InvalidElementError，但没有")
    case("NaN → 报错", t9)

    # ---- 10. 无穷大应报错 ----
    def t10():
        try:
            inverse_matrix([[1, float("inf")], [2, 3]])
        except InvalidElementError:
            return
        raise AssertionError("应该报 InvalidElementError，但没有")
    case("Inf → 报错", t10)

    # ---- 11. 空矩阵应报错 ----
    def t11():
        try:
            inverse_matrix([])
        except MatrixError:
            return
        raise AssertionError("应该报 MatrixError，但没有")
    case("空矩阵 → 报错", t11)

    # ---- 12. 参差不齐的行 ----
    def t12():
        try:
            inverse_matrix([[1, 2], [3]])
        except NonSquareMatrixError:
            return
        raise AssertionError("应该报 NonSquareMatrixError，但没有")
    case("行列不齐 → 报错", t12)

    # ---- 13. 希尔伯特矩阵（病态）----
    def t13():
        n = 4
        A = [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]
        inv, _ = inverse_matrix(A)
        err, ok = verify_inverse(A, inv, tol=1e-6)
        assert ok, f"残差过大：{err}"
    case("希尔伯特矩阵（病态）", t13)

    # ---- 14. 随机矩阵批量验证 ----
    def t14():
        import random
        random.seed(42)
        count = 0
        for _ in range(30):
            n = random.randint(2, 6)
            A = [[random.uniform(-5, 5) for _ in range(n)] for _ in range(n)]
            try:
                inv, _ = inverse_matrix(A)
            except SingularMatrixError:
                continue
            err, ok = verify_inverse(A, inv, tol=1e-6)
            assert ok, f"残差过大：{err}, 矩阵={A}"
            count += 1
        assert count >= 20, f"有效样本太少：{count}"
    case("30 个随机矩阵", t14)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    return passed, total, results


# ============================================================
# 第五部分：图形界面
# ============================================================

class MatrixInverseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("矩阵求逆演示器（合并版）")
        self.root.geometry("980x740")

        self.steps = []
        self.current = 0
        self.playing = False
        self.play_job = None

        self._build_ui()

    # ---------------- 界面搭建 ----------------
    def _build_ui(self):
        # 输入区
        top = ttk.LabelFrame(self.root, text="① 输入矩阵（每行用空格或逗号分隔）")
        top.pack(fill="x", padx=10, pady=5)

        self.input_text = tk.Text(top, height=6, font=("Consolas", 12))
        self.input_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.input_text.insert("1.0", "2 1\n1 1")

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side="right", padx=5)

        ttk.Button(btn_frame, text="计算", width=10,
                   command=self.on_calc).pack(pady=2)
        ttk.Button(btn_frame, text="示例 2×2", width=10,
                   command=self.on_example_2).pack(pady=2)
        ttk.Button(btn_frame, text="示例 3×3", width=10,
                   command=self.on_example_3).pack(pady=2)
        ttk.Button(btn_frame, text="自测", width=10,
                   command=self.on_self_test).pack(pady=2)
        ttk.Button(btn_frame, text="清空", width=10,
                   command=self.on_clear).pack(pady=2)

        # 矩阵展示
        mid = ttk.LabelFrame(self.root, text="② 当前矩阵  [ A | I ]")
        mid.pack(fill="both", expand=True, padx=10, pady=5)

        self.matrix_label = tk.Text(
            mid, height=14, font=("Consolas", 13),
            state="disabled", bg="#FAFAFA"
        )
        self.matrix_label.pack(fill="both", expand=True, padx=5, pady=5)

        # 说明区
        desc_frame = ttk.LabelFrame(self.root, text="③ 操作说明")
        desc_frame.pack(fill="x", padx=10, pady=5)

        self.desc_var = tk.StringVar(value="请点击「计算」开始。")
        ttk.Label(
            desc_frame, textvariable=self.desc_var,
            wraplength=920, justify="left",
            font=("Microsoft YaHei", 11)
        ).pack(anchor="w", padx=8, pady=6)

        # 进度
        self.progress_var = tk.StringVar(value="0 / 0")
        ttk.Label(self.root, textvariable=self.progress_var,
                  font=("Microsoft YaHei", 10)).pack()

        # 控制按钮
        ctrl = ttk.Frame(self.root)
        ctrl.pack(pady=8)

        ttk.Button(ctrl, text="上一步", width=10,
                   command=self.on_prev).pack(side="left", padx=4)
        ttk.Button(ctrl, text="下一步", width=10,
                   command=self.on_next).pack(side="left", padx=4)
        ttk.Button(ctrl, text="自动播放", width=10,
                   command=self.on_play).pack(side="left", padx=4)
        ttk.Button(ctrl, text="暂停", width=10,
                   command=self.on_pause).pack(side="left", padx=4)
        ttk.Button(ctrl, text="回到开头", width=10,
                   command=self.on_reset).pack(side="left", padx=4)

        # 结果
        self.result_var = tk.StringVar(value="")
        ttk.Label(
            self.root, textvariable=self.result_var,
            font=("Consolas", 11), justify="left"
        ).pack(anchor="w", padx=15, pady=8)

    # ---------------- 解析输入 ----------------
    def parse_input(self):
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            raise MatrixError("请输入矩阵，例如：\n2 1\n1 1")

        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            row = []
            for p in parts:
                try:
                    row.append(float(p))
                except ValueError:
                    raise MatrixError(
                        f"第 {line_no} 行里有无法识别的数字：{p!r}"
                    )
            rows.append(row)
        return rows

    # ---------------- 渲染步骤 ----------------
    def render(self):
        if not self.steps:
            return

        step = self.steps[self.current]
        mat = step["matrix"]
        n = len(mat)

        lines = []
        for i in range(n):
            left = " ".join(f"{mat[i][c]:>10.4f}" for c in range(n))
            right = " ".join(f"{mat[i][c]:>10.4f}" for c in range(n, 2 * n))
            marker = "▶" if i in step["highlight"] else " "
            lines.append(f"{marker} [ {left} ]   |   [ {right} ]")

        self.matrix_label.config(state="normal")
        self.matrix_label.delete("1.0", "end")
        self.matrix_label.insert("1.0", "\n".join(lines))
        self.matrix_label.tag_remove("hl", "1.0", "end")
        for i in step["highlight"]:
            self.matrix_label.tag_add("hl", f"{i + 1}.0", f"{i + 1}.end")
        self.matrix_label.tag_config("hl", background="#FFF59D")
        self.matrix_label.config(state="disabled")

        self.desc_var.set(step["description"])
        self.progress_var.set(
            f"第 {self.current + 1} 步 / 共 {len(self.steps)} 步"
        )

    # ---------------- 按钮回调 ----------------
    def on_calc(self):
        self.on_pause()

        try:
            A = self.parse_input()
        except MatrixError as e:
            messagebox.showerror("输入错误", str(e))
            return

        try:
            A_inv, steps = inverse_matrix(A)
        except NonSquareMatrixError as e:
            messagebox.showerror("不是方阵", str(e))
            return
        except SingularMatrixError as e:
            messagebox.showerror("无法求逆", str(e))
            return
        except InvalidElementError as e:
            messagebox.showerror("元素不合法", str(e))
            return
        except MatrixError as e:
            messagebox.showerror("输入错误", str(e))
            return
        except Exception as e:
            messagebox.showerror("未知错误", f"程序出错了：{e}")
            return

        self.steps = steps
        self.current = 0
        self.render()

        lines = ["逆矩阵 A⁻¹："]
        for row in A_inv:
            lines.append("   [ " + "  ".join(f"{v: .6f}" for v in row) + " ]")

        err, ok = verify_inverse(A, A_inv)
        lines.append("")
        lines.append(f"验证：A × A⁻¹ 与单位矩阵的最大误差 = {err:.3e}")
        lines.append("结果：" + ("通过 ✓" if ok else "误差偏大 ⚠"))
        self.result_var.set("\n".join(lines))

    def on_prev(self):
        if self.steps and self.current > 0:
            self.current -= 1
            self.render()

    def on_next(self):
        if self.steps and self.current < len(self.steps) - 1:
            self.current += 1
            self.render()

    def on_play(self):
        if not self.steps:
            return
        self.playing = True
        self._play_step()

    def _play_step(self):
        if not self.playing:
            return
        if self.current < len(self.steps) - 1:
            self.current += 1
            self.render()
            self.play_job = self.root.after(700, self._play_step)
        else:
            self.playing = False

    def on_pause(self):
        self.playing = False
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def on_reset(self):
        self.on_pause()
        if self.steps:
            self.current = 0
            self.render()

    def on_clear(self):
        self.on_pause()
        self.input_text.delete("1.0", "end")
        self.matrix_label.config(state="normal")
        self.matrix_label.delete("1.0", "end")
        self.matrix_label.config(state="disabled")
        self.steps = []
        self.current = 0
        self.desc_var.set("请点击「计算」开始。")
        self.result_var.set("")
        self.progress_var.set("0 / 0")

    def on_example_2(self):
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", "2 1\n1 1")

    def on_example_3(self):
        self.input_text.delete("1.0", "end")
        self.input_text.insert("1.0", "2 1 0\n1 3 1\n0 1 2")

    def on_self_test(self):
        """点击「自测」按钮：跑所有内置测试，弹窗显示结果"""
        self.on_pause()
        passed, total, results = run_self_tests()

        lines = [f"共 {total} 个测试，通过 {passed} 个，失败 {total - passed} 个。", ""]
        for name, ok, msg in results:
            mark = "✓" if ok else "✗"
            lines.append(f"  {mark}  {name}")
            if not ok:
                lines.append(f"       {msg}")

        text = "\n".join(lines)
        self.result_var.set(text)

        if passed == total:
            messagebox.showinfo("自测结果", f"全部通过！({passed}/{total})")
        else:
            messagebox.showwarning(
                "自测结果",
                f"有 {total - passed} 个失败，详见下方结果区。"
            )


# ============================================================
# 第六部分：程序入口
# ============================================================

def main():
    root = tk.Tk()
    MatrixInverseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()