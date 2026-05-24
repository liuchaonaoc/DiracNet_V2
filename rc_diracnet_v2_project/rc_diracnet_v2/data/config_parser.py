"""Electron configuration string parsing.

对应 frame_RC_V1_1.md §4.4 config_parser.py。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

# l 字符 → 数字
_L_LETTER_TO_INT = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
_L_INT_TO_LETTER = {v: k for k, v in _L_LETTER_TO_INT.items()}


@dataclass(frozen=True)
class Shell:
    """单个壳层 (n, l, j, occ)。

    Notes
    -----
    在相对论 j-j 耦合下，每个 (n, l) 拆成 j = l - 1/2 和 j = l + 1/2 两个 sub-shell。
    本类型把它们当作独立 Shell 处理；如果输入字符串只给 nl（非相对论），
    则默认按统计平均占据数分配到两个 j 子壳层。
    """

    n: int
    l: int
    j: float
    occ: int

    @property
    def kappa(self) -> int:
        """Dirac κ 量子数：l =  j - 1/2 时 κ =  l ；l = j + 1/2 时 κ = -(l+1)。"""
        if self.j == self.l - 0.5:
            return self.l
        return -(self.l + 1)


_SHELL_PATTERN = re.compile(
    r"(?P<n>\d+)(?P<l>[spdfghi])(?P<j>\d+/2)?(?P<occ>\d+)?"
)


def parse_config_string(s: str) -> list[Shell]:
    """例如 ``"1s2 2s2 2p6 3s1"`` → [Shell(...), ...]。

    相对论扩展支持 ``"2p1/22 2p3/24"`` 风格（``2p1/2^2`` 与 ``2p3/2^4``）。
    若 j 信息缺失，则按 ``2(2l+1)`` 计算两个 j 子壳层的统计占据。
    """
    s = s.strip()
    if not s:
        return []
    tokens = s.split()
    shells: list[Shell] = []
    for tok in tokens:
        m = _SHELL_PATTERN.fullmatch(tok)
        if not m:
            raise ValueError(f"无法解析壳层 token: {tok!r}")
        n = int(m.group("n"))
        l_char = m.group("l")
        l = _L_LETTER_TO_INT[l_char]
        occ_str = m.group("occ") or "0"
        occ = int(occ_str)
        j_str = m.group("j")
        if j_str is not None:
            j_num = int(j_str.split("/")[0])
            j_val = j_num / 2.0
            shells.append(Shell(n=n, l=l, j=j_val, occ=occ))
        else:
            if l == 0:
                shells.append(Shell(n=n, l=0, j=0.5, occ=occ))
            else:
                cap_low = 2 * l         # j = l - 1/2
                cap_high = 2 * l + 2    # j = l + 1/2
                if occ <= cap_low:
                    shells.append(Shell(n=n, l=l, j=l - 0.5, occ=occ))
                else:
                    shells.append(Shell(n=n, l=l, j=l - 0.5, occ=cap_low))
                    shells.append(Shell(n=n, l=l, j=l + 0.5, occ=min(occ - cap_low, cap_high)))
    return shells


def encode_config_to_tensor(
    shells: Sequence[Shell],
    max_seq: int,
    pad_value: int = 0,
) -> Tensor:
    """壳层序列 → [max_seq, 4] LongTensor (n, l, 2j, occ)。"""
    out = torch.full((max_seq, 4), pad_value, dtype=torch.long)
    for i, sh in enumerate(shells[:max_seq]):
        out[i, 0] = sh.n
        out[i, 1] = sh.l
        out[i, 2] = int(2 * sh.j)
        out[i, 3] = sh.occ
    return out


def shell_to_string(sh: Shell) -> str:
    return f"{sh.n}{_L_INT_TO_LETTER[sh.l]}{int(2*sh.j)}/2^{sh.occ}"


def config_to_string(shells: Sequence[Shell]) -> str:
    return " ".join(shell_to_string(s) for s in shells)
