"""token 下标 <-> 文本：子词合并、上下文窗口、【】标注。"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple


def encode_with_offsets(tokenizer, text: str):
    """编码并返回 (input_ids, offset_mapping)（单条、不加 batch）。

    offset_mapping[i] = (start_char, end_char) 在原文本中的字符区间，(0,0)=特殊 token。
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    return enc["input_ids"], enc["offset_mapping"]


# CJK/全角标点：对中文等无空格语言作为词边界，避免整段被合并成一个"词"。
# 不含 ASCII 标点，以免拆坏英文数学里的 0.25、12*(3/4) 等。
_CJK_PUNCT = set(
    "，。、；：？！…—～·　"
    "「」『』（）〈〉《》【】〔〕〖〗"
    "“”‘’"
)

# 语义分割标点：尖峰词向后延伸时的停止点（中英文句子/子句标点）
_SEGMENT_PUNCT = _CJK_PUNCT | set(".,!?;:")


def _is_word_boundary(ch: str) -> bool:
    return ch.isspace() or ch in _CJK_PUNCT


def _is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF


def _has_cjk(text: str) -> bool:
    return any(_is_cjk_char(ch) for ch in text)


@lru_cache(maxsize=256)
def _cjk_segments(text: str):
    """jieba 分词，返回 (start, end) 元组列表；jieba 不可用时返回 None。"""
    try:
        import jieba
    except ImportError:
        return None
    return tuple((start, end) for _w, start, end in jieba.tokenize(text))


def _cjk_word_span(text: str, s: int, e: int):
    """返回包含 token 起点 s 的 jieba 词 [start, end)；找不到返回 None。"""
    segs = _cjk_segments(text)
    if segs is None:
        return None
    for start, end in segs:
        if start <= s < end:
            return start, end
    return None


def word_span(text: str, offsets: List[Tuple[int, int]], idx: int) -> Tuple[int, int]:
    """把 token idx 的字符区间扩展成"完整单词"的字符区间（合并子词）。

    以该 token 的字符区间为起点，向左右扩展到边界（空白或 CJK 标点；会连带紧邻 ASCII 标点）。
    对中文等 CJK 文本，用 jieba 分词定位尖峰所在的词，达到与英文相同的词级粒度。
    注意：GPT-2/Qwen 系 BPE 会把词首空格并入 offset，先剥掉 token 自身的前后空白。
    """
    s, e = offsets[idx]
    if e <= s:
        return s, e                      # 特殊 token，无法定位
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if e <= s:
        # token 全是空白：返回零宽区间，下游据此跳过（不产生【】等空标注）
        return s, s
    # 若 token 本身全是标点，不扩展，只返回它自己
    if all(_is_word_boundary(ch) for ch in text[s:e]):
        return s, e
    # 中文等 CJK：用 jieba 分词定位尖峰所在的词（词级粒度）
    if _has_cjk(text):
        span = _cjk_word_span(text, s, e)
        if span is not None:
            return span
    # 英文等：向左右扩展到空白/CJK 标点边界
    while s > 0 and not _is_word_boundary(text[s - 1]):
        s -= 1
    while e < len(text) and not _is_word_boundary(text[e]):
        e += 1
    return s, e


def context_text(tokenizer, input_ids: List[int], idx: int,
                 n_before: int = 8, n_after: int = 8) -> str:
    """尖峰附近的文本片段（合并子词展示）。"""
    lo = max(0, idx - n_before)
    hi = min(len(input_ids), idx + n_after + 1)
    return tokenizer.decode(input_ids[lo:hi])


def peak_label(text: str, offsets: List[Tuple[int, int]], idx: int) -> str:
    """尖峰 token 的可读标签。

    有实质内容时返回合并后的词；token 为纯空白/换行时返回 <空白>/<换行>；
    特殊 token（offset 为空）或越界时返回 <t{idx}>。
    """
    if not (0 <= idx < len(offsets)):
        return f"<t{idx}>"
    s, e = word_span(text, offsets, idx)
    if e > s:
        return text[s:e]
    rs, re = offsets[idx]
    if re > rs:
        raw = text[rs:re]
        if "\n" in raw or "\r" in raw:
            return "<换行>"
        if raw.strip() == "":
            return "<空白>"
    return f"<t{idx}>"


def _extend_span(text: str, s: int, e: int, span_max_len: int, span_min_len: int) -> int:
    """把尖峰词 [s,e) 向后延伸，返回延伸后的结束位置。

    停止条件（在 span_max_len 字符内）：换行、语义分割标点、文本末尾。
    若 span_max_len 内都没遇到，则向后标 span_min_len 字符。
    （"下一个尖峰词"的链式处理在 annotate 里做，不在这里。）
    """
    limit = e + span_max_len
    end = e
    stopped = False
    while end < len(text) and end < limit:
        ch = text[end]
        if ch in "\r\n" or ch in _SEGMENT_PUNCT:
            stopped = True
            break
        end += 1
    if end >= len(text):               # 到达文本末尾也算自然停止（等同换行/段末）
        stopped = True
    if not stopped:
        end = min(e + span_min_len, len(text))
    return end


def annotate(text: str, offsets: List[Tuple[int, int]],
             peak_indices: List[int], span_extend: bool = False,
             span_max_len: int = 30, span_min_len: int = 8,
             raw_span: bool = False) -> str:
    """在原文本里把每个尖峰所在词（或延伸后的枢轴 span）用【】括起来。

    raw_span=True 时只标尖峰 token 本身的字符，关闭词合并/延伸等一切修饰。
    否则：多个尖峰落在同一词上会去重；重叠的【】按首次出现的词合并；
    span_extend=True 时优先把 span_max_len 内的下一个尖峰词作为末尾（并入它、跨过标点，
    该词不再作头部），否则向后延伸到换行/标点/文本末尾，未遇则标 span_min_len。
    """
    spans = []
    for idx in peak_indices:
        if 0 <= idx < len(offsets):
            if raw_span:
                s, e = offsets[idx]                   # 只标 token 本身
            else:
                s, e = word_span(text, offsets, idx)  # 合并成词
            if e > s and not text[s:e].isspace():
                spans.append((s, e))

    spans = sorted(set(spans))

    if span_extend and not raw_span and spans:
        extended = []
        i = 0
        while i < len(spans):
            s, e = spans[i]
            # 规则1：max_len 内若匹配到下一个尖峰词，优先它作为枢轴末尾（并入它，跨过中间标点）
            if i + 1 < len(spans) and spans[i + 1][0] <= e + span_max_len:
                extended.append((s, spans[i + 1][1]))
                i += 2   # 规则2：该尖峰词已作末尾，跳过、不再作头部
            else:
                extended.append((s, _extend_span(text, s, e, span_max_len, span_min_len)))
                i += 1
        spans = extended

    if raw_span:
        merged = spans                              # 不做合并，每个 token 单独标
    else:
        merged = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

    out = text
    for s, e in reversed(merged):        # 从后往前插入，避免下标偏移
        out = out[:s] + "【" + out[s:e] + "】" + out[e:]
    return out
