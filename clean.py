#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gutenberg Ebook Cleaner (Poetry-Aware Edition)
==============================================
从古登堡计划下载的纯文本电子书中清洗出正文内容。
智能识别并保护诗歌格式，避免将诗行错误合并。

功能：
1. 自动识别并移除 PG 标准头部/尾部
2. 智能诗歌检测与格式保护
3. 清理多余空行、页码、插图标记等
4. 仅合并散文段落的硬换行，保留诗歌断行
5. 支持批量处理和多种编码自动检测

使用方法：
    python gutenberg_cleaner_poetry.py --input ./raw_books/ --output ./cleaned_books/
"""

import os
import re
import argparse
import chardet
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


# ==================== 配置：头部/尾部识别标记 ====================

START_MARKERS = [
    r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG",
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK",
    r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK",
    r"^\s*Produced by .+?\n\s*\n",
    r"^\s*Transcribed from .+?\n\s*\n",
]

END_MARKERS = [
    r"\*\*\*\s*END OF (THE\s+)?PROJECT GUTENBERG",
    r"END OF (THE\s+)?PROJECT GUTENBERG EBOOK",
    r"End of (the\s+)?Project Gutenberg",
    r"End of the Project Gutenberg Etext",
    r"More information about this book is at the top of this file",
]

HEADER_METADATA_PATTERNS = [
    r"^Title:\s*",
    r"^Author:\s*",
    r"^Release Date:\s*",
    r"^Language:\s*",
    r"^Character set encoding:\s*",
    r"^Produced by",
    r"^Transcribed from",
    r"^This eBook is for the use of",
    r"^The Project Gutenberg [Ee]Book",
    r"^Project Gutenberg's",
]

SKIP_LINE_PATTERNS = [
    r"^\s*\[Illustration.*?\]\s*$",
    r"^\s*\[Footnote.*?\]\s*$",
    r"^\s*\[Pg\s+\d+\]\s*$",
    r"^\s*Page\s+\d+\s*$",
    r"^\s*\d+\s*$",
    r"^\s*_{3,}\s*$",
    r"^\s*={3,}\s*$",
    r"^\s*\*{3,}\s*$",
]


# ==================== 诗歌检测配置 ====================

@dataclass
class PoetryConfig:
    """诗歌检测参数配置"""
    min_consecutive_lines: int = 2      # 至少连续几行才被认为是诗歌
    max_line_length: int = 55           # 诗歌行最大长度（散文行通常更长）
    indent_threshold: int = 2           # 行首至少几个空格算缩进
    rhyme_punctuation_bonus: float = 0.3  # 行尾有韵脚标点的加分权重
    short_line_ratio: float = 0.7       # 段落中短行占比阈值
    max_stanza_gap: int = 1             # 诗节之间最多允许几个空行


# ==================== 核心清洗类 ====================

class GutenbergCleaner:
    def __init__(
        self,
        remove_illustrations: bool = True,
        remove_page_numbers: bool = True,
        remove_footnotes: bool = False,
        merge_line_breaks: bool = False,
        max_consecutive_blank_lines: int = 2,
        encoding: Optional[str] = None,
        protect_poetry: bool = True,
        poetry_config: Optional[PoetryConfig] = None,
    ):
        self.remove_illustrations = remove_illustrations
        self.remove_page_numbers = remove_page_numbers
        self.remove_footnotes = remove_footnotes
        self.merge_line_breaks = merge_line_breaks
        self.max_consecutive_blank_lines = max_consecutive_blank_lines
        self.encoding = encoding
        self.protect_poetry = protect_poetry
        self.poetry_config = poetry_config or PoetryConfig()

        self.start_regex = [re.compile(p, re.IGNORECASE) for p in START_MARKERS]
        self.end_regex = [re.compile(p, re.IGNORECASE) for p in END_MARKERS]

        skip_patterns = []
        if remove_illustrations:
            skip_patterns.append(SKIP_LINE_PATTERNS[0])
        if remove_footnotes:
            skip_patterns.append(SKIP_LINE_PATTERNS[1])
        if remove_page_numbers:
            skip_patterns.extend(SKIP_LINE_PATTERNS[2:6])
        skip_patterns.extend(SKIP_LINE_PATTERNS[6:])
        
        self.skip_regex = [re.compile(p, re.IGNORECASE) for p in skip_patterns]

    def detect_encoding(self, filepath: str) -> str:
        """自动检测文件编码"""
        if self.encoding:
            return self.encoding
        
        with open(filepath, "rb") as f:
            raw = f.read(100000)
            result = chardet.detect(raw)
            detected = result.get("encoding", "utf-8")
            confidence = result.get("confidence", 0)
            
            if detected and confidence > 0.5:
                if detected.lower() in ("ascii", "iso-8859-1", "windows-1252"):
                    return "utf-8"
                return detected
            return "utf-8"

    def find_body_boundaries(self, lines: List[str]) -> Tuple[int, int]:
        """查找正文起始和结束位置"""
        n = len(lines)
        start_idx = 0
        end_idx = n

        for i, line in enumerate(lines):
            for regex in self.start_regex:
                if regex.search(line):
                    start_idx = i + 1
                    while start_idx < n and (
                        lines[start_idx].strip() == "" 
                        or any(re.match(p, lines[start_idx]) for p in HEADER_METADATA_PATTERNS)
                    ):
                        start_idx += 1
                    break
            if start_idx > 0:
                break

        if start_idx == 0:
            last_header_line = -1
            for i, line in enumerate(lines[:min(200, n)]):
                if any(re.match(p, line) for p in HEADER_METADATA_PATTERNS):
                    last_header_line = i
            
            if last_header_line >= 0:
                for i in range(last_header_line + 1, min(last_header_line + 20, n)):
                    if lines[i].strip() != "":
                        start_idx = i
                        break

        for i in range(n - 1, start_idx, -1):
            for regex in self.end_regex:
                if regex.search(lines[i]):
                    end_idx = i
                    break
            if end_idx < n:
                break

        return start_idx, end_idx

    def _is_poetry_line(self, line: str) -> Tuple[bool, float]:
        """
        判断单行是否具有诗歌特征。
        返回: (是否是诗歌行, 置信度分数 0-1)
        """
        cfg = self.poetry_config
        stripped = line.rstrip()
        if not stripped:
            return False, 0.0

        # 特征1: 行首缩进
        leading_spaces = len(line) - len(line.lstrip())
        has_indent = leading_spaces >= cfg.indent_threshold
        
        # 特征2: 行长度
        line_length = len(stripped)
        is_short = line_length <= cfg.max_line_length
        
        # 特征3: 行尾标点（诗歌常以逗号、分号、无标点结尾，而非句号）
        # 但全大写的标题行不算
        trailing_punct = stripped[-1] if stripped else ''
        has_poetic_punct = trailing_punct in ',;:-!?…' or trailing_punct.isalpha()
        is_all_caps = stripped.isupper() and len(stripped) > 3
        
        score = 0.0
        if has_indent:
            score += 0.4
        if is_short:
            score += 0.35
        if has_poetic_punct and not is_all_caps:
            score += cfg.rhyme_punctuation_bonus
        
        # 如果行很短且有缩进，几乎肯定是诗歌
        if is_short and has_indent:
            score = max(score, 0.9)
        
        is_poetry = score >= 0.6 or (is_short and has_indent)
        return is_poetry, score

    def detect_poetry_regions(self, lines: List[str]) -> List[Tuple[int, int]]:
        """
        检测文本中所有诗歌区域。
        返回: [(start_idx, end_idx), ...] 的列表，边界为排他性。
        """
        if not self.protect_poetry:
            return []

        cfg = self.poetry_config
        n = len(lines)
        poetry_regions = []
        i = 0

        while i < n:
            # 寻找诗歌起始点：连续多行具有诗歌特征
            poetry_start = -1
            consecutive_poetry = 0
            temp_start = i

            for j in range(i, n):
                line = lines[j]
                is_poetry, score = self._is_poetry_line(line)
                
                # 空行在诗歌区域内是允许的（诗节间隔），但连续太多空行就中断
                if line.strip() == "":
                    if consecutive_poetry >= cfg.min_consecutive_lines:
                        # 可能是诗节间隔，继续观察
                        gap_count = 0
                        k = j
                        while k < n and lines[k].strip() == "":
                            gap_count += 1
                            k += 1
                        if gap_count <= cfg.max_stanza_gap and k < n:
                            next_is_poetry, _ = self._is_poetry_line(lines[k])
                            if next_is_poetry:
                                j = k - 1  # 跳过空行，继续检测
                                continue
                    # 空行中断诗歌
                    if consecutive_poetry >= cfg.min_consecutive_lines:
                        poetry_regions.append((temp_start, j))
                    consecutive_poetry = 0
                    poetry_start = -1
                    temp_start = j + 1
                    continue

                if is_poetry:
                    consecutive_poetry += 1
                    if poetry_start == -1:
                        poetry_start = j
                        temp_start = j
                else:
                    # 遇到非诗歌行
                    if consecutive_poetry >= cfg.min_consecutive_lines:
                        # 检查是否是散文中的短行（如对话），通过上下文判断
                        # 如果前后都是长行，中间夹一个短行，可能不是诗歌
                        if consecutive_poetry >= 3:
                            poetry_regions.append((temp_start, j))
                    consecutive_poetry = 0
                    poetry_start = -1
                    temp_start = j + 1

            # 处理文件末尾的诗歌
            if consecutive_poetry >= cfg.min_consecutive_lines:
                poetry_regions.append((temp_start, n))
            
            i = n  # 一次性扫描完

        # 合并相邻或重叠的区域
        if not poetry_regions:
            return []
        
        merged = [poetry_regions[0]]
        for start, end in poetry_regions[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + cfg.max_stanza_gap + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged

    def clean_lines(self, lines: List[str]) -> List[str]:
        """清洗行内容"""
        cleaned = []
        blank_count = 0

        for line in lines:
            line = line.rstrip()

            should_skip = False
            for regex in self.skip_regex:
                if regex.search(line):
                    should_skip = True
                    break
            if should_skip:
                continue

            if line == "":
                blank_count += 1
                if blank_count > self.max_consecutive_blank_lines:
                    continue
            else:
                blank_count = 0

            cleaned.append(line)

        return cleaned

    def merge_hard_line_breaks(self, lines: List[str]) -> List[str]:
        """
        合并散文段落的硬换行，但保护诗歌区域。
        """
        if not self.merge_line_breaks:
            return lines

        poetry_regions = self.detect_poetry_regions(lines)
        poetry_set = set()
        for start, end in poetry_regions:
            for idx in range(start, end):
                poetry_set.add(idx)

        merged = []
        paragraph_buffer = []
        i = 0
        n = len(lines)

        while i < n:
            # 如果是诗歌区域，先清空散文缓冲，然后原样输出诗歌
            if i in poetry_set:
                if paragraph_buffer:
                    merged.append(" ".join(paragraph_buffer))
                    paragraph_buffer = []
                
                # 输出诗歌块直到区域结束
                poetry_start = i
                while i < n and i in poetry_set:
                    merged.append(lines[i].rstrip())
                    i += 1
                continue

            line = lines[i]
            if line.strip() == "":
                if paragraph_buffer:
                    merged.append(" ".join(paragraph_buffer))
                    paragraph_buffer = []
                merged.append(line)
                i += 1
            else:
                # 判断是否是新段落
                stripped = line.lstrip()
                is_new_paragraph = (
                    line.startswith("  ") or
                    line.startswith("\t") or
                    (len(stripped) < 20 and stripped.endswith(".")) or
                    stripped.isupper()
                )
                
                if is_new_paragraph and paragraph_buffer:
                    merged.append(" ".join(paragraph_buffer))
                    paragraph_buffer = []
                
                paragraph_buffer.append(line.strip())
                i += 1

        if paragraph_buffer:
            merged.append(" ".join(paragraph_buffer))

        return merged

    def process_file(self, input_path: str, output_path: str) -> Dict:
        """处理单个文件"""
        result = {
            "input": input_path,
            "output": output_path,
            "status": "success",
            "encoding": None,
            "original_lines": 0,
            "cleaned_lines": 0,
            "poetry_regions_found": 0,
            "message": "",
        }

        try:
            encoding = self.detect_encoding(input_path)
            result["encoding"] = encoding
            
            with open(input_path, "r", encoding=encoding, errors="replace") as f:
                lines = f.read().splitlines()
            
            result["original_lines"] = len(lines)

            start_idx, end_idx = self.find_body_boundaries(lines)
            body_lines = lines[start_idx:end_idx]

            # 统计诗歌区域
            if self.protect_poetry:
                poetry_regions = self.detect_poetry_regions(body_lines)
                result["poetry_regions_found"] = len(poetry_regions)

            cleaned_lines = self.clean_lines(body_lines)
            
            if self.merge_line_breaks:
                cleaned_lines = self.merge_hard_line_breaks(cleaned_lines)

            while cleaned_lines and cleaned_lines[0].strip() == "":
                cleaned_lines.pop(0)
            while cleaned_lines and cleaned_lines[-1].strip() == "":
                cleaned_lines.pop()

            result["cleaned_lines"] = len(cleaned_lines)

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(cleaned_lines) + "\n")

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)

        return result

    def process_directory(self, input_dir: str, output_dir: str, pattern: str = "*.txt"):
        """批量处理目录"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob(pattern))
        if not files:
            print(f"⚠️  在 {input_dir} 中未找到匹配 {pattern} 的文件")
            return []

        results = []
        print(f"📚 找到 {len(files)} 个文件，开始清洗...\n")
        
        for i, filepath in enumerate(files, 1):
            out_file = output_path / filepath.name
            print(f"[{i}/{len(files)}] 处理: {filepath.name}")
            
            result = self.process_file(str(filepath), str(out_file))
            results.append(result)

            if result["status"] == "success":
                reduction = (1 - result["cleaned_lines"] / result["original_lines"]) * 100
                poetry_info = ""
                if self.protect_poetry:
                    poetry_info = f" | 诗歌区域: {result['poetry_regions_found']} 处"
                print(f"    ✅ 完成 | 编码: {result['encoding']} | "
                      f"行数: {result['original_lines']} → {result['cleaned_lines']} "
                      f"(减少 {reduction:.1f}%){poetry_info}")
            else:
                print(f"    ❌ 错误: {result['message']}")

        return results


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="清洗古登堡计划电子书（诗歌感知版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础清洗（自动保护诗歌，不合并断行）
  python gutenberg_cleaner_poetry.py -i ./books/ -o ./clean/

  # 合并散文断行，但保留诗歌格式
  python gutenberg_cleaner_poetry.py -i ./books/ -o ./clean/ --merge-lines

  # 关闭诗歌保护（全部按散文处理）
  python gutenberg_cleaner_poetry.py -i ./books/ -o ./clean/ --merge-lines --no-poetry-protect

  # 调整诗歌检测灵敏度（适合古英语或特殊排版）
  python gutenberg_cleaner_poetry.py -i ./books/ -o ./clean/ --merge-lines \
      --poetry-max-len 40 --poetry-min-lines 3
        """
    )
    
    parser.add_argument("-i", "--input", required=True, help="输入文件或目录")
    parser.add_argument("-o", "--output", required=True, help="输出文件或目录")
    parser.add_argument("--pattern", default="*.txt", help="批量处理文件匹配模式")
    parser.add_argument("--encoding", default=None, help="指定编码（默认自动检测）")
    parser.add_argument("--merge-lines", action="store_true", 
                        help="合并散文段落的硬换行（诗歌区域不受影响）")
    parser.add_argument("--remove-footnotes", action="store_true", help="移除脚注标记")
    parser.add_argument("--keep-illustrations", action="store_true", help="保留插图标记")
    parser.add_argument("--keep-page-numbers", action="store_true", help="保留页码标记")
    parser.add_argument("--max-blank", type=int, default=2, help="最多保留连续空行数")
    
    # 诗歌相关参数
    poetry_group = parser.add_argument_group("诗歌检测选项")
    poetry_group.add_argument("--no-poetry-protect", action="store_true", 
                              help="关闭诗歌保护（所有文本按散文处理）")
    poetry_group.add_argument("--poetry-max-len", type=int, default=55,
                              help="诗歌行最大长度阈值 (默认: 55)")
    poetry_group.add_argument("--poetry-min-lines", type=int, default=2,
                              help="最少连续几行才判定为诗歌 (默认: 2)")
    poetry_group.add_argument("--poetry-indent", type=int, default=2,
                              help="行首至少几个空格算缩进 (默认: 2)")
    
    args = parser.parse_args()

    poetry_cfg = PoetryConfig(
        max_line_length=args.poetry_max_len,
        min_consecutive_lines=args.poetry_min_lines,
        indent_threshold=args.poetry_indent,
    )

    cleaner = GutenbergCleaner(
        remove_illustrations=not args.keep_illustrations,
        remove_page_numbers=not args.keep_page_numbers,
        remove_footnotes=args.remove_footnotes,
        merge_line_breaks=args.merge_lines,
        max_consecutive_blank_lines=args.max_blank,
        encoding=args.encoding,
        protect_poetry=not args.no_poetry_protect,
        poetry_config=poetry_cfg,
    )

    input_path = Path(args.input)
    
    if input_path.is_file():
        result = cleaner.process_file(args.input, args.output)
        if result["status"] == "success":
            print(f"\n✅ 清洗完成！")
            print(f"   原始行数: {result['original_lines']}")
            print(f"   清洗后行数: {result['cleaned_lines']}")
            if not args.no_poetry_protect:
                print(f"   检测到诗歌区域: {result['poetry_regions_found']} 处")
            print(f"   输出: {args.output}")
        else:
            print(f"\n❌ 处理失败: {result['message']}")
    else:
        results = cleaner.process_directory(args.input, args.output, args.pattern)
        
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "error")
        total_poetry = sum(r["poetry_regions_found"] for r in results if r["status"] == "success")
        print(f"\n{'='*50}")
        print(f"📊 批量处理完成: 成功 {success} 个, 失败 {failed} 个")
        if not args.no_poetry_protect:
            print(f"📝 共检测到诗歌区域: {total_poetry} 处")
        print(f"📁 输出目录: {args.output}")


if __name__ == "__main__":
    main()
