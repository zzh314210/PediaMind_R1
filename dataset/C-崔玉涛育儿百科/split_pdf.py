#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
from PyPDF2 import PdfReader, PdfWriter

INPUT_FILE = "C-崔玉涛育儿百科.pdf"  # 原始PDF
OUTPUT_FILE_PREFIX = "split"  # 分割后的 PDF 文件前缀
PAGES_PER_SUBFILE = 10  # 每份分割的页数


# 分割 PDF
def split_pdf(reader, page_count):
    """将PDF按指定页数分割成多个小PDF"""
    subfile_count = page_count // PAGES_PER_SUBFILE
    if page_count % PAGES_PER_SUBFILE > 0:
        subfile_count += 1

    for subfile_index in range(subfile_count):
        print(f"正在处理第 {subfile_index + 1}/{subfile_count} 部分...")
        output_pdf = PdfWriter()

        # 计算当前子文件包含的页码范围
        start_page = subfile_index * PAGES_PER_SUBFILE
        end_page = min(start_page + PAGES_PER_SUBFILE, page_count)

        # 添加页面到输出PDF
        for page_num in range(start_page, end_page):
            output_pdf.add_page(reader.pages[page_num])

        # 保存分割后的PDF
        output_filename = f"./split/{OUTPUT_FILE_PREFIX}{subfile_index}.pdf"
        with open(output_filename, "wb") as f:
            output_pdf.write(f)

        print(f"已保存: {output_filename}")


if __name__ == '__main__':
    start_time = time.time()

    # 读取原始PDF
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到输入文件 '{INPUT_FILE}'")
        exit(1)

    reader = PdfReader(INPUT_FILE)
    page_count = len(reader.pages)

    print(f"开始分割 PDF 文件: {INPUT_FILE}")
    print(f"总页数: {page_count}")
    print(f"每页分割数: {PAGES_PER_SUBFILE}")

    # 执行分割
    split_pdf(reader, page_count)

    end_time = time.time()
    print(f"处理完成! 总耗时: {end_time - start_time:.2f} 秒")