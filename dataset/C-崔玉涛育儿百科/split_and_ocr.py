#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["TESSDATA_PREFIX"] = "C:/Program Files/Tesseract-OCR/tessdata"
print("TESSDATA_PREFIX =", os.environ["TESSDATA_PREFIX"])
print("chi_sim exists:", os.path.exists("C:/Program Files/Tesseract-OCR/tessdata/chi_sim.traineddata"))

import time
from PyPDF2 import PdfReader, PdfWriter
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
from fpdf import FPDF  # 新增库用于生成文字型PDF

INPUT_FILE = "C-崔玉涛育儿百科.pdf"   # 原始PDF
OUTOUT_FILE_NAME = "split"  # 分割后的 PDF 文件前缀
SUBFILE_PAGE_NUM = 10      # 每份分割页数

# OCR 提取图片PDF中的文字
def extract_text_from_pdf(pdf_path, txt_output_path, pdf_text_output_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="chi_sim")  # 中文用 chi_sim
        all_text += text + "\n"

    # 保存为 .txt
    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    # 保存为 OCR 文字版 PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("ArialUnicode", "", fname="C:\\Windows\\Fonts\\simhei.ttf", uni=True)
    pdf.set_font("ArialUnicode", size=12)

    for line in all_text.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf.output(pdf_text_output_path)

# 分割 PDF 并 OCR 输出文本和文字型 PDF
def subsplit(a, reader, pageCount):
    output_pdf = PdfWriter()
    outfile_pdf = f"{OUTOUT_FILE_NAME}{a}.pdf"
    outfile_txt = f"{OUTOUT_FILE_NAME}{a}.txt"
    outfile_text_pdf = f"{OUTOUT_FILE_NAME}{a}_text.pdf"

    for ipage in range(SUBFILE_PAGE_NUM):
        kp = ipage + SUBFILE_PAGE_NUM * a
        if kp < pageCount:
            output_pdf.add_page(reader.pages[kp])

    with open(outfile_pdf, "wb") as f:
        output_pdf.write(f)

    # OCR识别并输出为 txt 和 PDF
    extract_text_from_pdf(outfile_pdf, outfile_txt, outfile_text_pdf)

def split_and_ocr():
    reader = PdfReader(INPUT_FILE)
    pageCount = len(reader.pages)

    subFileNum = pageCount // SUBFILE_PAGE_NUM
    if pageCount % SUBFILE_PAGE_NUM > 0:
        subFileNum += 1

    for k in range(subFileNum):
        print(f"Processing part {k}...")
        subsplit(k, reader, pageCount)

if __name__ == '__main__':
    start = time.time()
    split_and_ocr()
    end = time.time()
    print("Total time: %.2f seconds" % (end - start))
