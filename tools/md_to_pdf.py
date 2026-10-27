"""기획서.md → PDF 변환. 이 문서가 쓰는 문법(제목, 목록, 표, 코드, 인용, 굵게, 인라인 코드)만 지원한다.

사용법 (프로젝트 폴더에서): python tools\\md_to_pdf.py 기획서.md 기획서.pdf
필요: reportlab, Windows 맑은 고딕 글꼴. 출력 PDF가 뷰어에서 열려 있으면 저장이 실패하니 닫고 실행한다.
"""
import re
import sys
import hashlib
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted

# 이 파이썬의 openssl md5는 usedforsecurity 인자를 받지 않아 reportlab 내부 호출을 우회
pdfdoc.md5 = lambda *a, **k: hashlib.md5(*a)

SRC, OUT = sys.argv[1], sys.argv[2]

pdfmetrics.registerFont(TTFont("MG", r"C:\Windows\Fonts\malgun.ttf"))
pdfmetrics.registerFont(TTFont("MG-B", r"C:\Windows\Fonts\malgunbd.ttf"))
pdfmetrics.registerFontFamily("MG", normal="MG", bold="MG-B", italic="MG", boldItalic="MG-B")

ACCENT = colors.HexColor("#2563eb")
SOFT = colors.HexColor("#e8efff")
MUTED = colors.HexColor("#5b636d")
LINE = colors.HexColor("#d5d9de")
CODE_BG = colors.HexColor("#f1f3f5")

title = ParagraphStyle("t", fontName="MG-B", fontSize=20, leading=26)
sub = ParagraphStyle("s", fontName="MG", fontSize=10.5, leading=15, textColor=MUTED, spaceAfter=6)
h2 = ParagraphStyle("h2", fontName="MG-B", fontSize=12.5, leading=17, textColor=ACCENT,
                    spaceBefore=12, spaceAfter=5, keepWithNext=1)
h3 = ParagraphStyle("h3", fontName="MG-B", fontSize=10.5, leading=15, spaceBefore=8, spaceAfter=3, keepWithNext=1)
body = ParagraphStyle("b", fontName="MG", fontSize=9.5, leading=14.5, spaceAfter=3)
bul = ParagraphStyle("bl", parent=body, leftIndent=12, bulletIndent=2, bulletFontName="MG", spaceAfter=1)
cell = ParagraphStyle("c", parent=body, fontSize=8.8, leading=12.5, spaceAfter=0)
cellb = ParagraphStyle("cb", parent=cell, fontName="MG-B")
note = ParagraphStyle("n", parent=body, fontSize=8.5, leading=12, textColor=MUTED)
code = ParagraphStyle("code", fontName="MG", fontSize=8, leading=11.5)

W = A4[0] - 36 * mm


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"`(.+?)`", r"<font color='#b4235a'>\1</font>", t)


def col_widths(rows):
    # 짧은 열(기간·담당 등)은 가장 긴 칸의 실제 글자 폭에 맞춰 고정하고,
    # 남은 폭을 긴 열에 글자 수 비율로 나눈다. 비율만 쓰면 짧은 열이 한 글자씩 줄바꿈된다.
    n = len(rows[0])
    plain = lambda c: re.sub(r"\*\*|`", "", c)
    natural = [max(pdfmetrics.stringWidth(plain(r[i]), "MG-B", cell.fontSize) for r in rows) + 12
               for i in range(n)]
    fixed = {i: natural[i] for i in range(n) if natural[i] < W / n * 0.6}
    lens = {i: max(min(len(plain(r[i])), 60) for r in rows) for i in range(n) if i not in fixed}
    rest = W - sum(fixed.values())
    total = sum(lens.values())
    return [fixed[i] if i in fixed else rest * lens[i] / total for i in range(n)]


def make_table(rows):
    header, body_rows = rows[0], rows[1:]
    data = [[Paragraph(inline(c), cellb) for c in header]]
    spans = []
    for r, row in enumerate(body_rows, start=1):
        # 둘째 칸 이후가 비어 있는 행은 둘째 칸부터 끝까지 합침 (일정표의 전원 체크포인트 행)
        if len(row) > 2 and row[1].strip() and all(not c.strip() for c in row[2:]):
            spans.append(r)
        data.append([Paragraph(inline(c), cell) for c in row])
    t = Table(data, colWidths=col_widths(rows), repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("BACKGROUND", (0, 0), (-1, 0), SOFT),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    for r in spans:
        st += [("SPAN", (1, r), (-1, r)), ("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff7e6"))]
    t.setStyle(TableStyle(st))
    return t


def code_block(lines):
    t = Table([[Preformatted("\n".join(lines), code)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    return t


def parse(md):
    story, lines, i = [], md.splitlines(), 0
    seen_title = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            if j == len(lines):
                sys.exit(f"닫히지 않은 코드 블록: {i + 1}번째 줄의 ``` 를 닫아 주세요")
            if story and isinstance(story[-1], Paragraph):
                story[-1].keepWithNext = 1  # 코드를 소개하는 문단이 코드와 다른 쪽에 남지 않게
            story += [code_block(lines[i + 1:j]), Spacer(1, 4)]
            i = j + 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):  # 구분선(|---|) 제외
                    rows.append(cells)
                i += 1
            story += [make_table(rows), Spacer(1, 4)]
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), title))
            seen_title = True
            i += 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), h2))
            i += 1
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), h3))
            i += 1
        elif line.startswith("- "):
            story.append(Paragraph(inline(line[2:]), bul, bulletText="•"))
            i += 1
        elif line.startswith("> "):
            # 제목 바로 뒤 인용은 부제, 그 외는 주석
            style = sub if seen_title and len(story) == 1 else note
            story.append(Paragraph(inline(line[2:]), style))
            i += 1
        elif line.strip():
            story.append(Paragraph(inline(line), body))
            i += 1
        else:
            i += 1
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("MG", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, f"얼굴 자동 모자이크 프로젝트 기획서  |  {doc.page}")
    canvas.restoreState()


with open(SRC, encoding="utf-8") as fp:
    story = parse(fp.read())
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=18 * mm,
                        title="YOLO 기반 얼굴 자동 모자이크 프로젝트 기획서")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("OK pages:", doc.page)
