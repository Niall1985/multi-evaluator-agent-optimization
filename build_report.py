"""Generates the BCSE497J capstone project report as a .docx matching the
formatting rules in 'BCE497J PROJECT REPORT TEMPLATE_V1.docx'.

Every experimental number is read from report_assets/report_data.json, which is
produced by report_analysis.py from the raw experiment CSVs. Nothing is hand
typed, so the report can be regenerated after a re-run:

    python report_analysis.py --mock-csv ... --live-csv ... --outdir report_assets
    python build_report.py --assets report_assets --out "Project_Report.docx"

Fill in CANDIDATE below before the final print.
"""
import argparse
import json
import os
from typing import Dict, List, Optional

from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor

# --------------------------------------------------------------------------
# Candidate / institutional details. EDIT THESE, then re-run.
# --------------------------------------------------------------------------
CANDIDATE = {
    "title": "MULTI-OBJECTIVE AGENT OPTIMIZATION THROUGH ADAPTIVE EVALUATION",
    "degree": "Bachelor of Technology in Computer Science and Engineering",
    "program": "B.Tech. Computer Science and Engineering",
    "name": "KUMAR SHREYASH",
    "reg_no": "23BCE1882",
    # Template: "1ST & 2ND PAGE LIST ALL THE TEAM MEMBERS"; the declaration,
    # certificate and acknowledgement carry only the submitting candidate.
    "team": [
        ("SUBHROJYOTI SEN", "23BCE1259"),
        ("KUMAR SHREYASH", "23BCE1882"),
        ("DCUNHA NIALL FRANCIS AJEET", "23BCE1985"),
    ],
    "guide_name": "Sreeja P S",
    "guide_designation": "Assistant Professor Senior Grade 1",
    "hod_name": "Dr. J. Prassanna",
    "school": "SCHOOL OF COMPUTER SCIENCE AND ENGINEERING",
    "institute": "VELLORE INSTITUTE OF TECHNOLOGY, CHENNAI",
    # Running prose uses title case, as the template's own declaration and
    # certificate wording does; ALL CAPS is reserved for the cover and title
    # pages, where the template specifies it explicitly.
    "title_prose": "Multi-Objective Agent Optimization through Adaptive Evaluation",
    "name_prose": "Kumar Shreyash",
    "school_prose": "School of Computer Science and Engineering",
    "institute_prose": "Vellore Institute of Technology, Chennai",
    "month_year": "October, 2026",
    "place": "Chennai",
    # Fills the template's <area> placeholder in the acknowledgement.
    "research_area": "multi-objective optimization of large language model agents",
}

# Template page specification (A4; margins in cm).
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 3.81, 2.54, 2.54, 2.54
BODY_FONT = "Times New Roman"
BODY_SIZE = 12
LINE_SPACING = 1.5

FIG_W = Cm(14.5)        # fits inside the A4 text column given the 3.81 cm binding margin


# --------------------------------------------------------------------------
# low-level docx helpers
# --------------------------------------------------------------------------
def _set_cell_border(cell, **kwargs):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        spec = kwargs.get(edge)
        if spec:
            el = OxmlElement(f"w:{edge}")
            for k, v in spec.items():
                el.set(qn(f"w:{k}"), str(v))
            borders.append(el)
    tc_pr.append(borders)


def _field(paragraph, instr: str):
    """Insert a Word field (used for PAGE numbers and the TOC placeholder)."""
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr_el)
    run._r.append(fld_sep)
    run._r.append(fld_end)


def _page_number_format(section, fmt: str, start: Optional[int] = None):
    """Set the page-number format for a section (lowerRoman for front matter)."""
    sect_pr = section._sectPr
    existing = sect_pr.find(qn("w:pgNumType"))
    if existing is not None:
        sect_pr.remove(existing)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))
    sect_pr.append(pg)


def _footer_page_number(section, link_previous: bool = False):
    footer = section.footer
    footer.is_linked_to_previous = link_previous
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(p, " PAGE ")
    for run in p.runs:
        run.font.name = BODY_FONT
        run.font.size = Pt(BODY_SIZE)


def _configure_section(section):
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = Cm(MARGIN_LEFT)
    section.right_margin = Cm(MARGIN_RIGHT)
    section.top_margin = Cm(MARGIN_TOP)
    section.bottom_margin = Cm(MARGIN_BOTTOM)


class ReportBuilder:
    def __init__(self, assets_dir: str):
        self.assets = assets_dir
        self.figdir = os.path.join(assets_dir, "figures")
        data_path = os.path.join(assets_dir, "report_data.json")
        with open(data_path, encoding="utf-8") as f:
            self.d: Dict = json.load(f)

        self.doc = Document()
        self._init_styles()
        _configure_section(self.doc.sections[0])

        self.figures: List[str] = []    # numbered captions, for the List of Figures
        self.tables: List[str] = []
        self.chapter = ""               # set by chapter_heading; drives auto-numbering
        self._fig_n = 0
        self._tab_n = 0

    # ---------------------------------------------------------------- styles
    def _init_styles(self):
        st = self.doc.styles["Normal"]
        st.font.name = BODY_FONT
        st.font.size = Pt(BODY_SIZE)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
        pf = st.paragraph_format
        pf.line_spacing = LINE_SPACING
        pf.space_after = Pt(6)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        # Word's built-in heading styles are blue Calibri. Override them to the
        # template's specification instead of side-stepping them, so the TOC
        # field has something to collect.
        for name, size, align in (
            ("Heading 1", 16, WD_ALIGN_PARAGRAPH.CENTER),
            ("Heading 2", 14, WD_ALIGN_PARAGRAPH.LEFT),
            ("Heading 3", 12, WD_ALIGN_PARAGRAPH.LEFT),
        ):
            h = self.doc.styles[name]
            h.font.name = BODY_FONT
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.italic = False
            h.font.color.rgb = RGBColor(0, 0, 0)
            h.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
            hpf = h.paragraph_format
            hpf.alignment = align
            hpf.line_spacing = LINE_SPACING
            hpf.space_before = Pt(10)
            hpf.space_after = Pt(6)
            hpf.keep_with_next = True
            # "No paragraph should have its opening line at the bottom of a page."
            hpf.page_break_before = False

        # Dedicated caption styles: the List of Figures and List of Tables are
        # generated from TOC fields keyed to these style names.
        for name in ("FigureCaption", "TableCaption"):
            try:
                cs = self.doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            except ValueError:
                cs = self.doc.styles[name]
            cs.base_style = self.doc.styles["Normal"]
            cs.font.name = BODY_FONT
            cs.font.size = Pt(10)          # template: captions not smaller than 10pt
            cs.font.bold = False
            cs.font.italic = False
            cs.font.color.rgb = RGBColor(0, 0, 0)
            cpf = cs.paragraph_format
            cpf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cpf.line_spacing = 1.0         # template permits single-spaced captions
            cpf.space_after = Pt(10)
            cpf.keep_with_next = (name == "TableCaption")

    # ------------------------------------------------------------- primitives
    def para(self, text: str = "", size: int = BODY_SIZE, bold: bool = False,
             italic: bool = False, align=None, caps: bool = False,
             spacing: Optional[float] = None, space_after: Optional[int] = None,
             underline: bool = False):
        p = self.doc.add_paragraph()
        if align is not None:
            p.alignment = align
        if spacing is not None:
            p.paragraph_format.line_spacing = spacing
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        run = p.add_run(text.upper() if caps else text)
        run.font.name = BODY_FONT
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.underline = underline
        return p

    def body(self, text: str):
        return self.para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    def bullets(self, items: List[str], style: str = "List Bullet"):
        for it in items:
            p = self.doc.add_paragraph(style=style)
            p.paragraph_format.line_spacing = LINE_SPACING
            run = p.add_run(it)
            run.font.name = BODY_FONT
            run.font.size = Pt(BODY_SIZE)

    def logo(self, space_after: int = 6):
        """VIT logo as placed by the template: centred, 8.78 cm wide."""
        path = os.path.join(self.assets, "images", "vit_logo.jpg")
        if not os.path.exists(path):
            print("  ! logo missing, skipped: report_assets/images/vit_logo.jpg")
            return
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.add_run().add_picture(path, width=Cm(8.78))
        return p

    def titlepage_line(self, text: str, size: int, bold: bool = False,
                       italic: bool = False, space_after: int = 6,
                       space_before: int = 0):
        """A centred line on the cover/title pages.

        Single line spacing and explicit gaps, so the whole block is predictable
        and fits the A4 column without spilling onto a second page.
        """
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf = p.paragraph_format
        pf.line_spacing = 1.0
        pf.space_before = Pt(space_before)
        pf.space_after = Pt(space_after)
        run = p.add_run(text)
        run.font.name = BODY_FONT
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        return p

    def signature_block(self, lines, space_before: int = 18, bold_first: int = 0,
                        right_col_cm: float = 8.4):
        """Signature / name / date lines, single spaced.

        The template sets these as compact tabbed lines ("Date:  Signature of the
        Candidate"), not as body text; at 1.5 spacing they overran the page.
        Each entry is (left_text, right_text); a right_text places a second
        column at a tab stop rather than padding with spaces.
        """
        from docx.enum.text import WD_TAB_ALIGNMENT
        for idx, (left, right) in enumerate(lines):
            p = self.doc.add_paragraph()
            pf = p.paragraph_format
            pf.line_spacing = 1.0
            pf.space_before = Pt(space_before if idx == 0 else 0)
            pf.space_after = Pt(4)
            pf.tab_stops.add_tab_stop(Cm(right_col_cm), WD_TAB_ALIGNMENT.LEFT)
            run = p.add_run(left + ("	" + right if right else ""))
            run.font.name = BODY_FONT
            run.font.size = Pt(BODY_SIZE)
            run.bold = idx < bold_first
        return p

    def landscape_figure(self, filename: str, caption: str, width_cm: float = 20.5):
        """Place one figure on its own landscape page.

        A full-width portrait column renders this diagram's labels at about six
        point. Turning the page gives roughly 60 per cent more width, so the
        same figure prints legibly. The page returns to portrait afterwards and
        page numbering continues, because the new sections inherit the body
        footer.
        """
        path = os.path.join(self.figdir, filename)
        if not os.path.exists(path):
            print(f"  ! figure missing, skipped: {filename}")
            return

        land = self.doc.add_section(WD_SECTION.NEW_PAGE)
        land.orientation = WD_ORIENT.LANDSCAPE
        land.page_width, land.page_height = Mm(297), Mm(210)
        land.left_margin, land.right_margin = Cm(MARGIN_TOP), Cm(MARGIN_TOP)
        land.top_margin, land.bottom_margin = Cm(MARGIN_RIGHT), Cm(MARGIN_RIGHT)
        land.footer.is_linked_to_previous = True

        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.add_run().add_picture(path, width=Cm(width_cm))

        self._fig_n += 1
        caption = f"Figure {self.chapter}.{self._fig_n} {caption}"
        self.figures.append(caption)
        cap = self.doc.add_paragraph(style="FigureCaption")
        cap.add_run(caption)

        back = self.doc.add_section(WD_SECTION.NEW_PAGE)
        back.orientation = WD_ORIENT.PORTRAIT
        back.page_width, back.page_height = Mm(210), Mm(297)
        _configure_section(back)
        back.footer.is_linked_to_previous = True
        return cap

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def chapter_heading(self, number: str, title: str):
        """Chapter heading, matching the template's sample chapter page:
        "Chapter n" at 12 pt bold above the title at 16 pt bold.
        Kept as one Heading 1 paragraph so the TOC collects both lines."""
        self.chapter = number
        self._fig_n = 0
        self._tab_n = 0
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(16)
        r1 = p.add_run(f"Chapter {number}\n")
        r1.font.size = Pt(12)
        r1.bold = True
        r2 = p.add_run(title)
        r2.font.size = Pt(16)
        r2.bold = True

    def section_heading(self, number: str, title: str):
        """Section heading: 14pt CAPS, Heading 2."""
        p = self.doc.add_paragraph(style="Heading 2")
        p.add_run(f"{number} {title.upper()}")

    def subsection_heading(self, number: str, title: str):
        """Subsection heading: 12pt CAPS, Heading 3."""
        p = self.doc.add_paragraph(style="Heading 3")
        p.add_run(f"{number} {title.upper()}")

    def front_heading(self, title: str):
        """Unnumbered front/back-matter heading that still belongs in the TOC."""
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(title.upper())
        run.underline = True
        return p

    # ------------------------------------------------------------ figures etc
    def figure(self, filename: str, caption: str):
        """Embed a figure. `caption` carries no number; one is assigned here."""
        path = os.path.join(self.figdir, filename)
        if not os.path.exists(path):
            print(f"  ! figure missing, skipped: {filename}")
            return
        self._fig_n += 1
        caption = f"Figure {self.chapter}.{self._fig_n} {caption}"
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=FIG_W)
        self.figures.append(caption)
        cap = self.doc.add_paragraph(style="FigureCaption")
        cap.add_run(caption)
        return cap

    def table(self, headers: List[str], rows: List[List[str]], caption: str,
              widths: Optional[List[float]] = None, font_size: int = 10):
        """Render a table. `caption` carries no number; one is assigned here."""
        self._tab_n += 1
        caption = f"Table {self.chapter}.{self._tab_n} {caption}"
        self.tables.append(caption)
        cap_p = self.doc.add_paragraph(style="TableCaption")
        cap_run = cap_p.add_run(caption)
        cap_run.bold = True
        t = self.doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        hdr = t.rows[0].cells
        for i, h in enumerate(headers):
            hdr[i].text = ""
            p = hdr[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(h)
            r.bold = True
            r.font.name = BODY_FONT
            r.font.size = Pt(font_size)
        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].text = ""
                p = cells[i].paragraphs[0]
                p.paragraph_format.line_spacing = 1.0
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run(str(val))
                r.font.name = BODY_FONT
                r.font.size = Pt(font_size)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(8)
        return t

    def last_fig(self) -> str:
        """Number of the most recently embedded figure, for prose cross-references."""
        return f"{self.chapter}.{self._fig_n}"

    def last_tab(self) -> str:
        """Number of the most recently rendered table, for prose cross-references."""
        return f"{self.chapter}.{self._tab_n}"

    def next_fig(self) -> str:
        """Number the NEXT figure will receive (for prose written before the figure)."""
        return f"{self.chapter}.{self._fig_n + 1}"

    def next_tab(self) -> str:
        """Number the NEXT table will receive."""
        return f"{self.chapter}.{self._tab_n + 1}"

    def save(self, path: str):
        self._mark_fields_dirty()
        self.doc.save(path)

    def _mark_fields_dirty(self):
        """Ask Word to recalculate every field when the document is opened.

        The table of contents, the lists of figures and tables and the page
        numbers are all field codes, and a field carries no text until something
        paginates the document. finalize_docx.py does that through Word's COM
        interface, but it needs a working Word installation; setting
        w:updateFields makes the document repair itself on first open instead,
        so the build does not depend on one.
        """
        from docx.oxml.ns import qn

        settings = self.doc.settings.element
        for existing in settings.findall(qn("w:updateFields")):
            settings.remove(existing)
        el = settings.makeelement(qn("w:updateFields"), {qn("w:val"): "true"})
        settings.append(el)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the BCSE497J project report")
    ap.add_argument("--assets", default="report_assets",
                    help="directory produced by report_analysis.py")
    ap.add_argument("--out", default="Project_Report.docx")
    args = ap.parse_args()

    import report_content as RC

    b = ReportBuilder(args.assets)

    # Pass 1 builds the body so that figure and table captions are collected, then
    # the front matter is assembled ahead of it. python-docx has no insert-before,
    # so the body is built into a throwaway document first purely to harvest the
    # caption lists, and the real document is built in order afterwards.
    probe = ReportBuilder(args.assets)
    RC.chapter1(probe)
    RC.chapter2(probe)
    RC.chapter3(probe)
    RC.chapter4(probe)
    RC.chapter5(probe)
    RC.chapter6(probe)
    RC.chapter7(probe)
    RC.appendices(probe)
    b.figures = list(probe.figures)
    b.tables = list(probe.tables)

    # ---- front matter (roman numerals, restarted at i)
    RC.front_matter(b, CANDIDATE)
    RC.table_of_contents(b)
    RC.list_of_figures_tables(b)
    RC.list_of_acronyms(b)
    RC.list_of_symbols(b)

    front = b.doc.sections[0]
    _page_number_format(front, "lowerRoman", start=1)
    _footer_page_number(front)

    # ---- body in a new section with arabic numerals restarted at 1
    body_section = b.doc.add_section(WD_SECTION.NEW_PAGE)
    _configure_section(body_section)
    _page_number_format(body_section, "decimal", start=1)
    _footer_page_number(body_section, link_previous=False)

    captions_before = len(b.figures), len(b.tables)
    b.figures, b.tables = [], []          # re-collected as the real body is written

    RC.chapter1(b)
    RC.chapter2(b)
    RC.chapter3(b)
    RC.chapter4(b)
    RC.chapter5(b)
    RC.chapter6(b)
    RC.chapter7(b)
    RC.references(b)
    RC.appendices(b)

    if (len(b.figures), len(b.tables)) != captions_before:
        print(f"  ! caption count drift: probe={captions_before} "
              f"final={(len(b.figures), len(b.tables))}")

    b.save(args.out)
    print(f"wrote {args.out}")
    print(f"  figures: {len(b.figures)}   tables: {len(b.tables)}")
    placeholders = [k for k, v in CANDIDATE.items() if isinstance(v, str) and v.startswith("<<")]
    if placeholders:
        print("  ! placeholders still to fill in CANDIDATE: " + ", ".join(placeholders))


if __name__ == "__main__":
    main()
