"""Post-process the generated report with Word itself.

python-docx cannot paginate or evaluate fields, so the Table of Contents, the
List of Figures and the List of Tables are written as Word field codes and left
unevaluated. This script drives Word via COM to refresh every field and repaginate,
so the delivered .docx opens with a populated contents page rather than a grey
placeholder, and reports the true page count.

    python finalize_docx.py BCSE497J_Project_Report.docx
"""
import argparse
import os
import sys

WD_FIELD_TOC = 13
WD_STATISTIC_PAGES = 2
WD_DO_NOT_SAVE_CHANGES = 0


def finalize(path: str, save: bool = True) -> int:
    import win32com.client as win32

    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    word = win32.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    doc = None
    try:
        doc = word.Documents.Open(path)

        # Fields first, then the TOC objects: a TOC's page numbers are only
        # correct once the body it indexes has been laid out.
        doc.Fields.Update()
        for toc in doc.TablesOfContents:
            toc.Update()
        for tof in doc.TablesOfFigures:
            tof.Update()
        doc.Repaginate()

        pages = int(doc.ComputeStatistics(WD_STATISTIC_PAGES))
        words = int(doc.ComputeStatistics(0))

        if save:
            doc.Save()
        doc.Close(WD_DO_NOT_SAVE_CHANGES if not save else -1)
        doc = None
        return pages, words
    finally:
        if doc is not None:
            try:
                doc.Close(WD_DO_NOT_SAVE_CHANGES)
            except Exception:
                pass
        word.Quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="BCSE497J_Project_Report.docx")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    pages, words = finalize(args.path, save=not args.no_save)
    print(f"pages: {pages}   words: {words}")
