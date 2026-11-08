"""Build a portable source-bound portfolio; PDF rendering uses an explicit Python.

Before the first real build, the operator must run the PDF skill's artifact marker.
The builder copies author-provided Markdown exactly and never computes study rates.
Optional --figure-captions JSON: {"figures": [{"id": "gallery_id", "caption": "...", "title": "optional"}]}.
Only these explicitly named figures are inserted into the paper (maximum four).
The complete, unchanged gallery is always included.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
from urllib.parse import unquote, urlsplit

MAX_REPRODUCIBILITY_BYTES = 200 * 1024**2


class PackageError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")


def safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise PackageError("Expected a portable relative file path")
    component = Path(relative)
    if component.is_absolute() or ":" in relative or ".." in component.parts:
        raise PackageError("Path must remain inside its package root")
    candidate = root / component
    if candidate.is_symlink() or not candidate.is_file() or root.resolve() not in candidate.resolve().parents:
        raise PackageError("Expected a regular contained file")
    return candidate


def regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PackageError(f"Expected a regular source file: {path}")
    return path.resolve()


def copied_file(source: Path, destination: Path) -> dict:
    before = sha256(source)
    shutil.copy2(source, destination)
    if sha256(source) != before or sha256(destination) != before:
        raise PackageError("A source changed while it was copied")
    return {"original_path": str(source), "bytes": source.stat().st_size, "sha256": before}


def verify_gallery(directory: Path) -> tuple[dict, list[Path]]:
    if directory.is_symlink() or not directory.is_dir():
        raise PackageError("Gallery must be a regular completed directory")
    root = directory.resolve()
    manifest = json.loads(regular_file(root / "figure_manifest.json").read_text(encoding="utf-8"))
    figures, tables = manifest.get("figures"), manifest.get("tables")
    if not isinstance(figures, list) or not isinstance(tables, list) or manifest.get("figure_count") != len(figures):
        raise PackageError("Incomplete gallery manifest")
    identifiers = [record.get("id") for record in figures]
    if len(set(identifiers)) != len(identifiers):
        raise PackageError("Duplicate gallery figure identifier")
    for record in figures:
        if set(record.get("paths", {})) != {"png", "svg"}:
            raise PackageError("Every gallery figure must retain PNG and SVG")
        for extension, relative in record["paths"].items():
            path = safe_file(root, relative)
            if sha256(path) != record.get("sha256", {}).get(extension):
                raise PackageError(f"Gallery figure hash mismatch: {relative}")
    for record in tables:
        if sha256(safe_file(root, record["path"])) != record.get("sha256"):
            raise PackageError("Gallery source-table hash mismatch")
    safe_file(root, "index.html")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PackageError("Gallery symbolic links cannot enter the portable package")
        if path.is_file():
            files.append(path)
    return manifest, files


def selected_figures(manifest: dict, captions: dict | None) -> list[dict]:
    if captions is None:
        return []
    rows = captions.get("figures")
    if not isinstance(rows, list) or len(rows) > 4:
        raise PackageError("Figure captions must explicitly select zero to four known figures")
    by_id = {row["id"]: row for row in manifest["figures"]}
    output, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in by_id or row["id"] in seen:
            raise PackageError("Unknown or duplicate explicitly selected figure")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise PackageError("Each paper figure needs an author-provided caption")
        source = by_id[row["id"]]
        title = row.get("title", source["title"])
        if not isinstance(title, str) or not title.strip():
            raise PackageError("Figure title must be text")
        output.append({"id": row["id"], "title": title, "caption": row["caption"],
                       "path": "gallery/" + source["paths"]["png"], "sha256": source["sha256"]["png"]})
        seen.add(row["id"])
    return output


def reproducibility_files(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise PackageError("Reproducibility evidence must be a regular directory")
    root = directory.resolve(); files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink(): raise PackageError("Symbolic links cannot enter reproducibility evidence")
        if path.name.lower() == ".env" or path.name.lower().startswith(".env.") or ".git" in path.relative_to(root).parts:
            raise PackageError("Environment secrets and Git internals cannot enter evidence")
        if path.is_file(): files.append(path)
    if not files or sum(path.stat().st_size for path in files) > MAX_REPRODUCIBILITY_BYTES:
        raise PackageError("Reproducibility evidence must be nonempty and at most 200 MiB; raw-payload duplication is not supported")
    for path in files:
        if path.suffix.lower() not in (".md", ".html", ".htm"): continue
        source = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".md":
            # Code examples are not navigable document links.
            source = "\n".join("\n".join(item["text"] for item in block["items"]) if block["kind"] == "list" else
                               "\n".join(" ".join(row) for row in block["rows"]) if block["kind"] == "table" else
                               block.get("text", "") for block in blocks(source) if block["kind"] != "code")
            links = [("src" if marker else "href", target) for marker, target in re.findall(r"(!?)\[[^\]]+\]\(([^)\s]+)(?:\s+[^)]*)?\)", source)]
        else:
            links = re.findall(r"(href|src)\s*=\s*['\"]([^'\"]+)['\"]", source, flags=re.I)
        for attribute, link in links:
            parsed = urlsplit(link)
            if parsed.scheme in ("http", "https", "mailto"):
                if attribute.lower() == "src": raise PackageError("Offline evidence cannot depend on remote image/script assets")
                continue
            if parsed.scheme or parsed.netloc or parsed.path.startswith(("/", "\\")):
                raise PackageError(f"Nonportable evidence link: {path.relative_to(root)}")
            if not parsed.path: continue
            target = (path.parent / unquote(parsed.path)).resolve()
            if target != root and root not in target.parents or not target.exists():
                raise PackageError(f"Broken or escaping evidence link in {path.relative_to(root)}: {link}")
    return files


def validate_replay_binding(report: dict, evidence: Path) -> dict:
    """Verify an existing detached replay report against copied evidence only."""
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise PackageError("Replay report must explicitly have status='passed'")
    bindings = report.get("bindings")
    if not isinstance(bindings, dict) or not isinstance(bindings.get("files"), dict) or not bindings["files"]:
        raise PackageError("Replay report has no frozen evidence-file bindings")
    required = {"src/__init__.py", "src/cot_audit.py", "src/cot_figures.py"}
    required |= {"inputs/" + name for name in ("slim.parquet", "download-spec.json", "metadata_audit.json", "intake_profile.json")}
    for phase in ("discovery", "confirmation"):
        required.add(f"protocols/{phase}.json")
        required |= {f"reference/{phase}/{name}" for name in ("analysis.json", "dataquality.json", "protocol.json", "pairs.csv", "per_question.parquet", "bootstrap.npz")}
    if not required.issubset(bindings["files"]):
        raise PackageError("Replay report omits required H2 producer/input/protocol/reference bindings")
    additional = bindings.get("additional_package_records", {})
    if not isinstance(additional, dict): raise PackageError("Invalid replay additional-package bindings")
    reported = dict(bindings["files"])
    for relative, record in additional.items():
        if relative in reported and reported[relative] != record: raise PackageError("Conflicting replay file bindings")
        reported[relative] = record
    verified = []
    for relative, record in sorted(reported.items()):
        if not isinstance(record, dict): raise PackageError("Invalid replay file binding")
        expected = record.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise PackageError("Invalid replay-bound SHA256")
        path = safe_file(evidence, relative)
        if sha256(path) != expected or type(record.get("bytes")) is not int or path.stat().st_size != record["bytes"]:
            raise PackageError(f"Copied evidence does not match detached replay: {relative}")
        verified.append({"path": relative, "bytes": record["bytes"], "sha256": expected})
    if bindings.get("producer_source_sha256") != bindings["files"]["src/cot_audit.py"]["sha256"]:
        raise PackageError("Detached replay producer identity is inconsistent")
    return {"status": "passed", "verified_copied_evidence_files": verified,
            "verification_scope": "Existing report and frozen evidence-file bindings; no recomputation by the packager"}


def blocks(markdown: str) -> list[dict]:
    """Small deterministic Markdown subset, with no HTML execution or downloads."""
    lines = markdown.splitlines(); output, paragraph = [], []
    def flush():
        if paragraph:
            output.append({"kind": "paragraph", "text": " ".join(paragraph)})
            paragraph.clear()
    index = 0
    while index < len(lines):
        line = lines[index]; stripped = line.strip()
        if not stripped:
            flush(); index += 1; continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush(); fence = stripped[:3]; code = []; index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code.append(lines[index]); index += 1
            if index == len(lines): raise PackageError("Unclosed Markdown code fence")
            output.append({"kind": "code", "text": "\n".join(code)}); index += 1; continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading:
            flush(); output.append({"kind": "heading", "level": len(heading[1]), "text": heading[2]}); index += 1; continue
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped):
            flush(); output.append({"kind": "rule"}); index += 1; continue
        if "|" in line and index + 1 < len(lines) and re.fullmatch(r"\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*", lines[index + 1]):
            flush(); rows = [[part.strip() for part in line.strip().strip("|").split("|")]]; index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append([part.strip() for part in lines[index].strip().strip("|").split("|")]); index += 1
            if any(len(row) != len(rows[0]) for row in rows): raise PackageError("Markdown table has inconsistent column counts")
            output.append({"kind": "table", "rows": rows}); continue
        item = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$", line)
        if item:
            flush(); items = []
            while index < len(lines):
                item = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$", lines[index])
                if not item: break
                items.append({"marker": item[2], "text": item[3], "indent": len(item[1])}); index += 1
            output.append({"kind": "list", "items": items}); continue
        if stripped.startswith(">"):
            flush(); quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].lstrip()); index += 1
            output.append({"kind": "quote", "text": " ".join(quote)}); continue
        paragraph.append(stripped); index += 1
    flush()
    return output


TOKEN = re.compile(r"(`[^`\n]+`|!?\[[^\]\n]+\]\([^\)\n]+\)|\*\*[^*\n]+\*\*|__[^_\n]+__|(?<!\*)\*[^*\n]+\*(?!\*)|(?<!_)_[^_\n]+_(?!_))")


def inline(text: str, pdf: bool = False, depth: int = 0) -> str:
    if depth > 4: return html.escape(text)
    result, previous = [], 0
    for match in TOKEN.finditer(text):
        result.append(html.escape(text[previous:match.start()]))
        token = match[0]
        if token.startswith("`"):
            content = html.escape(token[1:-1]); font = "Courier" if token[1:-1].isascii() else "Body"
            result.append('<font name="' + font + '">' + content + '</font>' if pdf else "<code>" + content + "</code>")
        elif token.startswith("[") or token.startswith("!["):
            link = re.fullmatch(r"!?\[([^\]]+)\]\(([^)]+)\)", token)
            label, target = link[1], link[2]
            allowed = re.match(r"^(https?://|mailto:)", target) or (not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) and not target.startswith(("/", "\\")))
            result.append('<a href="' + html.escape(target, quote=True) + '">' + inline(label, pdf, depth + 1) + "</a>" if allowed else html.escape(label))
        elif token.startswith(("**", "__")):
            result.append("<b>" + inline(token[2:-2], pdf, depth + 1) + "</b>")
        else:
            result.append("<i>" + inline(token[1:-1], pdf, depth + 1) + "</i>")
        previous = match.end()
    result.append(html.escape(text[previous:]))
    return "".join(result)


def manuscript_title(content: list[dict]) -> str:
    return next((block["text"] for block in content if block["kind"] == "heading" and block["level"] == 1), "Research portfolio")


def html_document(content: list[dict], figures: list[dict]) -> str:
    body = []
    for block in content:
        kind = block["kind"]
        if kind == "heading":
            level = min(6, block["level"]); body.append(f"<h{level}>" + inline(block["text"]) + f"</h{level}>")
        elif kind in ("paragraph", "quote"):
            tag = "blockquote" if kind == "quote" else "p"; body.append(f"<{tag}>" + inline(block["text"]) + f"</{tag}>")
        elif kind == "code": body.append("<pre><code>" + html.escape(block["text"]) + "</code></pre>")
        elif kind == "rule": body.append("<hr>")
        elif kind == "list":
            body.append('<div class="list">' + "".join('<p style="margin-left:' + str(min(item["indent"], 12) * .35) + 'em"><span class="marker">' + html.escape(item["marker"]) + '</span> ' + inline(item["text"]) + "</p>" for item in block["items"]) + "</div>")
        elif kind == "table":
            rows = block["rows"]
            body.append('<div class="table-wrap"><table><thead><tr>' + "".join("<th>" + inline(value) + "</th>" for value in rows[0]) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + inline(value) + "</td>" for value in row) + "</tr>" for row in rows[1:]) + "</tbody></table></div>")
    for index, figure in enumerate(figures, 1):
        body.append('<section class="figure-page"><h2>' + html.escape(figure["title"]) + '</h2><figure><img src="' + html.escape(figure["path"]) + '" alt="' + html.escape(figure["title"], quote=True) + '"><figcaption><b>Figure ' + str(index) + ".</b> " + inline(figure["caption"]) + "</figcaption></figure></section>")
    css = """@page{size:A4;margin:19mm 18mm}@page wide{size:A4 landscape;margin:15mm}*{box-sizing:border-box}body{font:17px/1.6 Georgia,serif;color:#243943;background:#f1f5f6;margin:0}main{max-width:1050px;margin:32px auto;padding:56px 68px;background:white;box-shadow:0 2px 15px #1c394416}h1,h2,h3,h4{font-family:Arial,sans-serif;color:#164755;line-height:1.22;break-after:avoid}h1{font-size:34px;border-bottom:3px solid #14877a;padding-bottom:22px}h2{font-size:24px;margin-top:34px}h3{font-size:20px}p{orphans:3;widows:3}a{color:#08758b;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font:14px/1.4 Arial,sans-serif;margin:18px 0}th{background:#164755;color:white;text-align:left}td,th{padding:9px 11px;border:1px solid #d4e0e4;vertical-align:top;overflow-wrap:anywhere}tr:nth-child(even){background:#f1f6f7}thead{display:table-header-group}.table-wrap{overflow:auto}code{font-size:.86em;overflow-wrap:anywhere}pre{background:#eff4f6;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere}blockquote{border-left:3px solid #14877a;margin-left:0;padding:6px 18px;color:#48606a}.list p{margin:5px 0}.marker{display:inline-block;min-width:1.4em;color:#14877a}figure{margin:20px 0}figure img{width:100%;height:auto}figcaption{font-size:14px;line-height:1.5;color:#405762}nav{font:14px Arial,sans-serif;border-bottom:1px solid #d4e0e4;padding-bottom:16px}.figure-page{break-before:page;page:wide}hr{border:0;border-top:1px solid #d4e0e4;margin:24px 0}@media print{body{background:white;font-size:10.5pt}main{padding:0;margin:0;box-shadow:none;max-width:none}nav{display:none}h1{font-size:23pt}h2{font-size:16pt}h3{font-size:13pt}table{font-size:8.6pt}td,th{padding:5pt}figcaption{font-size:9pt}a{color:inherit;text-decoration:none}.table-wrap{overflow:visible}.figure-page img{max-height:145mm;object-fit:contain}}@media(max-width:700px){main{padding:25px 22px;margin:0}h1{font-size:27px}}"""
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(manuscript_title(content)) + "</title><style>" + css + '</style><main><nav><a href="manuscript.pdf">PDF</a> | <a href="manuscript.md">Original Markdown</a> | <a href="gallery/index.html">Complete evidence gallery</a> | <a href="portfolio_manifest.json">Provenance manifest</a></nav>' + "".join(body) + "</main></html>"


def pdf_worker(output: Path) -> dict:
    """Called synchronously only through the explicitly selected PDF runtime."""
    import platform
    import reportlab
    import pypdf
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image, LongTable,
                                   NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer, TableStyle, XPreformatted)
    root = output.resolve()
    target = root / "manuscript.pdf"
    if target.exists(): raise PackageError("PDF output already exists; preserve it and use a new package attempt")
    config = json.loads(safe_file(root, "provenance/package_inputs.json").read_text(encoding="utf-8"))
    manuscript = safe_file(root, "manuscript.md").read_text(encoding="utf-8-sig")
    content = blocks(manuscript); title = manuscript_title(content)
    fonts = Path(reportlab.__file__).resolve().parent / "fonts"
    font_files = {"Body": "Vera.ttf", "Body-Bold": "VeraBd.ttf", "Body-Italic": "VeraIt.ttf", "Body-BoldItalic": "VeraBI.ttf"}
    for name, filename in font_files.items(): pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-BoldItalic")
    visible_text = manuscript + "\n".join(figure["title"] + " " + figure["caption"] for figure in config["selected_figures"])
    supported = set.intersection(*(set(pdfmetrics.getFont(name).face.charToGlyph) for name in font_files))
    unsupported = sorted({ord(char) for char in visible_text if not char.isspace() and ord(char) not in supported})
    if unsupported:
        raise PackageError("Embedded Vera fonts cannot represent manuscript characters " + ", ".join(f"U+{code:04X}" for code in unsupported) + "; use equivalent ASCII notation in the author manuscript")
    palette = {"ink": colors.HexColor("#243943"), "blue": colors.HexColor("#164755"), "teal": colors.HexColor("#14877a"), "line": colors.HexColor("#d4e0e4")}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("PortfolioBody", fontName="Body", fontSize=10.2, leading=15, textColor=palette["ink"], spaceAfter=9, splitLongWords=True, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle("PortfolioTitle", parent=styles["PortfolioBody"], fontName="Body-Bold", fontSize=21, leading=27, textColor=palette["blue"], spaceBefore=4, spaceAfter=19, keepWithNext=True))
    for level, size in ((1, 16), (2, 14), (3, 11.6), (4, 10.6), (5, 10.2), (6, 10.2)):
        styles.add(ParagraphStyle(f"PortfolioH{level}", parent=styles["PortfolioBody"], fontName="Body-Bold", fontSize=size, leading=size * 1.3, textColor=palette["blue"], spaceBefore=15 if level < 3 else 10, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle("PortfolioQuote", parent=styles["PortfolioBody"], leftIndent=14, rightIndent=8, textColor=colors.HexColor("#48606a")))
    styles.add(ParagraphStyle("PortfolioCaption", parent=styles["PortfolioBody"], fontSize=9, leading=13, spaceBefore=9, spaceAfter=0))
    styles.add(ParagraphStyle("PortfolioCell", parent=styles["PortfolioBody"], fontSize=8.3, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle("PortfolioHeader", parent=styles["PortfolioCell"], fontName="Body-Bold", textColor=colors.white))
    styles.add(ParagraphStyle("PortfolioCode", fontName="Courier", fontSize=7.7, leading=11, spaceBefore=4, spaceAfter=11, textColor=palette["ink"]))
    page_width, page_height = A4; margin = 49; wide_width, wide_height = landscape(A4)
    def page_decoration(canvas, document):
        width, height = canvas._pagesize
        canvas.saveState(); canvas.setStrokeColor(palette["line"]); canvas.setLineWidth(.5)
        canvas.line(margin, height - 31, width - margin, height - 31)
        canvas.setFillColor(palette["blue"]); canvas.setFont("Body", 7)
        canvas.drawString(margin, height - 23, "RESEARCH PORTFOLIO")
        canvas.drawRightString(width - margin, 23, str(document.page))
        canvas.setFillColor(colors.HexColor("#607782")); canvas.setFont("Body", 6.5)
        canvas.drawString(margin, 23, "Complete source tables and figures are included in the evidence gallery.")
        canvas.restoreState()
    doc = BaseDocTemplate(str(target), pagesize=A4, rightMargin=margin, leftMargin=margin, topMargin=49,
                          bottomMargin=43, title=title, author="AI-assisted research audit", subject="Author-provided research portfolio with source-bound evidence", pageCompression=1)
    doc.addPageTemplates([
        PageTemplate(id="Text", frames=[Frame(margin, 43, page_width - 2 * margin, page_height - 92, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_decoration, pagesize=A4),
        PageTemplate(id="Figure", frames=[Frame(margin, 43, wide_width - 2 * margin, wide_height - 92, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_decoration, pagesize=landscape(A4))])
    story, first_title = [], True
    for block in content:
        kind = block["kind"]
        if kind == "heading":
            style = styles["PortfolioTitle"] if first_title and block["level"] == 1 else styles[f"PortfolioH{block['level']}"]
            story.append(Paragraph(inline(block["text"], True), style))
            if block["level"] == 1: first_title = False
        elif kind in ("paragraph", "quote"):
            story.append(Paragraph(inline(block["text"], True), styles["PortfolioQuote" if kind == "quote" else "PortfolioBody"]))
        elif kind == "rule": story.extend([Spacer(1, 4), HRFlowable(width="100%", thickness=.7, color=palette["line"]), Spacer(1, 9)])
        elif kind == "code":
            folded = "\n".join("\n".join(textwrap.wrap(line, width=91, replace_whitespace=False, drop_whitespace=False)) if len(line) > 91 else line for line in block["text"].splitlines())
            style = styles["PortfolioCode"] if folded.isascii() else ParagraphStyle("UnicodeCode", parent=styles["PortfolioCode"], fontName="Body")
            story.append(XPreformatted(html.escape(folded), style))
        elif kind == "list":
            for item in block["items"]:
                marker = "-" if item["marker"] in ("*", "+") else item["marker"]
                style = ParagraphStyle("ListItem", parent=styles["PortfolioBody"], leftIndent=14 + min(item["indent"], 12) * 3, firstLineIndent=-10, spaceAfter=5)
                story.append(Paragraph(html.escape(marker) + " " + inline(item["text"], True), style))
            story.append(Spacer(1, 4))
        elif kind == "table":
            rows = block["rows"]; columns = len(rows[0]); width = page_width - 2 * margin
            data = [[Paragraph(inline(value, True), styles["PortfolioHeader" if i == 0 else "PortfolioCell"]) for value in row] for i, row in enumerate(rows)]
            table = LongTable(data, colWidths=[width / columns] * columns, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), palette["blue"]), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f6f7")]),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), .7, palette["teal"]),
                                       ("GRID", (0, 1), (-1, -1), .3, palette["line"]), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
            story.extend([table, Spacer(1, 12)])
    for index, figure in enumerate(config["selected_figures"], 1):
        source = safe_file(root, figure["path"])
        if sha256(source) != figure["sha256"]: raise PackageError("Selected paper figure changed")
        story.extend([NextPageTemplate("Figure"), PageBreak(), Paragraph(inline(figure["title"], True), styles["PortfolioH2"])])
        image = Image(str(source)); max_width, max_height = wide_width - 2 * margin, wide_height - 220
        scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
        image.hAlign = "CENTER"; story.append(image)
        story.append(Paragraph(f"<b>Figure {index}.</b> " + inline(figure["caption"], True), styles["PortfolioCaption"]))
    if not story: raise PackageError("Cannot render an empty manuscript")
    doc.build(story)
    reader = pypdf.PdfReader(target)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not reader.pages or len(text.strip()) < 20: raise PackageError("Rendered PDF has no readable text")
    executable = Path(sys.executable).resolve()
    report = {"python": platform.python_version(), "executable": str(executable), "executable_sha256": sha256(executable),
              "reportlab": reportlab.Version, "pypdf": pypdf.__version__, "reportlab_module_sha256": sha256(Path(reportlab.__file__)),
              "pypdf_module_sha256": sha256(Path(pypdf.__file__)), "font_sha256": {name: sha256(fonts / filename) for name, filename in font_files.items()},
              "builder_sha256": sha256(Path(__file__)), "page_count": len(reader.pages), "target_page_range": [6, 10],
              "font_family": "embedded Bitstream Vera; unsupported manuscript glyphs rejected before authoring",
              "target_page_range_met": 6 <= len(reader.pages) <= 10, "selected_figure_count": len(config["selected_figures"]),
              "extracted_text_characters": len(text), "pdf_sha256": sha256(target), "visual_review": "required_not_performed_by_builder"}
    dump(root / "provenance/pdf_render_report.json", report)
    return report


def build(manuscript: Path, gallery: Path, output: Path, pdf_python: Path, figure_captions: Path | None,
          reproducibility: Path | None = None, replay_report: Path | None = None) -> dict:
    source = regular_file(manuscript); runtime = regular_file(pdf_python); builder = regular_file(Path(__file__))
    root = output.resolve(); gallery_root = gallery.resolve()
    if root == gallery_root or gallery_root in root.parents or root in gallery_root.parents:
        raise PackageError("New output must be outside the original gallery")
    if root.exists(): raise PackageError("Output exists; preserve the previous attempt and choose a new directory")
    manifest, gallery_files = verify_gallery(gallery)
    evidence_root = reproducibility.resolve() if reproducibility else None
    evidence_files = reproducibility_files(reproducibility) if reproducibility else []
    replay_path = regular_file(replay_report) if replay_report else None
    if replay_path is not None and evidence_root is None:
        raise PackageError("--replay-report requires the copied --reproducibility evidence directory")
    if evidence_root is not None and (root == evidence_root or evidence_root in root.parents or root in evidence_root.parents):
        raise PackageError("New package output must be outside source evidence")
    captions_path = regular_file(figure_captions) if figure_captions else None
    captions = json.loads(captions_path.read_text(encoding="utf-8")) if captions_path else None
    selected = selected_figures(manifest, captions)
    content = blocks(source.read_text(encoding="utf-8-sig"))
    if not content: raise PackageError("Manuscript contains no content")
    root.mkdir(parents=True, exist_ok=False); (root / "provenance").mkdir(); (root / "gallery").mkdir()
    try:
        bindings = {"manuscript": copied_file(source, root / "manuscript.md"), "builder": copied_file(builder, root / "provenance/build_cot_portfolio.py"),
                    "pdf_runtime": {"executable": str(runtime), "executable_sha256": sha256(runtime)}}
        if captions_path: bindings["figure_captions"] = copied_file(captions_path, root / "provenance/figure_captions.json")
        copied_evidence = []
        if evidence_root is not None:
            (root / "evidence").mkdir()
            for file in evidence_files:
                relative = file.relative_to(evidence_root); destination = root / "evidence" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                record = copied_file(file, destination)
                copied_evidence.append({"path": "evidence/" + relative.as_posix(), "bytes": record["bytes"], "sha256": record["sha256"]})
            bindings["reproducibility"] = {"original_path": str(evidence_root), "file_count": len(copied_evidence),
                                           "bytes": sum(record["bytes"] for record in copied_evidence), "files": copied_evidence}
        replay_verification = None
        if replay_path is not None:
            bindings["replay_report"] = copied_file(replay_path, root / "provenance/replay_report.json")
            report = json.loads((root / "provenance/replay_report.json").read_text(encoding="utf-8"))
            replay_verification = validate_replay_binding(report, root / "evidence")
            bindings["replay_report"].update(status="passed", verified_evidence_file_count=len(replay_verification["verified_copied_evidence_files"]))
        copied_gallery = []
        for file in gallery_files:
            relative = file.relative_to(gallery_root); destination = root / "gallery" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            record = copied_file(file, destination); copied_gallery.append({"path": "gallery/" + relative.as_posix(), "bytes": record["bytes"], "sha256": record["sha256"]})
        config = {"schema_version": 1, "title": manuscript_title(content), "selected_figures": selected, "input_bindings": bindings,
                  "gallery_manifest_sha256": sha256(root / "gallery/figure_manifest.json"), "gallery_files": copied_gallery,
                  "detached_replay_verification": replay_verification,
                  "author_text_policy": "Original Markdown copied byte-for-byte; formatting only; no generated study numbers or claims",
                  "pdf_layout": {"text_page": "A4 portrait", "figure_page": "A4 landscape, one explicitly selected figure per page",
                                 "body_font_points": 10.2, "body_leading_points": 15, "margins_points": 49,
                                 "page_target": [6, 10], "artificial_padding": False}}
        dump(root / "provenance/package_inputs.json", config)
        (root / "manuscript.html").write_text(html_document(content, selected), encoding="utf-8")
        env = dict(os.environ)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): env[name] = "4"
        env["CUDA_VISIBLE_DEVICES"] = ""; env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [str(runtime), str(root / "provenance/build_cot_portfolio.py"), "--mode", "pdf-worker", "--output", str(root)]
        try:
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
        except subprocess.TimeoutExpired as exc:
            for name, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                (root / f"provenance/pdf-worker.{name}.txt").write_text(text, encoding="utf-8")
            raise PackageError("PDF worker timed out; partial output and logs preserved") from exc
        (root / "provenance/pdf-worker.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (root / "provenance/pdf-worker.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode: raise PackageError(f"PDF worker failed with exit code {result.returncode}; logs and all partial artifacts preserved")
        render_report = json.loads((root / "provenance/pdf_render_report.json").read_text(encoding="utf-8"))
        if render_report["builder_sha256"] != bindings["builder"]["sha256"] or render_report["executable_sha256"] != bindings["pdf_runtime"]["executable_sha256"]:
            raise PackageError("PDF worker source/runtime identity differs from the frozen package inputs")
        for file in gallery_files:
            relative = "gallery/" + file.relative_to(gallery_root).as_posix()
            expected = next(record["sha256"] for record in copied_gallery if record["path"] == relative)
            if sha256(file) != expected: raise PackageError("Original gallery changed during packaging")
        if sha256(source) != bindings["manuscript"]["sha256"]: raise PackageError("Original manuscript changed during packaging")
        if replay_path is not None and sha256(replay_path) != bindings["replay_report"]["sha256"]:
            raise PackageError("Original detached replay report changed during packaging")
        if evidence_root is not None:
            for record in copied_evidence:
                original = evidence_root / Path(record["path"]).relative_to("evidence")
                if sha256(original) != record["sha256"]: raise PackageError("Original reproducibility evidence changed during packaging")
        quoted_runtime = "'" + str(runtime).replace("'", "''") + "'"
        optional = " --figure-captions provenance/figure_captions.json" if captions_path else ""
        if evidence_root is not None: optional += " --reproducibility evidence"
        if replay_path is not None: optional += " --replay-report provenance/replay_report.json"
        readme = "# Research portfolio bundle\n\n" + "Open [the printable manuscript](manuscript.html), [the PDF](manuscript.pdf), or [the complete evidence gallery](gallery/index.html). The original author text is [manuscript.md](manuscript.md).\n\n" + "All gallery PNG/SVG files, source CSV tables and their input/output manifest are included unchanged. Paper illustrations are only those explicitly selected by the author; absence of a captions manifest inserts no illustrations.\n\n" + "## Reproduction\n\nFrom this directory, use the recorded PDF runtime (or a compatible Python with the exact ReportLab and pypdf versions recorded in provenance/pdf_render_report.json):\n\n```powershell\n& " + quoted_runtime + " provenance/build_cot_portfolio.py --manuscript manuscript.md --gallery gallery --output ../portfolio-rebuilt --pdf-python " + quoted_runtime + optional + "\n```\n\n" + "The build is synchronous, uses four CPU threads, and makes no network or model calls. Use a new output directory on each attempt; failed or partial artifacts are preserved.\n\n" + "## Provenance and review\n\n[portfolio_manifest.json](portfolio_manifest.json) hashes the package files. [package_inputs.json](provenance/package_inputs.json) binds manuscript, gallery, selected captions, source script and runtime executable. [pdf_render_report.json](provenance/pdf_render_report.json) records executable and library/font hashes, page count and target-range status. The builder does not assess scientific validity. Visual inspection of the rendered PDF pages remains required before delivery.\n"
        (root / "README.md").write_text(readme, encoding="utf-8")
        evidence_readme = ("\n## Offline scientific reproduction\n\n" +
            ("The [frozen evidence directory](evidence/) is included with file-by-file hashes. Use its README and pinned code, tests, protocols, lockfile, slim data and closed reviews for offline reanalysis. " if evidence_root is not None else "No scientific reproducibility directory was provided to this build. ") +
            "The 940 MB raw transcript payloads are not duplicated in this portfolio. The original author release is ameek/measuring_cot_monitorability_transcripts at revision a7a8b25ae467db9d80d63e55ac053b48753b2d36; use the frozen download specification and source-license audit for availability, exact file hashes and the release's model-training restrictions. Reanalysis uses four CPU threads, no model/API calls and no GPU.\n")
        with (root / "README.md").open("a", encoding="utf-8") as stream: stream.write(evidence_readme)
        replay_text = ("\n## Detached computational replay\n\n" +
                       ("A [passed detached replay report](provenance/replay_report.json) is included byte-for-byte. This package build verified every reported required and additional file hash/byte count against the copied frozen evidence. The packager performed no replay recomputation. Scientific validity is assessed separately.\n" if replay_path is not None else
                        "No detached replay report was supplied to this package build.\n"))
        with (root / "README.md").open("a", encoding="utf-8") as stream: stream.write(replay_text)
        outputs = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(root.rglob("*")) if path.is_file()]
        report = {"schema_version": 1, "status": "built_pending_visual_review", "manuscript_title": config["title"],
                  "source_bindings": bindings, "gallery_figure_count": manifest["figure_count"], "selected_paper_figures": selected,
                  "pdf": render_report, "files": outputs, "network_requests": 0, "model_calls": 0,
                  "detached_replay_verification": replay_verification,
                  "scientific_validity": "not_assessed", "visual_review": "required_not_performed_by_builder"}
        dump(root / "portfolio_manifest.json", report)
        return report
    except BaseException as exc:
        failure = root / "failure.json"
        if not failure.exists(): dump(failure, {"status": "failed", "error": type(exc).__name__, "message": str(exc), "partial_artifacts": "preserved"})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "pdf-worker"), default="build")
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--gallery", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdf-python", type=Path)
    parser.add_argument("--figure-captions", type=Path)
    parser.add_argument("--reproducibility", type=Path, help="Frozen offline evidence directory, at most 200 MiB")
    parser.add_argument("--replay-report", type=Path, help="Existing passed detached replay report bound to the copied evidence")
    args = parser.parse_args()
    if args.mode == "pdf-worker": report = pdf_worker(args.output)
    else:
        if not all((args.manuscript, args.gallery, args.pdf_python)): parser.error("build requires --manuscript, --gallery and --pdf-python")
        report = build(args.manuscript, args.gallery, args.output, args.pdf_python, args.figure_captions, args.reproducibility, args.replay_report)
    print(json.dumps({"status": report.get("status", "pdf_rendered"), "output": str(args.output),
                      "pages": report.get("pdf", report).get("page_count"), "visual_review": "required"}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
