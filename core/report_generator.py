"""
Report generator: produces a self-contained HTML report for a project.
No external dependencies — uses only stdlib + data already in the project object.
"""
import html
import os
from datetime import datetime
from typing import Optional


def _bar(pct: float, color: str = "#1F6FEB", height: int = 8) -> str:
    """Return an inline-HTML progress bar div filled to *pct* percent."""
    w = max(0.0, min(100.0, pct))
    return (
        f'<div style="background:#373E47;border-radius:4px;height:{height}px;width:100%;">'
        f'<div style="background:{color};border-radius:4px;height:{height}px;width:{w:.1f}%"></div>'
        f'</div>'
    )


def generate_html_report(project, output_path: str) -> str:
    """
    Write a self-contained HTML report to output_path.
    Returns the path on success.
    """
    p = project
    cfg = p.config
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Compute stats ──────────────────────────────────────────────────────────
    total_img = len(p.images)
    labeled = p.get_labeled_image_count()
    unlabeled = total_img - labeled
    roi_count = p.get_roi_count()
    label_counts = {}
    for img_path in p.images:
        lbl = p.image_labels.get(img_path)
        if lbl:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    total_labeled_imgs = sum(label_counts.values()) or 1
    runs = p.training_runs or []
    last_run = runs[-1] if runs else None
    last_metrics = (last_run or {}).get("metrics", {})

    # ── Class distribution rows ────────────────────────────────────────────────
    cls_rows = ""
    for cls, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        color = (p.labels.get(cls) or {}).get("color", "#1F6FEB")
        pct = cnt / total_labeled_imgs * 100
        cls_rows += f"""
        <tr>
          <td style="color:{html.escape(color)};font-weight:bold">{html.escape(cls)}</td>
          <td style="text-align:center">{cnt}</td>
          <td style="text-align:center">{pct:.1f}%</td>
          <td style="width:200px">{_bar(pct, color)}</td>
        </tr>"""

    # ── Training runs table ────────────────────────────────────────────────────
    run_rows = ""
    for i, run in enumerate(reversed(runs[-10:]), 1):
        m = run.get("metrics", {})
        acc = m.get("accuracy", 0) * 100
        f1 = m.get("macro_f1", 0) * 100
        ts = (run.get("timestamp", "") or "")[:19].replace("T", " ")
        run_rows += f"""
        <tr>
          <td style="text-align:center">{i}</td>
          <td>{html.escape(ts)}</td>
          <td>{html.escape(run.get('model_type', '?'))}</td>
          <td style="text-align:center">{html.escape(str(run.get('epochs', '?')))}</td>
          <td style="color:{'#3FB950' if acc >= 80 else '#D29922'};text-align:center;font-weight:bold">{acc:.2f}%</td>
          <td style="color:{'#3FB950' if f1 >= 80 else '#D29922'};text-align:center;font-weight:bold">{f1:.2f}%</td>
          <td style="font-size:10px;color:#7F8C8D">{html.escape(os.path.basename(run.get('best_model_path') or '–'))}</td>
        </tr>"""

    if not run_rows:
        run_rows = '<tr><td colspan="7" style="text-align:center;color:#7F8C8D">Keine Trainingsläufe vorhanden</td></tr>'

    # ── Warnings ───────────────────────────────────────────────────────────────
    warnings_html = ""
    if unlabeled > 0:
        warnings_html += f'<li>⚠ {unlabeled} Bild(er) noch nicht gelabelt.</li>'
    if len(label_counts) >= 2:
        vals = list(label_counts.values())
        if min(vals) > 0:
            ratio = max(vals) / min(vals)
            if ratio > 5:
                warnings_html += f'<li>⚠ Klassenungleichgewicht: Ratio {ratio:.1f}:1 — Augmentation empfohlen.</li>'
    if not warnings_html:
        warnings_html = '<li style="color:#3FB950">✓ Keine Warnungen.</li>'

    # ── Current model info ─────────────────────────────────────────────────────
    model_name = os.path.basename(p.current_model_path or "") or "Kein Modell"
    acc_str = f"{last_metrics.get('accuracy', 0)*100:.2f}%" if last_run else "–"
    f1_str  = f"{last_metrics.get('macro_f1',  0)*100:.2f}%" if last_run else "–"

    html_content = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Picture Studio – Projektbericht</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #22272E; color: #ADBAC7; margin: 0; padding: 24px; }}
  h1   {{ color: #388BFD; font-size: 24px; margin-bottom: 4px; }}
  h2   {{ color: #E6EDF3; font-size: 16px; border-bottom: 1px solid #444C56;
          padding-bottom: 6px; margin-top: 28px; }}
  .meta {{ color: #7F8C8D; font-size: 12px; margin-bottom: 24px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: #2D333B; border: 1px solid #444C56; border-radius: 8px;
           padding: 16px 24px; min-width: 140px; text-align: center; }}
  .card .val {{ font-size: 28px; font-weight: bold; color: #388BFD; }}
  .card .lbl {{ font-size: 11px; color: #7F8C8D; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ padding: 8px 12px; border-bottom: 1px solid #373E47; text-align: left; }}
  th {{ background: #373E47; color: #7F8C8D; font-size: 11px; font-weight: bold; text-transform: uppercase; }}
  tr:hover td {{ background: #373E47; }}
  ul {{ margin: 0; padding-left: 20px; line-height: 1.8; }}
  .section {{ background: #2D333B; border: 1px solid #444C56; border-radius: 8px;
              padding: 16px 20px; margin-bottom: 20px; }}
  .footer {{ color: #545D68; font-size: 10px; text-align: center; margin-top: 40px; }}
</style>
</head>
<body>
<h1>Picture Studio – Projektbericht</h1>
<p class="meta">
  Projekt: <strong style="color:#E6EDF3">{html.escape(cfg.name or "–")}</strong>
  &nbsp;|&nbsp; Erstellt: {html.escape((cfg.created_at or "")[:10])}
  &nbsp;|&nbsp; Bericht generiert: {now}
</p>

<h2>Datensatz-Übersicht</h2>
<div class="cards">
  <div class="card"><div class="val">{total_img}</div><div class="lbl">Bilder gesamt</div></div>
  <div class="card"><div class="val" style="color:#3FB950">{labeled}</div><div class="lbl">Gelabelt</div></div>
  <div class="card"><div class="val" style="color:#F85149">{unlabeled}</div><div class="lbl">Ungelabelt</div></div>
  <div class="card"><div class="val" style="color:#BC8CFF">{roi_count}</div><div class="lbl">ROIs</div></div>
  <div class="card"><div class="val" style="color:#D29922">{len(p.labels)}</div><div class="lbl">Klassen</div></div>
  <div class="card"><div class="val" style="color:#39C5CF">{len(runs)}</div><div class="lbl">Trainingsläufe</div></div>
</div>

<h2>Aktuelles Modell</h2>
<div class="section">
  <table>
    <tr><td>Modell</td><td><strong style="color:#E6EDF3">{html.escape(model_name)}</strong></td></tr>
    <tr><td>Accuracy</td><td><strong style="color:#3FB950">{acc_str}</strong></td></tr>
    <tr><td>F1-Score (Macro)</td><td><strong style="color:#3FB950">{f1_str}</strong></td></tr>
  </table>
</div>

<h2>Klassenverteilung</h2>
<div class="section">
<table>
  <thead><tr><th>Klasse</th><th>Bilder</th><th>Anteil</th><th>Verteilung</th></tr></thead>
  <tbody>{cls_rows}</tbody>
</table>
</div>

<h2>Trainingsläufe (letzte 10)</h2>
<div class="section">
<table>
  <thead><tr><th>#</th><th>Zeitstempel</th><th>Architektur</th><th>Epochen</th><th>Accuracy</th><th>F1</th><th>Modell</th></tr></thead>
  <tbody>{run_rows}</tbody>
</table>
</div>

<h2>Warnungen / Hinweise</h2>
<div class="section"><ul>{warnings_html}</ul></div>

<p class="footer">Picture Studio &nbsp;|&nbsp; Automatisch generiert am {now}</p>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path
