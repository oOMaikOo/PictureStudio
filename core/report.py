"""
Generate HTML training reports with metrics, curves, confusion matrix.
"""
import json
import os
from datetime import datetime
from typing import Dict


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Trainingsbericht – {name}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#1a1a2e; color:#eee; margin:0; padding:20px; }}
h1 {{ color:#3498DB; border-bottom:2px solid #3498DB; padding-bottom:8px; }}
h2 {{ color:#2ECC71; margin-top:30px; }}
.card {{ background:#16213e; border-radius:8px; padding:20px; margin:16px 0; border:1px solid #0f3460; }}
.metric {{ display:inline-block; background:#0f3460; border-radius:6px; padding:12px 20px;
           margin:8px; text-align:center; min-width:140px; }}
.metric .value {{ font-size:2em; font-weight:bold; color:#3498DB; }}
.metric .label {{ font-size:0.85em; color:#aaa; margin-top:4px; }}
table {{ border-collapse:collapse; width:100%; margin-top:10px; }}
th {{ background:#0f3460; color:#3498DB; padding:10px; text-align:left; }}
td {{ padding:8px 10px; border-bottom:1px solid #0f3460; }}
tr:hover {{ background:#0f3460; }}
.good {{ color:#2ECC71; }}
.warn {{ color:#F39C12; }}
.bad {{ color:#E74C3C; }}
.cm-cell {{ text-align:center; padding:6px 12px; font-weight:bold; }}
.cm-diag {{ background:#155724; color:#d4edda; }}
.cm-off {{ background:#4a1313; color:#f8d7da; }}
.cm-zero {{ color:#555; }}
pre {{ background:#0f3460; padding:14px; border-radius:6px; overflow-x:auto; font-size:0.9em; }}
</style>
</head>
<body>
<h1>Trainingsbericht</h1>
<div class="card">
  <strong>Projekt:</strong> {project_name}<br>
  <strong>Run-ID:</strong> {run_id}<br>
  <strong>Zeitstempel:</strong> {timestamp}<br>
  <strong>Architektur:</strong> {architecture}<br>
  <strong>Gerät:</strong> {device}<br>
  <strong>Klassen:</strong> {classes}
</div>

<h2>Hauptmetriken</h2>
<div class="card">
{metrics_cards}
</div>

<h2>Klassen-Metriken</h2>
<div class="card">
<table>
<tr><th>Klasse</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
{class_rows}
</table>
</div>

<h2>Konfusionsmatrix</h2>
<div class="card">
{confusion_matrix}
</div>

<h2>Hyperparameter</h2>
<div class="card"><pre>{hyperparameters}</pre></div>

<h2>Softwareversionen</h2>
<div class="card"><pre>{software_versions}</pre></div>

<h2>Trainingshistorie</h2>
<div class="card">
<pre>{history_text}</pre>
</div>

<footer style="margin-top:40px;color:#555;font-size:0.8em;">
Erstellt mit Image Labeling Studio – {generated_at}
</footer>
</body>
</html>
"""


def generate_html_report(run_data: Dict, output_path: str, project_name: str = "") -> str:
    """
    Generate and save an HTML training report. Returns the path.

    Parameters
    ----------
    run_data     : Dict produced by ``TrainingWorker.run()`` containing keys
                   ``metrics``, ``history``, ``hyperparameters``, ``class_names``,
                   ``run_id``, ``timestamp``, ``model_type``, ``device``,
                   ``train_size``, ``test_size``, and ``software_versions``.
    output_path  : Absolute path where the HTML file will be written.
                   Parent directory is created automatically.
    project_name : Human-readable project name shown in the report header.

    Returns
    -------
    str
        The *output_path* that was written.
    """
    metrics = run_data.get("metrics", {})
    history = run_data.get("history", {})
    hp = run_data.get("hyperparameters", {})
    class_names = run_data.get("class_names", [])

    # Metrics cards
    def _card(value: str, label: str, css_class: str = "") -> str:
        return (
            f'<div class="metric"><div class="value {css_class}">{value}</div>'
            f'<div class="label">{label}</div></div>'
        )

    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("macro_f1", 0)
    prec = metrics.get("macro_precision", 0)
    rec = metrics.get("macro_recall", 0)

    cards = (
        _card(f"{acc*100:.2f}%", "Accuracy", "good" if acc > 0.9 else ("warn" if acc > 0.75 else "bad")) +
        _card(f"{f1*100:.2f}%", "F1 (Macro)", "good" if f1 > 0.9 else ("warn" if f1 > 0.75 else "bad")) +
        _card(f"{prec*100:.2f}%", "Precision", "") +
        _card(f"{rec*100:.2f}%", "Recall", "") +
        _card(str(run_data.get("train_size", "?")), "Train-Samples", "") +
        _card(str(run_data.get("test_size", "?")), "Test-Samples", "")
    )

    # Class rows
    class_rows = ""
    for cls, vals in metrics.get("per_class", {}).items():
        f1v = vals["f1"]
        css = "good" if f1v > 0.9 else ("warn" if f1v > 0.75 else "bad")
        class_rows += (
            f"<tr><td>{cls}</td>"
            f"<td>{vals['precision']*100:.1f}%</td>"
            f"<td>{vals['recall']*100:.1f}%</td>"
            f"<td class='{css}'>{f1v*100:.1f}%</td>"
            f"<td>{vals['support']}</td></tr>"
        )

    # Confusion matrix
    cm = metrics.get("confusion_matrix", [])
    cm_html = "<table>"
    if cm and class_names:
        cm_html += "<tr><th>T\\P</th>" + "".join(f"<th>{n}</th>" for n in class_names) + "</tr>"
        for i, row in enumerate(cm):
            cm_html += f"<tr><th>{class_names[i]}</th>"
            for j, val in enumerate(row):
                css = "cm-diag" if i == j else ("cm-off" if val > 0 else "cm-zero")
                cm_html += f'<td class="cm-cell {css}">{val}</td>'
            cm_html += "</tr>"
    cm_html += "</table>"

    # History text
    history_lines = []
    epochs = len(history.get("train_loss", []))
    for e in range(epochs):
        tl = history.get("train_loss", [])[e] if e < len(history.get("train_loss", [])) else 0
        vl = history.get("val_loss", [])[e] if e < len(history.get("val_loss", [])) else 0
        ta = history.get("train_acc", [])[e] if e < len(history.get("train_acc", [])) else 0
        va = history.get("val_acc", [])[e] if e < len(history.get("val_acc", [])) else 0
        history_lines.append(
            f"Epoch {e+1:3d}/{epochs}: train_loss={tl:.4f} val_loss={vl:.4f} "
            f"train_acc={ta*100:.1f}% val_acc={va*100:.1f}%"
        )

    html = HTML_TEMPLATE.format(
        name=run_data.get("run_id", ""),
        project_name=project_name,
        run_id=run_data.get("run_id", ""),
        timestamp=run_data.get("timestamp", "")[:19].replace("T", " "),
        architecture=run_data.get("model_type", ""),
        device=run_data.get("device", "cpu"),
        classes=", ".join(class_names),
        metrics_cards=cards,
        class_rows=class_rows,
        confusion_matrix=cm_html,
        hyperparameters=json.dumps(hp, indent=2, ensure_ascii=False),
        software_versions=json.dumps(run_data.get("software_versions", {}), indent=2),
        history_text="\n".join(history_lines),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return output_path
