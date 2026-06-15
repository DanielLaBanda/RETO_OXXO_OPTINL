from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / 'outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_tables(results: list[dict[str, object]], sensitivity: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    results_by_format: dict[str, dict[str, dict[str, object]]] = {}
    for row in results:
        results_by_format.setdefault(str(row["formato"]), {})[str(row["algoritmo"])] = row

    lines = []
    lines.append("% Archivo generado por mathematical_model/nlp_model.py")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.18}")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Formato} & \\textbf{Nodos} & \\textbf{Arcos} & \\textbf{Var. NLP} & \\textbf{Var. grafo} & \\textbf{Params.} \\\\")
    lines.append("\\midrule")
    for formato, algs in results_by_format.items():
        ref = algs.get("GA main") or next(iter(algs.values()))
        lines.append(
            f"{latex_escape(formato)} & {ref['nodos']} & {ref['arcos']} & "
            f"{ref['variables_random_key']} & {ref['variables_grafo_equiv']} & {ref['parametros']} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Tamano del modelo no lineal random-key y su grafo equivalente por formato.}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.18}")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Formato} & \\textbf{Algoritmo} & \\textbf{Fitness} & "
        "\\textbf{Aprov.\\%} & \\textbf{Desp.\\%} & \\textbf{Incoh.} & "
        "\\textbf{Marca\\%} & \\textbf{E[ubic]} \\\\"
    )
    lines.append("\\midrule")
    for row in results:
        if row["algoritmo"] == "Historico":
            continue
        lines.append(
            f"{latex_escape(row['formato'])} & {latex_escape(row['algoritmo'])} & "
            f"{float(row['fitness']):.3f} & {float(row['aprovechamiento_pct']):.1f} & "
            f"{float(row['desperdicio_pct']):.1f} & {float(row['incoherencia_cm']):.2f} & "
            f"{float(row['agrupacion_marca_pct']):.1f} & {float(row['valor_esperado_ubicacion']):.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparacion del algoritmo genetico de la rama main contra el modelo no lineal random-key.}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.18}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Formato} & \\textbf{GA main} & \\textbf{NLP} & \\textbf{Dif.\\% NLP vs GA} & \\textbf{Tiempo NLP/GA} \\\\")
    lines.append("\\midrule")
    for formato, algs in results_by_format.items():
        ga = algs.get("GA main")
        nlp = algs.get("NLP random-key")
        if not ga or not nlp:
            continue
        diff = 100.0 * (float(nlp["fitness"]) - float(ga["fitness"])) / max(1e-9, abs(float(ga["fitness"])))
        if "tiempo_s" in ga and "tiempo_s" in nlp:
            ratio = float(nlp["tiempo_s"]) / max(1e-9, float(ga["tiempo_s"]))
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = "NA"
        lines.append(
            f"{latex_escape(formato)} & {float(ga['fitness']):.3f} & {float(nlp['fitness']):.3f} & "
            f"{diff:.2f} & {ratio_str} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Diferencia porcentual entre el modelo no lineal y el algoritmo genetico.}")
    lines.append("\\end{table}")
    lines.append("")

    if sensitivity:
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append("\\renewcommand{\\arraystretch}{1.18}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("\\textbf{Escenario} & \\textbf{Fitness} & \\textbf{Aprov.\\%} & \\textbf{Desp.\\%} & \\textbf{Incoh.} & \\textbf{Marca\\%} \\\\")
        lines.append("\\midrule")
        for row in sensitivity:
            lines.append(
                f"{latex_escape(row['escenario'])} & {float(row['fitness']):.3f} & "
                f"{float(row['aprovechamiento_pct']):.1f} & {float(row['desperdicio_pct']):.1f} & "
                f"{float(row['incoherencia_cm']):.2f} & {float(row['agrupacion_marca_pct']):.1f} \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Analisis de sensibilidad en BCO\\_CF\\_4.0\\_Refrescos con el optimizador no lineal.}")
        lines.append("\\end{table}")
        lines.append("")

    (OUT_DIR / "comparison_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def write_location_report(inst: Instance) -> None:
    rows = []
    for _, row in inst.prods.copy().iterrows():
        rows.append(
            {
                "UPC": str(row["UPC"]),
                "ITEM_DESC": row["ITEM_DESC"],
                "MARCA": row["MARCA"],
                "ANCHO_BLOQUE": round(float(row["ANCHO"]) * float(row["NUM_FRENTES"]), 4),
                "ALTO": round(float(row["ALTO"]), 4),
                "CHAROLA_MODAL": int(row["CHAROLA"]),
                "E_CHAROLA_NORM": round(float(row["E_CHAROLA_NORM"]), 6),
                "E_CHAROLA": round(float(row["E_CHAROLA_NORM"]) * inst.n_shelves, 4),
            }
        )
    path = OUT_DIR / f"expected_locations_{inst.formato}.csv"
    write_csv(path, rows)


def serializable_shelves(shelves: object) -> list[list[int]]:
    return [[int(j) for j in shelf] for shelf in shelves]  # type: ignore[union-attr]


def load_font(size: int = 18):
    from PIL import ImageFont

    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_line_chart(
    path: Path,
    title: str,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    x_label: str = "Iteracion normalizada",
    y_label: str = "Fitness",
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow no esta instalado; se omiten graficas PNG.")
        return

    width, height = 1100, 650
    margin_l, margin_r, margin_t, margin_b = 95, 40, 70, 95
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(18)
    title_font = load_font(28)
    small_font = load_font(15)

    left, top = margin_l, margin_t
    right, bottom = width - margin_r, height - margin_b
    values = [v for _, vals, _ in series for v in vals]
    ymin, ymax = min(values), max(values)
    pad = max(0.02, (ymax - ymin) * 0.12)
    ymin -= pad
    ymax += pad

    draw.text((width / 2, 24), title, fill=(30, 38, 48), font=title_font, anchor="ma")
    draw.line((left, bottom, right, bottom), fill=(70, 70, 70), width=2)
    draw.line((left, top, left, bottom), fill=(70, 70, 70), width=2)
    for t in range(6):
        y = bottom - t * (bottom - top) / 5
        val = ymin + t * (ymax - ymin) / 5
        draw.line((left - 6, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((left - 12, y), f"{val:.2f}", fill=(70, 70, 70), font=small_font, anchor="rm")

    for name, vals, color in series:
        if len(vals) < 2:
            continue
        pts = []
        for i, val in enumerate(vals):
            x = left + i * (right - left) / (len(vals) - 1)
            y = bottom - (val - ymin) * (bottom - top) / (ymax - ymin)
            pts.append((x, y))
        draw.line(pts, fill=color, width=4)
        draw.ellipse((pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5), fill=color)

    lx, ly = left + 20, top + 12
    for name, _vals, color in series:
        draw.rectangle((lx, ly, lx + 24, ly + 14), fill=color)
        draw.text((lx + 34, ly - 3), name, fill=(35, 35, 35), font=font)
        ly += 28

    draw.text(((left + right) / 2, height - 36), x_label, fill=(60, 60, 60), font=font, anchor="ma")
    draw.text((24, (top + bottom) / 2), y_label, fill=(60, 60, 60), font=font, anchor="mm")
    img.save(path)


def draw_grouped_bar_chart(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_label: str,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow no esta instalado; se omiten graficas PNG.")
        return

    width, height = 1200, 680
    margin_l, margin_r, margin_t, margin_b = 95, 45, 80, 140
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(17)
    title_font = load_font(28)
    small_font = load_font(14)
    left, top = margin_l, margin_t
    right, bottom = width - margin_r, height - margin_b
    ymax = max(v for _, vals, _ in series for v in vals) * 1.15
    ymax = max(ymax, 1.0)

    draw.text((width / 2, 28), title, fill=(30, 38, 48), font=title_font, anchor="ma")
    draw.line((left, bottom, right, bottom), fill=(70, 70, 70), width=2)
    draw.line((left, top, left, bottom), fill=(70, 70, 70), width=2)
    for t in range(6):
        y = bottom - t * (bottom - top) / 5
        val = t * ymax / 5
        draw.line((left - 6, y, right, y), fill=(226, 226, 226), width=1)
        draw.text((left - 12, y), f"{val:.1f}", fill=(70, 70, 70), font=small_font, anchor="rm")

    n_groups = len(labels)
    n_series = len(series)
    group_w = (right - left) / n_groups
    bar_w = min(46, group_w / (n_series + 1.4))
    for gi, label in enumerate(labels):
        cx = left + group_w * (gi + 0.5)
        base_x = cx - (n_series * bar_w + (n_series - 1) * 8) / 2
        for si, (_name, vals, color) in enumerate(series):
            x0 = base_x + si * (bar_w + 8)
            x1 = x0 + bar_w
            y1 = bottom
            y0 = bottom - vals[gi] * (bottom - top) / ymax
            draw.rectangle((x0, y0, x1, y1), fill=color)
            draw.text(((x0 + x1) / 2, y0 - 18), f"{vals[gi]:.2f}", fill=(35, 35, 35), font=small_font, anchor="ma")
        draw.text((cx, bottom + 18), label.replace("_Refrescos", ""), fill=(40, 40, 40), font=small_font, anchor="ma")

    lx, ly = left + 20, top + 8
    for name, _vals, color in series:
        draw.rectangle((lx, ly, lx + 24, ly + 14), fill=color)
        draw.text((lx + 34, ly - 3), name, fill=(35, 35, 35), font=font)
        ly += 28

    draw.text(((left + right) / 2, height - 36), 'Formato', fill=(60, 60, 60), font=font, anchor='ma')
    draw.text((24, (top + bottom) / 2), y_label, fill=(60, 60, 60), font=font, anchor='mm')
    img.save(path)
