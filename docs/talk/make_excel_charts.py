"""
Книга Excel с главными графиками доклада об ускоренном BSBS (docs/talk/BSBS_charts.xlsx).

Листы: время полного отбора, рост ускорения с числом признаков, применение (R² персептрона
от числа каналов), матрица возбуждения–испускания. Исходные числа — замеры и результаты
сеток проекта (источник подписан на каждом листе); производные величины — формулы Excel,
диаграммы — родные диаграммы Excel, построенные по этим ячейкам.

Запуск: python docs/talk/make_excel_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference, ScatterChart, Series
from openpyxl.chart.axis import ChartLines
from openpyxl.chart.data_source import NumDataSource, NumRef
from openpyxl.chart.error_bar import ErrorBars
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.chart.marker import Marker
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "BSBS_charts.xlsx"
IONS = ["Cu", "Ni", "Al", "Co", "Cr", "NO3"]
BLUE, ORANGE, GREY = "2A78D6", "EB6834", "5A6270"
FONT = "Arial"

title_font = Font(name=FONT, size=14, bold=True)
head_font = Font(name=FONT, size=10, bold=True)
input_font = Font(name=FONT, size=10, color="0000FF")
body_font = Font(name=FONT, size=10)
note_font = Font(name=FONT, size=9, italic=True, color="555555")
head_fill = PatternFill("solid", fgColor="E8EDF4")
thin = Side(style="thin", color="C8CED8")
box = Border(top=thin, bottom=thin, left=thin, right=thin)


def header(ws, row: int, labels: list[str], col: int = 1) -> None:
    for offset, label in enumerate(labels):
        cell = ws.cell(row=row, column=col + offset, value=label)
        cell.font, cell.fill, cell.border = head_font, head_fill, box
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def put(ws, row: int, col: int, value, number_format: str | None = None, is_input: bool = False):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = input_font if is_input else body_font
    cell.border = box
    if number_format:
        cell.number_format = number_format
    return cell


def line_style(series, color: str, width: int = 28575, dash: str | None = None, marker: str | None = "circle") -> None:
    series.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill=color, w=width, prstDash=dash))
    if marker:
        series.marker = Marker(symbol=marker, size=8)
        series.marker.graphicalProperties = GraphicalProperties(solidFill=color, ln=LineProperties(solidFill=color))
    else:
        series.marker = Marker(symbol="none")
    series.smooth = False


def show_axes(chart) -> None:
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    # Заголовки не накладываются на область построения.
    for title in (chart.title, chart.x_axis.title, chart.y_axis.title):
        if title is not None:
            title.overlay = False


def value_labels(number_format: str) -> DataLabelList:
    """Подписи точек — только значение, без имени ряда, категории и значка легенды."""
    labels = DataLabelList()
    labels.showVal = True
    labels.showSerName = False
    labels.showCatName = False
    labels.showLegendKey = False
    labels.showPercent = False
    # Формат подписи берётся из ячейки (там единицы «с» и «×»): numFmt без sourceLinked Excel игнорирует.
    return labels


def inner_layout(x: float, y: float, w: float, h: float) -> Layout:
    """Явное место области построения: иначе Excel кладёт заголовок и легенду поверх осей."""
    return Layout(manualLayout=ManualLayout(layoutTarget="inner", xMode="edge", yMode="edge", x=x, y=y, w=w, h=h))


def light_grid() -> ChartLines:
    return ChartLines(spPr=GraphicalProperties(ln=LineProperties(solidFill="D9DEE6", w=6350)))


def sheet_time(wb: Workbook) -> None:
    ws = wb.create_sheet("Время отбора")
    bench = json.loads((ROOT / "experiments/6p_fast_bsbs_benchmark/benchmark.json").read_text(encoding="utf-8"))
    ws["A1"] = "Время полного отбора BSBS: один поток, Cu²⁺, 5427 признаков, 5-кратная кросс-валидация, пакет 10%"
    ws["A1"].font = title_font
    header(ws, 3, ["Вариант", "Время, с", "Время, ч", "Во сколько раз дольше ускоренного", "Как получено"])
    rows = [
        ("Ускоренный BSBS", bench["fast_seconds_total"], "измерено"),
        ("Исходный BSBS, готовые матрицы", bench["prepared_seconds_total_estimate"], "оценка по замерам при 9 значениях p"),
        ("Исходный BSBS, обучение с нуля", bench["sklearn_seconds_total_estimate"], "оценка по замерам при 9 значениях p"),
    ]
    for i, (label, seconds, how) in enumerate(rows, start=4):
        put(ws, i, 1, label)
        put(ws, i, 2, round(float(seconds), 1), '#,##0" с"', is_input=True)
        put(ws, i, 3, f"=B{i}/3600", "0.00")
        put(ws, i, 4, f"=B{i}/$B$4", "0")
        put(ws, i, 5, how)
    ws["A8"] = "Источник: experiments/6p_fast_bsbs_benchmark/benchmark.json (benchmark_6p_fast_bsbs.py). Синие числа — исходные данные, чёрные — формулы."
    ws["A8"].font = note_font
    for col, width in zip("ABCDE", (34, 12, 11, 20, 38)):
        ws.column_dimensions[col].width = width

    chart = BarChart()
    chart.type = "bar"
    chart.title = "Время полного отбора"
    chart.y_axis.title = "Время, с (логарифмическая шкала)"
    chart.y_axis.scaling.logBase = 10
    chart.y_axis.scaling.min = 10
    chart.y_axis.scaling.max = 1_000_000
    chart.y_axis.majorGridlines = light_grid()
    chart.y_axis.number_format = "#,##0"
    chart.x_axis.scaling.orientation = "maxMin"
    # При обратном порядке категорий ось значений иначе уезжает наверх.
    chart.y_axis.crosses = "max"
    chart.legend = None
    data = Reference(ws, min_col=2, min_row=3, max_row=6)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=4, max_row=6))
    series = chart.series[0]
    series.graphicalProperties = GraphicalProperties(solidFill=ORANGE)
    from openpyxl.chart.marker import DataPoint

    point = DataPoint(idx=0)
    point.graphicalProperties = GraphicalProperties(solidFill=BLUE)
    series.dPt.append(point)
    series.dLbls = value_labels('#,##0" с"')
    chart.gapWidth = 60
    show_axes(chart)
    chart.layout = inner_layout(0.3, 0.13, 0.62, 0.66)
    chart.width, chart.height = 22, 9
    ws.add_chart(chart, "A10")


def sheet_speedup(wb: Workbook) -> None:
    ws = wb.create_sheet("Ускорение")
    small = pd.read_csv(ROOT / "experiments/6p_fast_bsbs_benchmark/small_runs.csv")
    bench = json.loads((ROOT / "experiments/6p_fast_bsbs_benchmark/benchmark.json").read_text(encoding="utf-8"))
    ws["A1"] = "Ускорение относительно исходного BSBS с заранее вычисленными матрицами"
    ws["A1"].font = title_font
    header(ws, 3, ["Число признаков p", "Исходный BSBS, с", "Ускоренный BSBS, с", "Ускорение (измерено)", "Ускорение (оценка)", "Рейтинги совпали"])
    row = 4
    for item in small.itertuples():
        put(ws, row, 1, int(item.p), "0", is_input=True)
        put(ws, row, 2, round(float(item.direct_seconds), 3), "0.000", is_input=True)
        put(ws, row, 3, round(float(item.fast_seconds), 3), "0.000", is_input=True)
        put(ws, row, 4, f"=B{row}/C{row}", '0"×"')
        put(ws, row, 5, None)
        put(ws, row, 6, "да" if bool(item.same_ranking) else "нет", is_input=True)
        row += 1
    put(ws, row, 1, 5427, "0", is_input=True)
    put(ws, row, 2, round(float(bench["prepared_seconds_total_estimate"]), 1), "#,##0.0", is_input=True)
    put(ws, row, 3, round(float(bench["fast_seconds_total"]), 1), "0.0", is_input=True)
    put(ws, row, 4, None)
    put(ws, row, 5, f"=B{row}/C{row}", '0"×"')
    put(ws, row, 6, "не проверялось целиком")
    last = row
    ws.cell(row=last + 2, column=1, value=(
        "p = 150, 300, 600: оба алгоритма выполнены целиком, рейтинги совпали полностью. "
        "p = 5427: время исходного алгоритма — оценка. Наибольшее относительное расхождение "
        "ошибки на 900+ выборочных моделях ≤ 10⁻¹⁴."
    )).font = note_font
    ws.cell(row=last + 3, column=1, value="Источник: experiments/6p_fast_bsbs_benchmark/small_runs.csv и benchmark.json.").font = note_font
    for col, width in zip("ABCDEF", (18, 18, 18, 18, 18, 22)):
        ws.column_dimensions[col].width = width

    chart = ScatterChart()
    chart.title = "Выигрыш растёт с числом признаков"
    chart.x_axis.title = "Число признаков"
    chart.y_axis.title = "Ускорение, раз"
    chart.x_axis.scaling.logBase = 10
    chart.y_axis.scaling.logBase = 10
    chart.x_axis.scaling.min, chart.x_axis.scaling.max = 100, 10000
    chart.y_axis.scaling.min, chart.y_axis.scaling.max = 10, 1000
    chart.x_axis.number_format = "#,##0"
    chart.y_axis.number_format = "0"
    chart.x_axis.majorGridlines = light_grid()
    chart.y_axis.majorGridlines = light_grid()
    chart.x_axis.crosses = "min"
    chart.y_axis.crosses = "min"
    measured = Series(Reference(ws, min_col=4, min_row=4, max_row=last - 1), Reference(ws, min_col=1, min_row=4, max_row=last - 1),
                      title="измерено")
    line_style(measured, BLUE)
    measured.dLbls = value_labels('0"×"')
    measured.dLbls.position = "t"
    estimate = Series(Reference(ws, min_col=5, min_row=last, max_row=last), Reference(ws, min_col=1, min_row=last, max_row=last),
                      title="оценка")
    line_style(estimate, BLUE, marker="circle")
    estimate.marker.graphicalProperties = GraphicalProperties(solidFill="FFFFFF", ln=LineProperties(solidFill=BLUE, w=28575))
    estimate.dLbls = value_labels('0"×"')
    estimate.dLbls.position = "t"
    chart.series.extend([measured, estimate])
    chart.legend.position = "b"
    show_axes(chart)
    chart.layout = inner_layout(0.12, 0.13, 0.84, 0.62)
    chart.width, chart.height = 18, 10
    ws.add_chart(chart, f"A{last + 5}")


def sheet_application(wb: Workbook) -> None:
    ws = wb.create_sheet("Применение")
    points = pd.read_csv(ROOT / "experiments/6p_selnorm_grid/analysis/points.csv")
    part = points[(points.method == "bsbs") & (points.channel == "zscore")]
    ks = sorted(part["k"].unique())
    ws["A1"] = "R² персептрона на тестовом наборе от числа отобранных каналов (персептрон учится на исходных спектрах)"
    ws["A1"].font = title_font
    ws["A2"] = "Нормировка спектров влияет только на выбор каналов BSBS; каналы при отборе — z-score. 3 инициализации весов."
    ws["A2"].font = note_font

    full = {}
    for ion in IONS:
        point = json.loads((ROOT / f"experiments/6p_selnorm_grid/all_channels/{ion}/curve_points.json").read_text(encoding="utf-8"))[0]
        full[ion] = (point["mean"], point["std"])

    # Исходные данные по ионам: R² (среднее по 3 seed) и его std.
    blocks = {"max": "Отбор по спектрам, нормированным на максимум", "raw": "Отбор по исходным спектрам"}
    start_row = 4
    block_rows = {}
    for spectrum, title in blocks.items():
        ws.cell(row=start_row, column=1, value=title).font = head_font
        header(ws, start_row + 1, ["k"] + [f"R² {ion}" for ion in IONS] + [f"std {ion}" for ion in IONS])
        first = start_row + 2
        for i, k in enumerate(ks):
            r = first + i
            put(ws, r, 1, int(k), "0", is_input=True)
            for j, ion in enumerate(IONS):
                row = part[(part.spectrum == spectrum) & (part.ion == ion) & (part.k == k)].iloc[0]
                put(ws, r, 2 + j, round(float(row["test"]), 4), "0.000", is_input=True)
                put(ws, r, 8 + j, round(float(row["test_std"]), 4), "0.000", is_input=True)
        block_rows[spectrum] = (first, first + len(ks) - 1)
        start_row = first + len(ks) + 1

    ws.cell(row=start_row, column=1, value="Все 5427 каналов (без отбора)").font = head_font
    header(ws, start_row + 1, ["k"] + [f"R² {ion}" for ion in IONS] + [f"std {ion}" for ion in IONS])
    full_row = start_row + 2
    put(ws, full_row, 1, 5427, "0", is_input=True)
    for j, ion in enumerate(IONS):
        put(ws, full_row, 2 + j, round(full[ion][0], 4), "0.000", is_input=True)
        put(ws, full_row, 8 + j, round(full[ion][1], 4), "0.000", is_input=True)

    # Сводная таблица для диаграммы: средние по ионам и их погрешность — формулы.
    summary_row = full_row + 3
    ws.cell(row=summary_row - 1, column=1, value="Среднее по 6 ионам (формулы); погрешность — корень из суммы квадратов std / 6").font = head_font
    header(ws, summary_row, ["k", "R², отбор по нормированным на максимум", "± (макс.)", "R², отбор по исходным", "± (исх.)", "R², все каналы"])
    first_max, _ = block_rows["max"]
    first_raw, _ = block_rows["raw"]
    for i, _ in enumerate(ks):
        r = summary_row + 1 + i
        source_max, source_raw = first_max + i, first_raw + i
        put(ws, r, 1, f"=A{source_max}", "0")
        put(ws, r, 2, f"=AVERAGE(B{source_max}:G{source_max})", "0.000")
        put(ws, r, 3, f"=SQRT(SUMSQ(H{source_max}:M{source_max}))/6", "0.0000")
        put(ws, r, 4, f"=AVERAGE(B{source_raw}:G{source_raw})", "0.000")
        put(ws, r, 5, f"=SQRT(SUMSQ(H{source_raw}:M{source_raw}))/6", "0.0000")
        put(ws, r, 6, f"=AVERAGE($B${full_row}:$G${full_row})", "0.000")
    summary_last = summary_row + len(ks)
    ws.cell(row=summary_last + 2, column=1, value=(
        "Источник: experiments/6p_selnorm_grid/analysis/points.csv и all_channels/<ион>/curve_points.json "
        "(run_6p_selnorm_grid.py, сервер, разбиение CV1)."
    )).font = note_font
    ws.column_dimensions["A"].width = 12
    for col in range(2, 14):
        ws.column_dimensions[get_column_letter(col)].width = 13

    chart = ScatterChart()
    chart.title = "Применение: отбор каналов для нейросетевого решения"
    chart.x_axis.title = "Число отобранных каналов"
    chart.y_axis.title = "R² на тестовом наборе"
    chart.x_axis.scaling.logBase = 10
    chart.x_axis.scaling.min, chart.x_axis.scaling.max = 1, 10000
    chart.y_axis.scaling.min, chart.y_axis.scaling.max = 0.45, 0.78
    chart.x_axis.number_format = "#,##0"
    chart.y_axis.number_format = "0.00"
    chart.x_axis.majorGridlines = None
    chart.y_axis.majorGridlines = light_grid()
    chart.x_axis.crosses = "min"
    chart.y_axis.crosses = "min"
    xs = Reference(ws, min_col=1, min_row=summary_row + 1, max_row=summary_last)
    for col, err_col, color, title in ((2, 3, BLUE, "отбор по спектрам, нормированным на максимум"), (4, 5, ORANGE, "отбор по исходным спектрам")):
        series = Series(Reference(ws, min_col=col, min_row=summary_row + 1, max_row=summary_last), xs, title=title)
        line_style(series, color)
        error = NumDataSource(numRef=NumRef(f="'Применение'!${0}${1}:${0}${2}".format(get_column_letter(err_col), summary_row + 1, summary_last)))
        series.errBars = ErrorBars(errDir="y", errBarType="both", errValType="cust", plus=error, minus=error, noEndCap=False)
        chart.series.append(series)
    full_series = Series(Reference(ws, min_col=6, min_row=summary_row + 1, max_row=summary_last), xs, title="все 5427 каналов")
    line_style(full_series, GREY, width=19050, dash="dash", marker=None)
    chart.series.append(full_series)
    show_axes(chart)
    chart.legend.position = "b"
    chart.layout = inner_layout(0.09, 0.1, 0.88, 0.66)
    chart.width, chart.height = 22, 12
    ws.add_chart(chart, f"H{summary_row}")


def sheet_eem(wb: Workbook) -> None:
    ws = wb.create_sheet("Матрица ВИ")
    blob = np.load(ROOT / "experiments/6p_full_patience300/xy_cv1.npz")
    X, y = blob["X"], blob["y"]
    row_index = int(np.flatnonzero((y[:, :5] == 0).all(axis=1))[0])
    cube = X[row_index].reshape(27, 201)
    ws["A1"] = "Матрица возбуждения–испускания ФЛ раствора УТ без солей (интенсивность, отн. ед.)"
    ws["A1"].font = title_font
    ws["A2"] = (f"Строка {row_index + 1} массива CV1 (все катионы 0 мМ). Строки — длина волны возбуждения, "
                "столбцы — испускания, нм. Цвет — условное форматирование (Excel не строит тепловые карты как диаграммы).")
    ws["A2"].font = note_font
    ws.cell(row=4, column=1, value="Возб. \\ Исп., нм").font = head_font
    for j in range(201):
        cell = ws.cell(row=4, column=2 + j, value=375 + j)
        cell.font = head_font
        cell.alignment = Alignment(text_rotation=90, horizontal="center")
        ws.column_dimensions[get_column_letter(2 + j)].width = 2.6
    for i in range(27):
        ws.cell(row=5 + i, column=1, value=280 + 5 * i).font = head_font
        for j in range(201):
            cell = ws.cell(row=5 + i, column=2 + j, value=round(float(cube[i, j]), 2))
            cell.font = Font(name=FONT, size=6, color="FFFFFF")
            cell.number_format = ";;;"
    ws.column_dimensions["A"].width = 16
    for i in range(27):
        ws.row_dimensions[5 + i].height = 15
    area = f"B5:{get_column_letter(202)}31"
    ws.conditional_formatting.add(
        area,
        ColorScaleRule(start_type="num", start_value=0, start_color="F7F6F2",
                       mid_type="percentile", mid_value=60, mid_color="6DA7EC",
                       end_type="percentile", end_value=99, end_color="0D366B"),
    )
    ws.freeze_panes = "B5"
    ws.cell(row=33, column=1, value="Значения в ячейках скрыты форматом «;;;» — выделите ячейку, чтобы увидеть число.").font = note_font


def main() -> None:
    wb = Workbook()
    readme = wb.active
    readme.title = "Описание"
    lines = [
        ("Графики доклада «Ускоренный алгоритм пакетного обратного отбора входных признаков…»", title_font),
        ("Время отбора — сравнение ускоренного и исходного BSBS (слайд «Полный отбор — 3 минуты вместо часов»).", body_font),
        ("Ускорение — рост выигрыша с числом признаков и проверка совпадения (слайд «Выигрыш растёт…»).", body_font),
        ("Применение — R² персептрона от числа отобранных каналов (слайд «Применение»).", body_font),
        ("Матрица ВИ — пример матрицы возбуждения–испускания (слайд «Обратная задача…»).", body_font),
        ("Синие числа — исходные данные из расчётов проекта, чёрные — формулы Excel. Диаграммы ссылаются на ячейки.", note_font),
        ("Файл собирает docs/talk/make_excel_charts.py.", note_font),
    ]
    for i, (text, font) in enumerate(lines, start=1):
        readme.cell(row=i, column=1, value=text).font = font
    readme.column_dimensions["A"].width = 110
    sheet_time(wb)
    sheet_speedup(wb)
    sheet_application(wb)
    sheet_eem(wb)
    wb.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
