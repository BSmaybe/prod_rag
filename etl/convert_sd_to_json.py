#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, time, re
from pathlib import Path
import pandas as pd

# ---- Алиасы под частые названия колонок SD (можно расширять) ----
ALIASES = {
    # Ключевые поля
    "id": ["Номер запроса", "ID", "Идентификатор"],
    "type": ["Вид запроса", "Тип"],
    "summary": ["Краткое описание", "Кратко", "Тема", "Описание (кратко)"],
    "description": ["Описание", "Подробное описание"],
    "service": ["Услуга", "Сервис"],
    "resolution": ["Описание решения", "Результат работ", "Решение"],
    "assignee": ["Кем решен (сотрудник)", "Исполнитель"],
    "created": ["Дата/время регистрации", "Создано"],

    # Доп. поля, которые мы кладём в metadata / текст
    "status": ["Текущий статус", "Статус"],
    "system_status": ["Системный статус"],
    "effort": ["Фактическое время выполнения"],
    "sla_usage": ["Процент использования срока"],
    "component": ["Компонент", "Модуль"],
    "env": ["Среда", "Окружение"],
    "version": ["Версия", "Release"],
    "updated": ["Дата/время обновления", "Обновлено"],
    "resolved": ["Дата/время выполнения", "Решено", "Закрыто"],
    "comments": ["Комментарии", "Комментарий"],
    "workaround": ["Временное решение", "Обходное решение"],
    "rca": ["Причина инцидента", "RCA"],
    "impact": ["Влияние", "Воздействие"],
    "labels": ["Метки", "Теги", "Ключевые слова"],
}

UP = "A-ZА-ЯЁӘІҢҒҮҰҚӨҺ"
LW = "a-zа-яёәіңғүұқөһ"

def normalize_headers(cols):
    """Строим карту: логическое поле -> список реальных (оригинальных) имён колонок."""
    mapping = {k: [] for k in ALIASES.keys()}
    original = [str(c).strip() for c in cols]
    for idx, name in enumerate(original):
        for key, variants in ALIASES.items():
            if any(v.lower() == name.lower() for v in variants):
                mapping[key].append(cols[idx])  # сохраняем исходное имя
    return mapping

def _to_scalar(v):
    if pd.isna(v):
        return ""
    if isinstance(v, pd.Timestamp):
        try:
            return v.to_pydatetime().isoformat()
        except Exception:
            return str(v)
    s = str(v).strip()
    return "" if s in ("nan", "NaT") else s

def coalesce_vals(row, candidate_cols):
    """Берём первый непустой скаляр из списка колонок (учёт дублей имён)."""
    for col in candidate_cols or []:
        if col not in row.index:
            continue
        val = row[col]
        if isinstance(val, pd.Series):
            for v in val:
                s = _to_scalar(v)
                if s:
                    return s
        else:
            s = _to_scalar(val)
            if s:
                return s
    return ""

def explode_labels(val):
    if not val:
        return []
    parts = re.split(r"[;,/#\|\s]+", str(val))
    return [p.strip() for p in parts if p.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", help="Путь к .xlsx выгрузке SD")
    ap.add_argument("--sheet", default=0, help="Имя или индекс листа (default: 0)")
    ap.add_argument("--header", type=int, default=0, help="Индекс строки заголовков (default: 0)")
    ap.add_argument("--outdir", default="data/raw", help="Куда писать JSON (default: data/raw)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name=args.sheet, header=args.header)
    # оставляем оригинальные имена (для возврата значений), но без хвостовых пробелов
    df.columns = [str(c).strip() for c in df.columns]

    colmap = normalize_headers(df.columns)

    if args.debug:
        print("COLUMNS:", df.columns.tolist())
        for k, v in colmap.items():
            if v:
                print(f"  {k:<12} <- {v}")

    records = []
    for _, row in df.iterrows():
        rid = coalesce_vals(row, colmap.get("id"))
        summary = coalesce_vals(row, colmap.get("summary")) or coalesce_vals(row, colmap.get("description"))
        issue_key = rid or (summary[:64] if summary else "")
        if not issue_key:
            continue

        rec_type   = coalesce_vals(row, colmap.get("type"))
        service    = coalesce_vals(row, colmap.get("service"))
        component  = coalesce_vals(row, colmap.get("component"))
        env        = coalesce_vals(row, colmap.get("env"))
        version    = coalesce_vals(row, colmap.get("version"))
        created    = coalesce_vals(row, colmap.get("created"))
        updated    = coalesce_vals(row, colmap.get("updated"))
        resolved   = coalesce_vals(row, colmap.get("resolved"))
        dt         = resolved or updated or created or ""

        description = coalesce_vals(row, colmap.get("description"))
        comments    = coalesce_vals(row, colmap.get("comments"))
        workaround  = coalesce_vals(row, colmap.get("workaround"))
        resolution  = coalesce_vals(row, colmap.get("resolution"))
        rca         = coalesce_vals(row, colmap.get("rca"))
        impact      = coalesce_vals(row, colmap.get("impact"))
        assignee    = coalesce_vals(row, colmap.get("assignee"))
        labels      = explode_labels(coalesce_vals(row, colmap.get("labels")))

        merged_text = []
        if summary:
            merged_text.append(f"SUMMARY:\n{summary}")
        # не дублируем, если описание == краткому
        if description and description.strip() != (summary or "").strip():
            merged_text.append(f"DESCRIPTION:\n{description}")
        if comments:
            merged_text.append(f"COMMENTS:\n{comments}")
        if workaround:
            merged_text.append(f"WORKAROUND:\n{workaround}")
        if resolution:
            merged_text.append(f"RESOLUTION:\n{resolution}")
        if rca:
            merged_text.append(f"RCA:\n{rca}")
        if impact:
            merged_text.append(f"IMPACT:\n{impact}")

        rec = {
            "source": "SD",
            "issue_key": str(issue_key),
            "type": rec_type,
            "service": service,
            "component": component,
            "env": env,
            "version": version,
            "dt": dt,
            "summary": summary,
            "description": "\n\n".join(merged_text),
            "steps": "",
            "workaround": workaround,
            "resolution": resolution,
            "rca_summary": rca,
            "customer_impact": impact,
            "metadata": {
                "assignee": assignee,
                "labels": labels,
                "created_raw": created,
                "updated_raw": updated,
                "resolved_raw": resolved,
                "status": coalesce_vals(row, colmap.get("status")),
                "system_status": coalesce_vals(row, colmap.get("system_status")),
                "effort": coalesce_vals(row, colmap.get("effort")),
                "sla_usage": coalesce_vals(row, colmap.get("sla_usage")),
            },
        }
        records.append(rec)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"sd_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(records)} records -> {fn}")

if __name__ == "__main__":
    main()