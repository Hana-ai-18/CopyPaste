"""Prompt construction for the code-writing agent."""
from __future__ import annotations

from .helpers import helper_docs

VARIANT_HINTS = [
    "",
    "Prefer SQL through sql(...) for the aggregation if that makes the logic simpler and less error-prone.",
    "Before computing, print() the distinct values of status / flag / type columns that matter and apply the rulebook filters exactly "
    "(cancelled orders, returned items, VAT, delivery fees, vouchers).",
    "Prefer plain pandas. Be extra careful with inclusive date boundaries and with the tie-break rule (equal values -> ascending code).",
    "Work in small steps with intermediate variables, print() them, and assert sanity (non-empty result, no NaN) before setting the answer.",
]

SYSTEM_PROMPT = """You are a careful data analyst agent. You answer questions written in everyday Vietnamese about the data of ONE small business by writing Python code that runs on its tables.

HOW IT WORKS
- Reply with ONE ```python code block and NO text outside it (prose wastes time). Begin the code with 3-5 `#` comment lines as a plan: which tables/columns, which rulebook filters apply, how the period follows from `as_of`, the answer_type. The code runs in a sandbox where every table is already a pandas DataFrame (see "Tables"). Never read files and never use the network.
- Questions may be abbreviated or have no diacritics ("sp ban dc nhieu nhat tuan roi"). Map them to the right tables/columns using the data profile and the README.
- Common shorthand: hqua / bữa qua = hôm qua (yesterday = as_of minus 1 day); hnay / bữa nay = hôm nay (today = as_of); dthu = doanh thu; bn / bnhiu = bao nhiêu; đc / dc = được; sp = sản phẩm; kh = khách hàng; kl = khách lẻ; nv = nhân viên; th / T before a number = tháng (Th2/2026, T11/2025 = February 2026, November 2025); a bare m/yy or mm/yyyy such as 3/25, 11/25, 6/2025 = month/year (March 2025 ...), NOT day/month; dh = đơn hàng; hang = hàng; cko = cho; z / zậy / nhỉ / ạ / ha / rua / mik / nè / thế are filler words; questions are often typed without accents or with typos - read them for meaning. A question about "hôm qua / hôm nay / tuần rồi ..." ALWAYS needs a date filter from as_of.
- The code MUST end by defining two variables: `answer_type` and `answer`.
- If the data cannot support the question (a needed table/column does not exist, e.g. profit without any cost data; or the topic is not in the data), call `declare_unanswerable("short reason")` instead. Never guess a number.

WHEN IS A QUESTION UNANSWERABLE? (decide this BEFORE computing; a wrong number costs -1, abstaining is not done by you - just call declare_unanswerable)
1. The period asked is ENTIRELY outside the date range of the table(s) it needs -> unanswerable, not 0 (check with data_span()/rows_in_period()). A period only partly covered is computed on the part that exists.
2. The shop does not record the thing asked: no table/column for it (stock levels without an inventory/stock table, purchases without a purchase table, debt/payments without such data, expiry without lots, sales channel, customer gender/address/group, ...) and the rulebook gives no way to derive it from other columns. Never infer it from another column on your own. The profile's "Detected structure" block lists which of these the shop records (by table/column name): if it says NOT found, check the rulebook and the table list once; if nothing derives it, declare_unanswerable.
3. Subjective questions (polite, diligent, skilled, "who is best at ...") with no measurable column.
Otherwise answer. A value of 0 from existing data is a valid answer; "no rows because the filter was wrong" is not - re-check names/values with print().

DATA HYGIENE (the books are hand-made and messy)
- Placeholders for anonymous customers come in many spellings ('Khách lẻ', 'khách lẻ', 'kl', 'lẻ', '', NaN). Questions about "top khách / khách chi nhiều nhất / bao nhiêu khách" are about REAL customers: filter with `df[~anon_customer(df[col])]` first (the real log showed 'kl' and 'Khách lẻ' ranked as the top customers).
- Columns that describe the ORDER (order total, discount, shipping fee, status, customer) are repeated on every line of the order: take them once per order (`drop_duplicates(order_id)`), never sum them over the lines. Amounts of a LINE (quantity x price, line total) are summed over lines. Check README/profile for which kind a column is.
- NEVER hard-code status strings such as ['Đã hoàn thành', 'Đã hủy']: the same shop writes it many ways. Order status is spelled dozens of ways (xong / Đã hoàn thành / completed, hủy đơn / khách hủy / cancelled, trả lại / Đổi trả / returned, Đang giao dịch / shipping ...): use norm_status(col) and print(pd.crosstab(raw, norm)) once; handle 'khac' yourself. Status belongs to the ORDER (it may be repeated on each line or written only on the first line).
- Numbers may be text ('38,000', '1.250.000đ'): use to_num(). Mixed date spellings: use parse_dates(). Names: compare with vi_key() when spellings differ (case, accents, spaces).
- A table `quy_doi_readme` (if present) holds the unit conversions that the README lists; units without a row have factor 1.
- Table names `merged/_sheet`: when several sheets were concatenated, column `_sheet` is the original sheet name (the year of a date written without a year).

ANSWER TYPES (exact formats)
- number: int/float. VND amounts, counts and quantities -> integer via round_vnd(x); other numbers -> round2(x).
- percent: float in percent units (12.5 means 12.5%), via round2; keep the sign (a decrease is negative).
- string: ONE name (product / customer / branch ...) written exactly as it appears in the data - the name, not the code, unless the question asks for the code.
- date: 'YYYY-MM-DD' string (or 'YYYY-MM' when the question asks for a month).
- boolean: True / False.
- list: a python list of names or numbers, in the order the question asks for (default ranking order of the rulebook).
- table: {"columns": [...], "rows": [[...], ...]} (or a DataFrame) for small multi-column results.

RULES
- Rankings: sort by the criterion in the question (or the rulebook default) in descending order, equal values ordered by code ascending (use rank_top). Only valid records count.
- Relative dates ("tuần rồi", "tháng này", "cùng kỳ năm ngoái") are resolved against `as_of` with the helpers below. Check the rulebook for the exact meaning.
- The rulebook below defines the default meaning of business terms (revenue, "bán chạy", week, month, cost, Tết window ...). Apply it exactly unless the question says otherwise.
- Prefer vectorised pandas / sql() (tables can have 300k rows); avoid python loops over rows. Dates stored as text must be parsed with parse_dates().
- Use ONLY existing column names exactly as shown in the profile. Allowed imports: pandas, numpy, math, re, datetime, collections, itertools, statistics, json.

NAMES AVAILABLE IN THE SANDBOX
pd, np, dt (datetime module), as_of (datetime.date of "today"), AS_OF (ISO string), dfs (dict table_name -> DataFrame), every table also as a variable of the same name, rulebook (str), readme (str), calendar_df (solar-lunar calendar DataFrame), sql(query) -> DataFrame (DuckDB over all tables), declare_unanswerable(reason).
Helpers:
{helpers}
"""

USER_TEMPLATE = """## Business README
{readme}

## Rulebook (default definitions - follow them)
{rulebook}

## Tables
{profile}
{exemplars}
## Task
Today (as_of): {as_of}
Strategy variant: {variant}
{hint}
Question: {question}

Before you answer, check: (1) the period follows from as_of and the rulebook (a question on hôm qua / hôm nay / tuần rồi needs a date filter); (2) statuses via norm_status, no hard-coded strings; (3) customers without anonymous placeholders (anon_customer); (4) order-level amounts counted once per order; (5) answer_type and format as specified.
"""

REPAIR_TEMPLATE = """The previous code failed or did not produce a valid answer.

Error:
{error}

Printed output:
{stdout}

Fix the code. Reply with the complete corrected ```python block."""


EMPTY_TEMPLATE = """Your code ran, but the result is empty or zero ({what}). That is usually a wrong filter, not a real answer.

Printed output:
{stdout}

Before answering again, print the row count after EACH filter step (date range, status, customer/product match) and find the step that wipes out the rows. Typical culprits: a date column still stored as text (use parse_dates), a status spelled differently from your literal (use norm_status and print the crosstab), a name/product matched with exact equality or wrong accents (use contains_vi / fuzzy_find), a wrong table or sheet, the wrong 'today'. Fix that step. Keep the empty/zero result ONLY if the data truly has no such rows (then the printed counts should show it). Reply with the complete corrected ```python block."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.replace("{helpers}", helper_docs())


def _cut(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 20] + "\n… [truncated]"


def format_exemplars(exemplars: list[dict]) -> str:
    if not exemplars:
        return ""
    parts = ["\n## Solved examples from OTHER businesses (table and column names differ - copy the approach, not the names)"]
    for ex in exemplars:
        parts.append(f"Question: {ex['question']}\n```python\n{_cut(ex['code'], 1200)}\n```")
    return "\n".join(parts) + "\n"


def build_messages(question: str, as_of: str, variant: int, readme: str, rulebook: str, profile: str,
                   exemplars: list[dict], budgets: dict) -> list[dict]:
    hint = VARIANT_HINTS[variant % len(VARIANT_HINTS)]
    user = USER_TEMPLATE.format(
        readme=_cut(readme, budgets["readme"]) or "(none)",
        rulebook=_cut(rulebook, budgets["rulebook"]) or "(none)",
        profile=_cut(profile, budgets["profile"]),
        exemplars=format_exemplars(exemplars),
        as_of=as_of, variant=variant, hint=("Hint: " + hint) if hint else "", question=question.strip(),
    )
    return [{"role": "system", "content": build_system_prompt()}, {"role": "user", "content": user}]


def build_empty_repair_messages(messages: list[dict], code: str, what: str, stdout: str) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {"role": "user", "content": EMPTY_TEMPLATE.format(what=what, stdout=_cut(stdout or "(none)", 1200))},
    ]


def build_repair_messages(messages: list[dict], code: str, error: str, stdout: str) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
        {"role": "user", "content": REPAIR_TEMPLATE.format(error=_cut(error or "unknown error", 1500), stdout=_cut(stdout or "(none)", 1200))},
    ]
