# 입력: data/reference/hanacard19149_benefits.csv (하나카드 신용카드 CD_PD_SEQ=19149 혜택표 —
#         사용자가 공식 상품 페이지 공시 내용을 전사. 영역 6개(A~F), 적립률/월한도/한도공유그룹/
#         업종매핑(원더카드 2.0에서 쓴 매핑을 그대로 재사용) 포함)
#       config/assumptions.yaml (merchant_fee_rate, g_star_threshold,
#         avg_monthly_transactions_per_card — 재정의하지 않고 로드만 한다)
#       outputs/tables/segment_industry_amt.csv (전가맹점 영역의 가중평균 f, 업종별 월평균 지출)
#       data/processed/02_cleaned.parquet (업종중분류 대리 지표 — 계산2 비교용)
# 출력: outputs/tables/hana19149_nominal_gstar.csv   (계산1: 한도 없는 순수 적립률 기준 g*)
#       outputs/tables/hana19149_limit_hit.csv       (계산2: 한도 도달 지출액 vs 실측 월평균 지출)
#       outputs/tables/hana19149_breakeven.csv       (계산3: 한도 하 손익분기 필요 소비액 S)
#       outputs/tables/hana19149_shared_limit.csv    (계산4: 통합한도 vs 개별한도)
#       outputs/tables/hana19149_streak_sweep.csv    (계산5: 3일 연속 조건 비중 q 민감도)
#       outputs/tables/hana19149_spend_ref.csv       (계산2 근거: 카드 1장당 월평균 지출 추정)
# 목적: H10(실제 시장 상품 대조)·H11~H14(원더카드 2.0 전체 투입)에 이어, 한도 구조가
#       전혀 다른 실제 상품(통합한도 + 전월실적 조건 없음 + 일실적/연속이용 조건)을
#       같은 손익분기 모델에 투입한다 (H15~H18).
#
# 수식은 전부 기존 코드에서 import해 쓴다 (재정의 금지):
#   src/04_economics.py          g_star(f, r) = r / (f - r),  f <= r 이면 inf
#   src/06_product_wondercard.py min_spend_for_threshold(f, r, L, t) = (1+t)/t * L/f
#                                weighted_fee_for_allmerchant(fee_rates)  ('전가맹점' 영역용)
#
# 주의(CLAUDE.md 규칙 1·3):
#   - A(국내외 가맹점)의 적립 한도는 상품 페이지에 명시가 없다. 추정해서 채우지 않고
#     계산3을 빈칸으로 두고 사유를 적는다.
#   - 영역->업종 매핑은 원더카드 2.0 분석에서 쓴 사용자 가정치를 그대로 재사용한 것이다.
#     새 매핑을 만들지 않았다. 컬럼명에 (사용자가정·재사용)을 명시한다.
#   - 업종중분류 대리(종합소매점/음식배달서비스/교통서비스)는 이 스크립트에서 새로 붙인
#     가정이다. 계산2의 참고 열로만 쓰고 (신규가정)으로 표시한다. f 산정에는 쓰지 않는다.
#   - 1 하나머니 = 1원 (사용자 지정 전제).

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BENEFITS_PATH = Path("data/reference/hanacard19149_benefits.csv")
CONFIG_PATH = Path("config/assumptions.yaml")
SEGMENT_INDUSTRY_PATH = Path("outputs/tables/segment_industry_amt.csv")
CLEANED_PATH = Path("data/processed/02_cleaned.parquet")
ECONOMICS_PATH = Path("src/04_economics.py")
WONDERCARD_PATH = Path("src/06_product_wondercard.py")

OUT_NOMINAL = Path("outputs/tables/hana19149_nominal_gstar.csv")
OUT_LIMIT_HIT = Path("outputs/tables/hana19149_limit_hit.csv")
OUT_BREAKEVEN = Path("outputs/tables/hana19149_breakeven.csv")
OUT_SHARED = Path("outputs/tables/hana19149_shared_limit.csv")
OUT_STREAK = Path("outputs/tables/hana19149_streak_sweep.csv")
OUT_SPEND = Path("outputs/tables/hana19149_spend_ref.csv")

UNMAPPABLE = "전가맹점"      # 단일 업종 f를 배정할 수 없는 영역 표식 (원더카드 2.0과 동일)
N_MONTHS = 4                 # 03_segment.py와 동일 (2025-12 ~ 2026-03)
EXCLUDED_AGES = {1, 11}      # 03_segment.py와 동일 (0-9세, 100세 이상)
# 중분류 대리지표가 같은 대분류 대비 이 비율 미만이면 "대리지표로 못 쓴다"고 본다.
# (예: 배달앱 결제는 앱 사업자 가맹점으로 잡혀 지역 오프라인 데이터에 거의 안 남는다)
MID_PROXY_MIN_SHARE = 0.01
Q_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)   # 계산5: 조건 충족일 결제액 비중
ALLOC_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)  # 계산4: 통합한도 내 배분 비중


def load_module(path: Path, name: str):
    """숫자 접두사 때문에 일반 import가 안 되므로 importlib 경유. 두 스크립트 모두
    `if __name__ == "__main__"` 가드가 있어 import만으로 main()이 돌지 않는다."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


econ = load_module(ECONOMICS_PATH, "economics_04")
wonder = load_module(WONDERCARD_PATH, "wondercard_06")
g_star = econ.g_star
min_spend_for_threshold = wonder.min_spend_for_threshold
weighted_fee_for_allmerchant = wonder.weighted_fee_for_allmerchant


def parse_pct(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(str(value).strip().rstrip("%")) / 100


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_benefits() -> pd.DataFrame:
    if not BENEFITS_PATH.exists():
        raise SystemExit(f"[중단] {BENEFITS_PATH} 없음.")
    return pd.read_csv(BENEFITS_PATH, encoding="utf-8")


# ---------------------------------------------------------------- 공통: f 배정
def resolve_fee(mapping: str, fee_rates: dict, f_weighted: float):
    """영역의 업종매핑(원더카드 2.0 재사용)으로 f를 배정한다.
    '전가맹점'은 단일 업종 배정이 불가능하므로 06_product_wondercard.py가 제안한
    '실측 업종별 매출비중 가중평균 f'를 그대로 쓴다(새로 만들지 않음)."""
    if mapping == UNMAPPABLE:
        return f_weighted, "실측 소비구성비 가중평균(06_product_wondercard.py 방식 재사용)"
    return fee_rates[mapping], f"업종매핑 {mapping} (사용자가정·원더카드2.0에서 재사용)"


# ---------------------------------------------------------------- 계산 1
def build_nominal(benefits: pd.DataFrame, fee_rates: dict, f_weighted: float, threshold: float) -> pd.DataFrame:
    rows = []
    for _, b in benefits.iterrows():
        mapping = str(b["업종매핑(원더카드2.0재사용)"]).strip()
        f, f_basis = resolve_fee(mapping, fee_rates, f_weighted)
        cases = [("기본", parse_pct(b["기본적립률"]))]
        r_streak = parse_pct(b["연속3일적립률"])
        if r_streak is not None:
            cases.append(("3일연속", r_streak))
        for case, r in cases:
            if r is None:
                continue
            g = g_star(f, r)
            rows.append(
                {
                    "영역코드": b["영역코드"],
                    "영역명": b["영역명"],
                    "적립률구분": case,
                    "r(적립률)": r,
                    "업종매핑(사용자가정·재사용)": mapping,
                    "f(수수료율)": f,
                    "f 산정근거": f_basis,
                    "g*(한도무시·순수적립률)": g,
                    "손익분기_미성립(f<=r)": bool(not np.isfinite(g)),
                    f"threshold({threshold:.0%})_충족": bool(np.isfinite(g) and g <= threshold),
                    "비고": "f <= r 이라 어떤 증분 이용률로도 회수 불가" if not np.isfinite(g) else "",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 계산 2 근거: 월평균 지출
def build_spend_reference(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """카드 1장당 월평균 지출 추정.
      N(세그먼트) = 월추정이용건수 합 / avg_monthly_transactions_per_card (config 가정치)
      1장당 월평균 지출(업종) = 월추정매출액(업종) / N
    04_economics.py가 연회비 풀을 추정할 때 쓴 환산식과 같은 방식이다(재정의 아님)."""
    if not SEGMENT_INDUSTRY_PATH.exists():
        raise SystemExit(f"[중단] {SEGMENT_INDUSTRY_PATH} 없음. 먼저 03_segment.py 실행할 것.")
    seg = pd.read_csv(SEGMENT_INDUSTRY_PATH, encoding="utf-8-sig")
    tx_per_card = cfg["avg_monthly_transactions_per_card"]

    n_cards_total = seg["월추정이용건수"].sum() / tx_per_card
    amt_total = seg["월추정매출액"].sum()

    rows = []
    by_ind = seg.groupby("card_tpbuz_nm_1")[["월추정매출액", "월추정이용건수"]].sum()
    for ind, row in by_ind.sort_values("월추정매출액", ascending=False).iterrows():
        rows.append(
            {
                "구분": "업종대분류(실측)",
                "업종": ind,
                "월추정매출액(원)": row["월추정매출액"],
                "카드1장당_월평균지출(원)": row["월추정매출액"] / n_cards_total,
                "근거": "outputs/tables/segment_industry_amt.csv / N=이용건수÷15(config 가정치)",
            }
        )
    rows.append(
        {
            "구분": "업종대분류(실측)",
            "업종": "전체(9개 업종 합)",
            "월추정매출액(원)": amt_total,
            "카드1장당_월평균지출(원)": amt_total / n_cards_total,
            "근거": "outputs/tables/segment_industry_amt.csv / N=이용건수÷15(config 가정치)",
        }
    )

    # 업종중분류 대리 — 계산2 참고열 전용 (신규가정). 03_segment.py와 같은 필터로 재집계.
    mid = {}
    if CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH, columns=["card_tpbuz_nm_1", "card_tpbuz_nm_2", "amt", "cnt", "is_inflow_area", "age"])
        df = df[~df["is_inflow_area"] & ~df["age"].isin(EXCLUDED_AGES)]
        # 검증: 03_segment.py의 집계와 같은 모집단인지 확인 (다르면 비교가 성립하지 않음)
        recomputed = df["amt"].sum() / N_MONTHS
        if abs(recomputed - amt_total) / amt_total > 1e-6:
            raise SystemExit(
                f"[중단] 모집단 불일치. 재집계 월매출 {recomputed:,.0f} vs "
                f"segment_industry_amt.csv 합계 {amt_total:,.0f}. 필터 조건을 맞출 것."
            )
        m = df.groupby("card_tpbuz_nm_2", observed=True)["amt"].sum() / N_MONTHS
        mid = m.to_dict()
        for name, amt in m.sort_values(ascending=False).items():
            rows.append(
                {
                    "구분": "업종중분류(신규가정 대리지표)",
                    "업종": name,
                    "월추정매출액(원)": amt,
                    "카드1장당_월평균지출(원)": amt / n_cards_total,
                    "근거": "data/processed/02_cleaned.parquet 재집계(03_segment.py와 동일 필터)",
                }
            )
    spend = pd.DataFrame(rows)
    meta = {"n_cards_total": n_cards_total, "amt_total": amt_total, "mid": mid, "tx_per_card": tx_per_card}
    return spend, meta


def mid_proxy_spend(expr, meta: dict):
    """'종합소매점+음/식료품소매' 같은 합산 표현을 카드 1장당 월평균 지출로 환산."""
    if pd.isna(expr) or str(expr).strip() == "":
        return None, None
    expr = str(expr).strip()
    if expr == "전체":
        return meta["amt_total"] / meta["n_cards_total"], "전체 업종 합"
    parts = [p.strip() for p in expr.split("+")]
    missing = [p for p in parts if p not in meta["mid"]]
    if missing:
        return None, f"중분류 미존재: {', '.join(missing)}"
    total = sum(meta["mid"][p] for p in parts)
    return total / meta["n_cards_total"], expr


# ---------------------------------------------------------------- 계산 2
def build_limit_hit(benefits: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """한도 L에 도달하는 월 지출 = L / r. 적립률별로 계산하고, 그 금액이 현실적인지
    판단할 수 있도록 같은 영역의 실측 월평균 지출을 나란히 놓는다."""
    rows = []
    for _, b in benefits.iterrows():
        limit = None if pd.isna(b["월한도_하나머니"]) else float(b["월한도_하나머니"])
        mapping = str(b["업종매핑(원더카드2.0재사용)"]).strip()
        big_spend = None
        if mapping != UNMAPPABLE:
            big_spend = meta["by_industry"].get(mapping)
        else:
            big_spend = meta["amt_total"] / meta["n_cards_total"]
        mid_spend, mid_label = mid_proxy_spend(b["소비데이터_중분류대리(신규가정)"], meta)

        cases = [("기본", parse_pct(b["기본적립률"]))]
        r_streak = parse_pct(b["연속3일적립률"])
        if r_streak is not None:
            cases.append(("3일연속", r_streak))
        for case, r in cases:
            if r is None:
                continue
            hit = None if limit is None else limit / r
            rows.append(
                {
                    "영역코드": b["영역코드"],
                    "영역명": b["영역명"],
                    "한도성격": b["한도성격"],
                    "월한도L(하나머니=원)": limit,
                    "적립률구분": case,
                    "r(적립률)": r,
                    "한도도달_월지출_L/r(원)": hit,
                    "비교_업종대분류_카드1장당월평균지출(원)": big_spend,
                    "비교_업종대분류(사용자가정·재사용)": mapping,
                    "한도도달지출÷업종대분류월평균": (hit / big_spend) if (hit and big_spend) else None,
                    "비교_중분류대리_카드1장당월평균지출(원)": mid_spend,
                    "비교_중분류대리(신규가정)": mid_label,
                    "한도도달지출÷중분류대리월평균": (hit / mid_spend) if (hit and mid_spend) else None,
                    "중분류대리_과소계상의심": (
                        bool(mid_spend / big_spend < MID_PROXY_MIN_SHARE)
                        if (mid_spend and big_spend and mid_label != "전체 업종 합") else None
                    ),
                    "비고": "" if limit is not None else "적립한도 미확인(상품페이지 미명시) — 계산 불가",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 계산 3
def build_breakeven(benefits: pd.DataFrame, fee_rates: dict, f_weighted: float, threshold: float) -> pd.DataFrame:
    """한도 L이 걸린 상태의 손익분기 필요 소비액 S = (1+t)/t * L/f.
    06_product_wondercard.min_spend_for_threshold()를 그대로 호출한다(재구현 금지).
    r이 식에서 소거되는지 확인하기 위해 적립률별로 각각 계산해 나란히 둔다."""
    s_col = f"손익분기 필요 월소비액 S(원) @threshold={threshold:.0%}"
    rows = []
    for _, b in benefits.iterrows():
        limit = None if pd.isna(b["월한도_하나머니"]) else float(b["월한도_하나머니"])
        mapping = str(b["업종매핑(원더카드2.0재사용)"]).strip()
        f, f_basis = resolve_fee(mapping, fee_rates, f_weighted)
        cases = [("기본", parse_pct(b["기본적립률"]))]
        r_streak = parse_pct(b["연속3일적립률"])
        if r_streak is not None:
            cases.append(("3일연속", r_streak))
        for case, r in cases:
            if r is None:
                continue
            row = {
                "영역코드": b["영역코드"],
                "영역명": b["영역명"],
                "한도성격": b["한도성격"],
                "월한도L(원)": limit,
                "적립률구분": case,
                "r(적립률)": r,
                "f(수수료율)": f,
                "f 산정근거": f_basis,
                "한도구속_r>=f/3(=t·f/(1+t))": None,
                s_col: None,
                "S가 r에 의존하는가": None,
                "비고": "",
            }
            if limit is None:
                row["비고"] = "적립한도 미확인(상품페이지 미명시) — L 없이는 S 계산 불가, 빈칸"
            else:
                r_allowed = threshold * f / (1 + threshold)
                binds = r > r_allowed
                row["한도구속_r>=f/3(=t·f/(1+t))"] = bool(binds)
                row[s_col] = min_spend_for_threshold(f, r, limit, threshold)
                row["S가 r에 의존하는가"] = "아니오(r 소거)" if binds else "예(S=0, 한도 무관하게 성립)"
                if not binds:
                    row["비고"] = f"명목 r이 이미 허용선({r_allowed:.3%}) 이하 → 한도와 무관하게 성립"
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 계산 4
def build_shared_limit(benefits: pd.DataFrame, fee_rates: dict, threshold: float) -> pd.DataFrame:
    """통합한도(C~F 합산 5만) vs 영역별 개별한도(각 5만)의 카드사 최대 부담 비교.
    그리고 통합한도 하에서 배분 비중을 흔들어 손익분기 소비액 S_total이 하나로
    정해지는지 확인한다. S_total = (1+t)/t * L / f_blended,
    f_blended = Σ f_i·S_i / ΣS_i 이므로 f_i가 영역마다 다르면 S_total은 배분에 의존한다."""
    grp = benefits[benefits["한도공유그룹"] == "CDEF"].copy()
    L = float(grp["월한도_하나머니"].iloc[0])
    areas = []
    for _, b in grp.iterrows():
        mapping = str(b["업종매핑(원더카드2.0재사용)"]).strip()
        areas.append(
            {
                "영역코드": b["영역코드"],
                "영역명": b["영역명"],
                "r": parse_pct(b["기본적립률"]),
                "업종매핑": mapping,
                "f": fee_rates[mapping],
            }
        )

    rows = []
    # 4-a. 카드사 최대 부담
    rows.append(
        {
            "항목": "1. 카드사 월 최대 부담",
            "시나리오": f"통합한도(C~F 합산 L={L:,.0f}원) — 공시 구조",
            "배분": "배분 무관",
            "카드사_월최대부담(원)": L,
            "f_blended(혼합수수료율)": None,
            "손익분기 필요 월소비액 S_total(원)": None,
            "비고": "네 영역 적립 합계가 L을 넘을 수 없으므로 배분과 무관하게 상한이 L로 고정",
        }
    )
    rows.append(
        {
            "항목": "1. 카드사 월 최대 부담",
            "시나리오": f"가정 대조: 영역별 개별한도(각 {L:,.0f}원) — 실제 공시 아님",
            "배분": "네 영역 모두 한도 소진 시",
            "카드사_월최대부담(원)": L * len(areas),
            "f_blended(혼합수수료율)": None,
            "손익분기 필요 월소비액 S_total(원)": None,
            "비고": f"영역 {len(areas)}개 × {L:,.0f}원 = 통합한도 대비 {len(areas)}배",
        }
    )

    # 4-b. 코너 케이스: 한 영역에 전액 몰았을 때의 S_total
    for a in areas:
        s = min_spend_for_threshold(a["f"], a["r"], L, threshold)
        rows.append(
            {
                "항목": "2. 배분 코너케이스(한 영역에 전액)",
                "시나리오": f"{a['영역코드']}. {a['영역명']} 100% (f={a['f']:.2%}, r={a['r']:.1%})",
                "배분": f"{a['영역코드']}=100%",
                "카드사_월최대부담(원)": L,
                "f_blended(혼합수수료율)": a["f"],
                "손익분기 필요 월소비액 S_total(원)": s,
                "비고": "S=(1+t)/t·L/f — 부담은 같아도 f가 다르면 S가 달라진다",
            }
        )

    # 4-c. 배분 스윕: 소매/유통(C) 대 나머지(D·E·F, f=0.40%로 동일)
    f_c = next(a["f"] for a in areas if a["영역코드"] == "C")
    f_rest = sorted({a["f"] for a in areas if a["영역코드"] != "C"})
    if len(f_rest) != 1:
        raise SystemExit(f"[중단] D·E·F의 f가 단일하지 않다: {f_rest}. 스윕 가정을 다시 볼 것.")
    f_rest = f_rest[0]
    for w in ALLOC_GRID:
        f_blend = w * f_c + (1 - w) * f_rest
        rows.append(
            {
                "항목": "3. 배분 스윕(지출 비중)",
                "시나리오": f"C(장보기·f={f_c:.2%}) 비중 {w:.0%} / D·E·F(f={f_rest:.2%}) {1 - w:.0%}",
                "배분": f"C={w:.0%}",
                "카드사_월최대부담(원)": L,
                "f_blended(혼합수수료율)": f_blend,
                "손익분기 필요 월소비액 S_total(원)": (1 + threshold) / threshold * L / f_blend,
                "비고": "부담은 L로 고정, S_total은 f_blended를 통해 배분에 의존",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 계산 5
def build_streak_sweep(benefits: pd.DataFrame, fee_rates: dict, f_weighted: float, threshold: float) -> pd.DataFrame:
    """3일 연속 조건을 고정 가정치로 두지 않고, 조건 충족일에 발생하는 결제액 비중 q를
    0.0~1.0으로 흔든다.
      실효 적립률  r_eff(q) = r_base(1-q) + r_streak·q      (r_streak = 2·r_base → r_base(1+q))
      g*(q)      = r_eff/(f - r_eff)                        ← 무혜택 대비
      g_incr(q)  = (r_eff - r_base)/(f - r_eff)             ← 기본적립률만 주는 상품 대비
    g_incr가 "적립률을 두 배로 주는 대신 이용액이 몇 % 늘어야 본전인가"에 해당한다."""
    rows = []
    for _, b in benefits.iterrows():
        r_base = parse_pct(b["기본적립률"])
        r_streak = parse_pct(b["연속3일적립률"])
        if r_streak is None:
            continue
        mapping = str(b["업종매핑(원더카드2.0재사용)"]).strip()
        f, _ = resolve_fee(mapping, fee_rates, f_weighted)
        limit = None if pd.isna(b["월한도_하나머니"]) else float(b["월한도_하나머니"])
        for q in Q_GRID:
            r_eff = r_base * (1 - q) + r_streak * q
            g = g_star(f, r_eff)
            g_base = g_star(f, r_base)
            if f <= r_eff:
                g_incr = np.inf
            else:
                g_incr = (r_eff - r_base) / (f - r_eff)
            rows.append(
                {
                    "영역코드": b["영역코드"],
                    "영역명": b["영역명"],
                    "q(조건충족일 결제액 비중)": q,
                    "r_base": r_base,
                    "r_streak": r_streak,
                    "r_eff(실효적립률)": r_eff,
                    "f(수수료율)": f,
                    "g*(무혜택 대비)": g,
                    "g*_손익성립(f>r_eff)": bool(np.isfinite(g)),
                    f"g*_threshold({threshold:.0%})충족": bool(np.isfinite(g) and g <= threshold),
                    "g_기준선(q=0, 기본적립률만)": g_base,
                    "g_incr(기본적립률 상품 대비 필요 증분이용률)": g_incr,
                    f"g_incr_threshold({threshold:.0%})이하": bool(np.isfinite(g_incr) and g_incr <= threshold),
                    "한도도달_월지출_L/r_eff(원)": (limit / r_eff) if limit else None,
                }
            )
    return pd.DataFrame(rows)


def crossing_q(f: float, r_base: float, threshold: float) -> dict:
    """q에 대한 임계점을 해석적으로 푼다 (스윕 격자 0.1 사이 값을 읽기 위함).
      r_eff(q) = r_base(1+q)
      손익 붕괴(f<=r_eff):            q >= f/r_base - 1
      g* > threshold:                 q >  f/((1+1/t)·r_base) - 1   (t=threshold)
      g_incr > threshold:             q >  t(f - r_base)/(r_base(1+t))
    """
    t = threshold
    return {
        "q_손익붕괴(g*=inf)": f / r_base - 1,
        "q_g*가 threshold 초과": (t * f / (1 + t)) / r_base - 1,
        "q_g_incr가 threshold 초과": t * (f - r_base) / (r_base * (1 + t)),
    }


# ---------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    fee_rates = cfg["merchant_fee_rate"]
    threshold = cfg["g_star_threshold"]
    benefits = load_benefits()

    f_weighted, f_basis = weighted_fee_for_allmerchant(fee_rates)
    if f_weighted is None:
        raise SystemExit(f"[중단] 전가맹점 f 산정 불가: {f_basis}")

    print("=" * 100)
    print("하나카드 신용카드 (CD_PD_SEQ=19149) — 손익분기 역산")
    print("=" * 100)
    print(f"[로드] {BENEFITS_PATH} — 영역 {len(benefits)}개 (사용자가 공식 상품페이지에서 전사)")
    print(f"[로드] {CONFIG_PATH} — g_star_threshold={threshold:.0%}, merchant_fee_rate {len(fee_rates)}개 업종 (재정의 없음)")
    print(f"[재사용] src/04_economics.py g_star(f,r)=r/(f-r)")
    print(f"[재사용] src/06_product_wondercard.py min_spend_for_threshold(), weighted_fee_for_allmerchant()")
    print(f"[산정] '전가맹점'(A·B) f = {f_weighted:.4%} — {f_basis}")
    print("\n[확인된 값 vs 가정한 값]")
    print("  확인: 영역별 적립률/월한도/한도공유구조/실적조건 (상품 공시), 매출구간별 수수료율 (BC카드 공시)")
    print("  가정: 영역→업종 매핑(원더카드 2.0에서 쓴 사용자 가정 재사용), threshold=50%(사용자 확정),")
    print(f"        카드 1장당 월 이용건수={cfg['avg_monthly_transactions_per_card']}건(config), 1 하나머니=1원(사용자 지정)")
    print("  미확인: A(국내외 가맹점) 적립 한도 — 상품페이지 미명시. 계산3 빈칸 처리.")

    OUT_NOMINAL.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- 계산 1
    nominal = build_nominal(benefits, fee_rates, f_weighted, threshold)
    nominal.to_csv(OUT_NOMINAL, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("계산1. 한도 없는 순수 적립률 기준 g* = r/(f-r)")
    print(f"(근거 스크립트: src/07_product_hanacard.py / 입력: {BENEFITS_PATH}, {CONFIG_PATH}, {SEGMENT_INDUSTRY_PATH})")
    print("=" * 100)
    show = nominal.copy()
    show["r"] = show["r(적립률)"].map("{:.1%}".format)
    show["f"] = show["f(수수료율)"].map("{:.4%}".format)
    show["g*"] = show["g*(한도무시·순수적립률)"].map(lambda v: "성립불가(inf)" if not np.isfinite(v) else f"{v:.1%}")
    print(
        show[["영역코드", "영역명", "적립률구분", "r", "업종매핑(사용자가정·재사용)", "f", "g*",
              f"threshold({threshold:.0%})_충족"]].to_string(index=False)
    )
    inf_rows = nominal[nominal["손익분기_미성립(f<=r)"]]
    print(f"\n  ● f <= r 이라 손익분기 자체가 성립하지 않는 조합: {len(inf_rows)}/{len(nominal)}건")
    for _, r in inf_rows.iterrows():
        print(f"      - {r['영역코드']}. {r['영역명']}({r['적립률구분']}) r={r['r(적립률)']:.1%} > f={r['f(수수료율)']:.4%}")
    fin = nominal[~nominal["손익분기_미성립(f<=r)"]]
    print(f"  ● g*가 유한한 조합: {len(fin)}건")
    for _, r in fin.iterrows():
        print(
            f"      - {r['영역코드']}. {r['영역명']}({r['적립률구분']}) g*={r['g*(한도무시·순수적립률)']:.1%} "
            f"→ threshold {threshold:.0%} {'충족' if r[f'threshold({threshold:.0%})_충족'] else '초과'}"
        )
    print(f"[완료] {OUT_NOMINAL} ({len(nominal)} rows)")

    # ---------------- 계산 2
    spend, meta = build_spend_reference(cfg)
    by_ind = (
        spend[spend["구분"] == "업종대분류(실측)"]
        .set_index("업종")["카드1장당_월평균지출(원)"]
        .to_dict()
    )
    meta["by_industry"] = by_ind
    spend.to_csv(OUT_SPEND, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("계산2. 한도에 도달하는 월 지출액 (= L/r) vs 실측 월평균 지출")
    print(f"(근거 스크립트: src/07_product_hanacard.py / 입력: {BENEFITS_PATH}, {SEGMENT_INDUSTRY_PATH}, {CLEANED_PATH})")
    print("=" * 100)
    print(
        f"  [월평균 지출 산정] 카드수 N = 월추정이용건수 합 ÷ {meta['tx_per_card']}건 "
        f"= {meta['n_cards_total']:,.0f}장 (config 가정치 avg_monthly_transactions_per_card)"
    )
    print(f"  전체 월추정매출액 {meta['amt_total']:,.0f}원 → 카드 1장당 월평균 지출 {meta['amt_total'] / meta['n_cards_total']:,.0f}원")

    limit_hit = build_limit_hit(benefits, meta)
    limit_hit.to_csv(OUT_LIMIT_HIT, index=False, encoding="utf-8-sig")
    lh = limit_hit.copy()
    lh["L"] = lh["월한도L(하나머니=원)"].map(lambda v: "미확인" if pd.isna(v) else f"{v:,.0f}")
    lh["r"] = lh["r(적립률)"].map("{:.1%}".format)
    lh["한도도달 월지출"] = lh["한도도달_월지출_L/r(원)"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}원")
    lh["대분류 월평균"] = lh["비교_업종대분류_카드1장당월평균지출(원)"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}원")
    lh["배수(대분류)"] = lh["한도도달지출÷업종대분류월평균"].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}배")
    lh["중분류대리 월평균"] = lh["비교_중분류대리_카드1장당월평균지출(원)"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}원")
    lh["배수(중분류)"] = lh["한도도달지출÷중분류대리월평균"].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}배")
    print()
    print(
        lh[["영역코드", "영역명", "한도성격", "L", "적립률구분", "r", "한도도달 월지출",
            "비교_업종대분류(사용자가정·재사용)", "대분류 월평균", "배수(대분류)",
            "비교_중분류대리(신규가정)", "중분류대리 월평균", "배수(중분류)"]].to_string(index=False)
    )
    print("\n  ● 빈칸 사유:")
    for _, r in limit_hit[limit_hit["비고"] != ""].iterrows():
        print(f"      - {r['영역코드']}. {r['영역명']}({r['적립률구분']}): {r['비고']}")
    print("      - B. 온라인 간편결제: 원천 데이터에 결제채널 구분 컬럼이 없어 중분류 대리지표 지정 불가")
    bad = limit_hit[limit_hit["중분류대리_과소계상의심"] == True]  # noqa: E712
    if len(bad):
        print(f"\n  ● 중분류 대리지표 과소계상 의심 (같은 대분류의 {MID_PROXY_MIN_SHARE:.0%} 미만) — 배수(중분류) 열을 결과로 읽지 말 것:")
        for _, r in bad.drop_duplicates(subset=["영역코드"]).iterrows():
            print(
                f"      - {r['영역코드']}. {r['영역명']}: 대리업종 '{r['비교_중분류대리(신규가정)']}' "
                f"월평균 {r['비교_중분류대리_카드1장당월평균지출(원)']:,.0f}원 = 대분류"
                f"({r['비교_업종대분류(사용자가정·재사용)']}, {r['비교_업종대분류_카드1장당월평균지출(원)']:,.0f}원)의 "
                f"{r['비교_중분류대리_카드1장당월평균지출(원)'] / r['비교_업종대분류_카드1장당월평균지출(원)']:.3%}"
            )
        print("        → 이 영역들의 현실성 판단은 '배수(대분류)' 열만 쓸 것 (대분류는 상한, 중분류 대리는 바닥값)")
    print("  ● C~F는 통합한도 5만이므로 위 'L/r'은 그 영역 하나로 통합한도를 전부 소진할 때의 값이다.")
    print(f"[완료] {OUT_LIMIT_HIT} ({len(limit_hit)} rows), {OUT_SPEND} ({len(spend)} rows)")

    # ---------------- 계산 3
    breakeven = build_breakeven(benefits, fee_rates, f_weighted, threshold)
    breakeven.to_csv(OUT_BREAKEVEN, index=False, encoding="utf-8-sig")
    s_col = f"손익분기 필요 월소비액 S(원) @threshold={threshold:.0%}"
    print("\n" + "=" * 100)
    print(f"계산3. 한도 L 하의 손익분기 필요 월소비액 S = (1+t)/t × L/f  (t={threshold:.0%} → S = {(1 + threshold) / threshold:.0f}L/f)")
    print(f"(근거 스크립트: src/07_product_hanacard.py, 수식 함수 src/06_product_wondercard.py:min_spend_for_threshold / 입력: {BENEFITS_PATH}, {CONFIG_PATH})")
    print("=" * 100)
    be = breakeven.copy()
    be["L"] = be["월한도L(원)"].map(lambda v: "미확인" if pd.isna(v) else f"{v:,.0f}")
    be["r"] = be["r(적립률)"].map("{:.1%}".format)
    be["f"] = be["f(수수료율)"].map("{:.4%}".format)
    be["S"] = be[s_col].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}원")
    print(
        be[["영역코드", "영역명", "한도성격", "L", "적립률구분", "r", "f", "S",
            "S가 r에 의존하는가", "비고"]].to_string(index=False)
    )

    print("\n  ● r 소거 확인 (L과 f가 같고 적립률 r만 다른 조합끼리 S를 비교):")
    checked = 0
    for (limit, f), grp in breakeven.dropna(subset=[s_col]).groupby(["월한도L(원)", "f(수수료율)"], sort=False):
        if grp["r(적립률)"].nunique() < 2:
            continue
        checked += 1
        same = grp[s_col].round(6).nunique() == 1
        detail = " / ".join(
            f"{r['영역코드']}({r['적립률구분']}) r={r['r(적립률)']:.1%} → S={r[s_col]:,.0f}원"
            for _, r in grp.iterrows()
        )
        print(f"      - L={limit:,.0f}원, f={f:.4%}: {detail}")
        print(f"        → {'동일(r 소거됨)' if same else '다름 — 수식 확인 필요'}")
    if checked == 0:
        raise SystemExit("[중단] r 소거를 확인할 수 있는 (L, f) 동일·r 상이 조합이 없다.")
    print(
        f"      한도가 구속되는 구간에서는 S = (1+t)/t·L/f 에 r이 없다. 단 구속 조건 자체가 "
        f"r > t·f/(1+t) (= f/3, t=50%)이고, 이는 한도 소진 시작점 L/r ≤ S 와 동치다 —\n"
        f"      즉 한도가 걸리는 한 r은 정확히 소거되고, r이 그 선 아래면 한도와 무관하게 S=0으로 성립한다."
    )
    print(f"[완료] {OUT_BREAKEVEN} ({len(breakeven)} rows)")

    # ---------------- 계산 4
    shared = build_shared_limit(benefits, fee_rates, threshold)
    shared.to_csv(OUT_SHARED, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("계산4. 통합한도(C~F 합산 5만) vs 영역별 개별한도")
    print(f"(근거 스크립트: src/07_product_hanacard.py / 입력: {BENEFITS_PATH}, {CONFIG_PATH})")
    print("=" * 100)
    sh = shared.copy()
    sh["최대부담"] = sh["카드사_월최대부담(원)"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}원")
    sh["f_blended"] = sh["f_blended(혼합수수료율)"].map(lambda v: "" if pd.isna(v) else f"{v:.4%}")
    sh["S_total"] = sh["손익분기 필요 월소비액 S_total(원)"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}원")
    print(sh[["항목", "시나리오", "최대부담", "f_blended", "S_total", "비고"]].to_string(index=False))
    sweep = shared[shared["항목"] == "3. 배분 스윕(지출 비중)"]
    s_vals = sweep["손익분기 필요 월소비액 S_total(원)"]
    print(
        f"\n  ● 카드사 최대 부담: 배분과 무관하게 50,000원 고정 (개별한도였다면 최대 200,000원 = 4배)"
    )
    print(
        f"  ● 손익분기 소비액 S_total: 배분에 따라 {s_vals.min():,.0f}원 ~ {s_vals.max():,.0f}원 "
        f"({s_vals.max() / s_vals.min():.2f}배 차이) — 하나로 정해지지 않는다"
    )
    print(f"[완료] {OUT_SHARED} ({len(shared)} rows)")

    # ---------------- 계산 5
    streak = build_streak_sweep(benefits, fee_rates, f_weighted, threshold)
    streak.to_csv(OUT_STREAK, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 100)
    print("계산5. 3일 연속 조건 — 조건 충족일 결제액 비중 q 민감도 (q=0.0~1.0, 0.1 단위)")
    print(f"(근거 스크립트: src/07_product_hanacard.py / 입력: {BENEFITS_PATH}, {CONFIG_PATH}, {SEGMENT_INDUSTRY_PATH})")
    print("=" * 100)
    for (code, name), grp in streak.groupby(["영역코드", "영역명"], sort=False):
        f = grp["f(수수료율)"].iloc[0]
        r_base = grp["r_base"].iloc[0]
        print(f"\n--- {code}. {name} (r_base={r_base:.1%} → r_streak={grp['r_streak'].iloc[0]:.1%}, f={f:.4%}) ---")
        t = grp.copy()
        t["q"] = t["q(조건충족일 결제액 비중)"].map("{:.1f}".format)
        t["r_eff"] = t["r_eff(실효적립률)"].map("{:.3%}".format)
        t["g*(무혜택대비)"] = t["g*(무혜택 대비)"].map(lambda v: "성립불가(inf)" if not np.isfinite(v) else f"{v:.1%}")
        t["g_incr(기본적립대비)"] = t["g_incr(기본적립률 상품 대비 필요 증분이용률)"].map(
            lambda v: "성립불가(inf)" if not np.isfinite(v) else f"{v:.1%}"
        )
        t["한도도달지출"] = t["한도도달_월지출_L/r_eff(원)"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}원")
        cols = ["q", "r_eff", "g*(무혜택대비)", f"g*_threshold({threshold:.0%})충족",
                "g_incr(기본적립대비)", f"g_incr_threshold({threshold:.0%})이하", "한도도달지출"]
        print(t[cols].to_string(index=False))
        cr = crossing_q(f, r_base, threshold)
        for label, q in cr.items():
            if q <= 0:
                print(f"    · {label}: q=0(=기본적립률만 줘도) 이미 해당 — 해석적 해 q={q:.4f}")
            elif q > 1:
                print(f"    · {label}: q<=1 범위 안에서는 발생하지 않음 (해석적 해 q={q:.4f} > 1)")
            else:
                print(f"    · {label}: q = {q:.4f} ({q:.1%})")
    print(f"\n[완료] {OUT_STREAK} ({len(streak)} rows)")


if __name__ == "__main__":
    main()
