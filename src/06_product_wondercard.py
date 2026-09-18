# 입력: data/reference/wondercard2_benefits.csv (원더카드 2.0 혜택표 — 사용자가 상품
#       설명서를 보고 직접 전사. 56개 영역 x 상품유형 3종(범용형/특화형/혼합형)의
#       최대 혜택률, 전월실적 40/80/120만원 구간별 월 한도, 업종 매핑(사용자 가정) 포함)
#       config/assumptions.yaml (merchant_fee_rate, g_star_threshold — 재정의하지 않고 로드)
#       outputs/tables/segment_industry_amt.csv (전가맹점 영역의 f 대안 산정용)
# 출력: outputs/tables/wondercard_nominal_gstar.csv   (계산1: 명목 혜택률 기준 g*)
#       outputs/tables/wondercard_breakeven.csv       (계산2: 월 한도 반영 — 최종 산출물)
#       outputs/tables/wondercard_limit_families.csv  (계산3: 한도 계열 분류)
#       outputs/tables/wondercard_sensitivity.csv     (계산4: 업종매핑·threshold 민감도)
# 목적: H10(실제 시장 카드 상품 1개와 대조)을 상품 "전체"로 확장한다. 영역 56개 x
#       상품유형 3종 전부를 본 프로젝트의 손익분기 모델에 투입해, 즉시할인 단독 모델이
#       실제 상품 구조를 어디까지 설명하고 어디서부터 설명하지 못하는지 확인한다 (H11~H13).
#
# 수식은 src/04_economics.py의 g_star()를 그대로 import해서 쓴다 (재정의 금지).
#   g* = r / (f - r),  f <= r 이면 inf(성립 불가)
#
# 월 한도 L이 걸리면 지출 S에서의 실효 혜택률은 min(r*S, L)/S 이다.
#   S <  L/r  : 실효 = r        (한도 미도달)
#   S >= L/r  : 실효 = L/S      (한도 소진, S가 커질수록 희석)
# g*(f, L/S) <= threshold 를 풀면 threshold를 충족하는 최소 지출액은
#   S_min = (1 + threshold) / threshold * L / f
# (threshold=0.50이면 S_min = 3L/f. 단 명목 r 자체가 이미 threshold를 만족하는
#  경우 — r <= threshold*f/(1+threshold) — 는 한도와 무관하게 성립하므로 S_min=0.)
#
# 주의(CLAUDE.md 규칙 1·3):
#   - CSV에 없는 혜택률/한도/실적조건은 추측해서 채우지 않는다. 빈칸은 빈칸으로 두고
#     `비고`에 사유를 적은 뒤 결측 목록을 콘솔에 보고한다.
#   - 영역->업종 매핑은 사용자가 임의로 붙인 가정치다. 모든 출력 표의 컬럼명에
#     `(사용자가정)`을 명시하고, 계산4에서 흔들어 결론이 뒤집히는지 확인한다.

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BENEFITS_PATH = Path("data/reference/wondercard2_benefits.csv")
CONFIG_PATH = Path("config/assumptions.yaml")
SEGMENT_INDUSTRY_PATH = Path("outputs/tables/segment_industry_amt.csv")
ECONOMICS_PATH = Path("src/04_economics.py")

OUT_NOMINAL = Path("outputs/tables/wondercard_nominal_gstar.csv")
OUT_BREAKEVEN = Path("outputs/tables/wondercard_breakeven.csv")
OUT_FAMILIES = Path("outputs/tables/wondercard_limit_families.csv")
OUT_SENSITIVITY = Path("outputs/tables/wondercard_sensitivity.csv")

PRODUCT_TYPES = {
    "범용형": "혜택률_범용형최대",
    "특화형": "혜택률_특화형최대",
    "혼합형": "혜택률_혼합형최대",
}
# 전월실적 구간 -> (구간 하한 금액, 월 한도 컬럼명)
PERF_TIERS = {
    "실적40만": (400_000, "월한도_실적40만"),
    "실적80만": (800_000, "월한도_실적80만"),
    "실적120만": (1_200_000, "월한도_실적120만"),
}
# 업종 매핑이 단일 업종으로 불가능한 영역 표식 (CSV의 업종매핑 값)
UNMAPPABLE = "전가맹점"

THRESHOLD_SWEEP = [0.50, 0.70, 1.00]

# 계산4 민감도 대상 — 사용자가 지정한 영역군. (영역목록, 기존매핑업종, 대안매핑업종)
# 대안은 "일반(2.08%)" 구간을 쓰는 업종을 대리로 지정한 것(요율 자체가 같으므로 동치).
SENSITIVITY_GROUPS = {
    "주유·대중교통": (["주유/LPG충전", "대중교통"], "생활서비스", "미디어/통신"),
    "백화점3사·면세점": (
        ["신세계백화점", "롯데백화점", "현대백화점", "면세점"],
        "소매/유통",
        "미디어/통신",
    ),
}

# 검증 케이스 (사용자 지정) — 마트/특화형 10%/실적120만(L=12,000)/f=1.15% -> S=3,130,435원
VERIFY = {"영역명": "마트", "상품유형": "특화형", "실적구간": "실적120만", "expected_S": 3_130_435}


def load_g_star():
    """src/04_economics.py의 g_star()를 그대로 쓴다 (숫자 접두사 때문에 importlib 경유)."""
    spec = importlib.util.spec_from_file_location("economics_04", ECONOMICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.g_star


g_star = load_g_star()


def parse_pct(value):
    """'10.0%' -> 0.10. 빈칸이면 None (추측해서 채우지 않는다)."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(str(value).strip().rstrip("%")) / 100


def parse_limit(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def parse_universal_limit(value):
    """범용형_한도 컬럼('통합 5만원' / '제한없음')을 해석한다.
    CSV는 범용형에 대해 '영역별·실적구간별 한도'가 아니라 별도 한도 체계를 적고 있다.
    -> (한도금액 또는 None, 한도성격 문자열). 규격에 없는 표현이 오면 추측하지 않고 중단."""
    raw = str(value).strip()
    if raw == "제한없음":
        return None, "한도 없음(CSV: 제한없음)"
    m = re.fullmatch(r"통합\s*([\d,]+)만원", raw)
    if m:
        return float(m.group(1).replace(",", "")) * 10_000, "통합한도(전 영역 합산, CSV상 실적 무관)"
    raise SystemExit(f"[중단] 범용형_한도 값 '{raw}'을 해석할 수 없다. 추측해서 채우지 않는다.")


def resolve_limit(b: pd.Series, ptype: str, tier: str):
    """(월한도, 한도성격, 실적구간이 의미 있는가) 를 돌려준다.

    CSV 구조상 한도 체계가 상품유형별로 다르다 — 이 차이를 뭉개지 않는다(CLAUDE.md 규칙 1):
      - 특화형/혼합형: 월한도_실적40/80/120만 = 영역별·실적구간별 한도
      - 범용형: 범용형_한도 = '통합 5만원'(전 영역 합산, 실적 구간 구분 없음) 또는 '제한없음'
    따라서 범용형에는 실적구간별 한도가 CSV에 존재하지 않는다. 특화형 한도를 범용형에
    가져다 쓰는 것은 없는 값을 지어내는 것이므로 하지 않는다."""
    if ptype == "범용형":
        limit, kind = parse_universal_limit(b["범용형_한도"])
        return limit, kind, False
    _, limit_col = PERF_TIERS[tier]
    return parse_limit(b[limit_col]), "영역별·실적구간별(CSV 월한도_실적XX)", True


def min_spend_for_threshold(f: float, r: float, limit: float, threshold: float) -> float:
    """실효 혜택률 min(r*S,L)/S 기준으로 g* <= threshold를 만족하는 최소 지출액.
    명목 r 자체가 이미 threshold를 만족하면 한도와 무관하게 성립하므로 0."""
    r_allowed = threshold * f / (1 + threshold)
    if r <= r_allowed:
        return 0.0
    return (1 + threshold) / threshold * limit / f


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_benefits() -> pd.DataFrame:
    if not BENEFITS_PATH.exists():
        raise SystemExit(f"[중단] {BENEFITS_PATH} 없음.")
    return pd.read_csv(BENEFITS_PATH, encoding="utf-8-sig")


# ---------------------------------------------------------------- 계산 1
def build_nominal(benefits: pd.DataFrame, fee_rates: dict) -> pd.DataFrame:
    rows = []
    for _, b in benefits.iterrows():
        mapping = str(b["업종매핑(내가정)"]).strip()
        f = fee_rates.get(mapping)
        for ptype, col in PRODUCT_TYPES.items():
            r = parse_pct(b[col])
            row = {
                "영역명": b["영역명"],
                "업종매핑(사용자가정)": mapping,
                "f(수수료율,가정)": f,
                "상품유형": ptype,
                "r(명목최대혜택률)": r,
                "g*(명목)": None,
                "성립불가(r>=f)": None,
                "비고": "",
            }
            if r is None:
                row["비고"] = "CSV에 혜택률 값 없음 — 추정하지 않음"
            elif f is None:
                row["비고"] = f"업종매핑 불가({mapping}) — 단일 f 배정 불가, 계산4에서 별도 처리"
            else:
                g = g_star(f, r)
                row["g*(명목)"] = g
                row["성립불가(r>=f)"] = bool(not np.isfinite(g))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 계산 2
def build_breakeven(benefits: pd.DataFrame, fee_rates: dict, threshold: float) -> pd.DataFrame:
    """영역 x 상품유형 x 실적구간별로 한도를 반영한 손익분기 지출액을 계산한다.
    범용형은 CSV상 실적구간별 한도가 없으므로(통합한도 또는 제한없음) 실적구간 축 없이
    1행만 만들고, 실적 대비 배수는 비워둔다 — 없는 실적조건을 지어내지 않기 위함."""
    s_col = f"threshold({threshold:.0%})충족최소지출S(원)"
    rows = []
    for _, b in benefits.iterrows():
        mapping = str(b["업종매핑(내가정)"]).strip()
        f = fee_rates.get(mapping)
        for ptype, col in PRODUCT_TYPES.items():
            r = parse_pct(b[col])
            tiers = list(PERF_TIERS) if ptype != "범용형" else ["실적무관(CSV상 범용형은 실적별 한도 없음)"]
            for tier in tiers:
                if ptype == "범용형":
                    limit, limit_kind, tier_meaningful = resolve_limit(b, ptype, None)
                    floor_amt = None
                else:
                    limit, limit_kind, tier_meaningful = resolve_limit(b, ptype, tier)
                    floor_amt = PERF_TIERS[tier][0]
                row = {
                    "영역명": b["영역명"],
                    "업종매핑(사용자가정)": mapping,
                    "f(수수료율,가정)": f,
                    "상품유형": ptype,
                    "r(명목최대혜택률)": r,
                    "실적구간": tier,
                    "전월실적하한(원)": floor_amt,
                    "월한도L(원)": limit,
                    "한도성격(CSV근거)": limit_kind,
                    "한도소진시작지출_L/r(원)": None,
                    "실효혜택률=f가되는지출_L/f(원)": None,
                    s_col: None,
                    "실적대비배수_S/실적하한": None,
                    "배수>1(단독성립불가)": None,
                    "비고": "",
                }
                notes = []
                if r is None:
                    notes.append("CSV에 혜택률 값 없음")
                if f is None:
                    notes.append(f"업종매핑 불가({mapping})")
                if limit is None and "제한없음" not in limit_kind:
                    notes.append("CSV에 월한도 값 없음")

                if r is not None and f is not None and limit is None and "제한없음" in limit_kind:
                    # 한도가 없으면 실효 혜택률은 지출과 무관하게 r 그대로 -> 명목 g*가 그대로 결론
                    g = g_star(f, r)
                    row[s_col] = 0.0 if g <= threshold else np.inf
                    notes.append("한도 없음 — 실효혜택률=r 고정, 지출액으로 희석 불가(명목 g*가 곧 결론)")
                elif r is not None and f is not None and limit is not None:
                    s_min = min_spend_for_threshold(f, r, limit, threshold)
                    row["한도소진시작지출_L/r(원)"] = limit / r
                    row["실효혜택률=f가되는지출_L/f(원)"] = limit / f
                    row[s_col] = s_min
                    if tier_meaningful:
                        multiple = s_min / floor_amt
                        row["실적대비배수_S/실적하한"] = multiple
                        row["배수>1(단독성립불가)"] = bool(multiple > 1)
                    else:
                        notes.append(
                            "범용형 통합한도 — 이 S는 '이 영역 하나로 통합한도를 전부 소진한다'는 "
                            "경계 가정의 값. CSV에 범용형 전월실적 조건이 없어 배수는 비워둠"
                        )
                if notes:
                    row["비고"] = " / ".join(notes)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 계산 3
def build_limit_families(benefits: pd.DataFrame, breakeven: pd.DataFrame) -> pd.DataFrame:
    limit_cols = [c for _, (_, c) in PERF_TIERS.items()]
    fam = benefits.dropna(subset=limit_cols).copy()
    fam["한도계열"] = fam[limit_cols].astype(int).astype(str).agg("/".join, axis=1)

    tier_order = {t: i for i, t in enumerate(PERF_TIERS)}
    rows = []
    for family, grp in fam.groupby("한도계열", sort=True):
        areas = grp["영역명"].tolist()
        for tier, (floor_amt, limit_col) in PERF_TIERS.items():
            limit = float(grp[limit_col].iloc[0])
            sub = breakeven[
                breakeven["영역명"].isin(areas)
                & (breakeven["실적구간"] == tier)
                & (breakeven["상품유형"] == "특화형")
                & breakeven["실적대비배수_S/실적하한"].notna()
            ]
            mult = sub["실적대비배수_S/실적하한"]
            rows.append(
                {
                    "한도계열(40/80/120만)": family,
                    "실적구간": tier,
                    "_order": tier_order[tier],
                    "전월실적하한(원)": floor_amt,
                    "월한도L(원)": limit,
                    "한도/전월실적 비율": limit / floor_amt,
                    "소속영역수": len(areas),
                    "소속영역예시": ", ".join(areas[:4]) + ("..." if len(areas) > 4 else ""),
                    "특화형_실적대비배수_최소": mult.min() if len(sub) else None,
                    "특화형_실적대비배수_중앙값": mult.median() if len(sub) else None,
                    "특화형_실적대비배수_최대": mult.max() if len(sub) else None,
                }
            )
    out = pd.DataFrame(rows).sort_values(["한도계열(40/80/120만)", "_order"])
    return out.drop(columns="_order").reset_index(drop=True)


# ---------------------------------------------------------------- 계산 4
def summarize_area_group(benefits_sub: pd.DataFrame, f: float, threshold: float) -> dict:
    """주어진 f 하에서 영역군의 명목 성립불가 비율과 한도 반영 배수를 집계."""
    nominal_flags, multiples = [], []
    for _, b in benefits_sub.iterrows():
        for ptype, col in PRODUCT_TYPES.items():
            r = parse_pct(b[col])
            if r is None:
                continue
            nominal_flags.append(not np.isfinite(g_star(f, r)))
            if ptype == "범용형":
                continue  # CSV에 범용형 실적조건이 없어 실적 대비 배수를 만들 수 없음
            for tier, (floor_amt, _) in PERF_TIERS.items():
                limit, _, _ = resolve_limit(b, ptype, tier)
                if limit is None:
                    continue
                multiples.append(min_spend_for_threshold(f, r, limit, threshold) / floor_amt)
    if not nominal_flags:
        return {"inf_rate": None, "n_ok": None, "n_total": 0, "median_multiple": None}
    m = pd.Series(multiples, dtype=float)
    return {
        "inf_rate": float(np.mean(nominal_flags)),
        "n_ok": int((m <= 1).sum()) if len(m) else None,
        "n_total": len(m),
        "median_multiple": float(m.median()) if len(m) else None,
    }


def weighted_fee_for_allmerchant(fee_rates: dict):
    """전가맹점 영역(국내외 가맹점·간편결제)은 단일 업종 f를 배정할 수 없다.
    제안: 이 프로젝트의 실측 소비 구성비로 가중평균한 f를 쓴다.
    (근거: outputs/tables/segment_industry_amt.csv — 업종별 월추정매출액)"""
    if not SEGMENT_INDUSTRY_PATH.exists():
        return None, f"{SEGMENT_INDUSTRY_PATH} 없음 — 03_segment.py 먼저 실행 필요"
    seg = pd.read_csv(SEGMENT_INDUSTRY_PATH, encoding="utf-8-sig")
    amt_col = [c for c in seg.columns if "매출액" in c][0]
    tot = seg.groupby("card_tpbuz_nm_1")[amt_col].sum()
    tot = tot[tot.index.isin(fee_rates)]
    share = tot / tot.sum()
    f_w = float((share * pd.Series(fee_rates)[share.index]).sum())
    detail = ", ".join(f"{k} {v:.1%}" for k, v in share.sort_values(ascending=False).head(4).items())
    return f_w, f"실측 업종별 매출비중 가중평균 (비중 상위: {detail})"


def build_sensitivity(benefits: pd.DataFrame, fee_rates: dict, base_threshold: float) -> pd.DataFrame:
    rows = []

    # 4-a/4-b: 업종 매핑을 흔들었을 때 판정이 뒤집히는가
    for group, (areas, base_ind, alt_ind) in SENSITIVITY_GROUPS.items():
        sub = benefits[benefits["영역명"].isin(areas)]
        for label, f in [
            (f"기존매핑: {base_ind}({fee_rates[base_ind]:.2%})", fee_rates[base_ind]),
            (f"대안매핑: 일반구간({fee_rates[alt_ind]:.2%})", fee_rates[alt_ind]),
        ]:
            s = summarize_area_group(sub, f, base_threshold)
            rows.append(
                {
                    "항목": f"1. 업종매핑 민감도 — {group}",
                    "시나리오": label,
                    "대상영역": ", ".join(areas),
                    "f(수수료율)": f,
                    "명목g*_성립불가(r>=f)_비율": s["inf_rate"],
                    "한도반영_배수<=1_건수": s["n_ok"],
                    "전체건수": s["n_total"],
                    "실적대비배수_중앙값": s["median_multiple"],
                    "비고": "업종매핑은 사용자 가정치 — 이 행은 그 가정을 흔든 결과",
                }
            )

    # 4-c: 전가맹점 영역 별도 처리 제안 + 상·하한 대조
    allm = benefits[benefits["업종매핑(내가정)"] == UNMAPPABLE]
    f_w, basis = weighted_fee_for_allmerchant(fee_rates)
    scenarios = []
    if f_w is not None:
        scenarios.append((f"제안: 소비구성비 가중평균 f({f_w:.2%})", f_w, basis))
    scenarios += [
        (f"하한 대조: 최저요율 업종({min(fee_rates.values()):.2%})", min(fee_rates.values()),
         "가중평균 f가 어느 범위 안에 있는지 확인용"),
        (f"상한 대조: 최고요율 업종({max(fee_rates.values()):.2%})", max(fee_rates.values()),
         "가중평균 f가 어느 범위 안에 있는지 확인용"),
    ]
    for label, f, note in scenarios:
        s = summarize_area_group(allm, f, base_threshold)
        rows.append(
            {
                "항목": "2. 전가맹점 영역 처리 (단일 업종 매핑 불가)",
                "시나리오": label,
                "대상영역": ", ".join(allm["영역명"].tolist()),
                "f(수수료율)": f,
                "명목g*_성립불가(r>=f)_비율": s["inf_rate"],
                "한도반영_배수<=1_건수": s["n_ok"],
                "전체건수": s["n_total"],
                "실적대비배수_중앙값": s["median_multiple"],
                "비고": note,
            }
        )

    # 4-d: threshold 스윕
    nominal = build_nominal(benefits, fee_rates)
    nv = nominal[nominal["g*(명목)"].notna()]
    for th in THRESHOLD_SWEEP:
        be = build_breakeven(benefits, fee_rates, th)
        valid = be[be["실적대비배수_S/실적하한"].notna()]
        for ptype in list(PRODUCT_TYPES) + ["전체"]:
            v = valid if ptype == "전체" else valid[valid["상품유형"] == ptype]
            n = nv if ptype == "전체" else nv[nv["상품유형"] == ptype]
            rows.append(
                {
                    "항목": "3. threshold 민감도",
                    "시나리오": f"threshold={th:.0%} / {ptype}",
                    "대상영역": "전체 56개 영역 중 업종매핑 가능분",
                    "f(수수료율)": None,
                    "명목g*_성립불가(r>=f)_비율": float((n["g*(명목)"] == np.inf).mean()) if len(n) else None,
                    "한도반영_배수<=1_건수": int((v["실적대비배수_S/실적하한"] <= 1).sum()),
                    "전체건수": len(v),
                    "실적대비배수_중앙값": float(v["실적대비배수_S/실적하한"].median()),
                    "비고": "명목 성립불가 비율은 r>=f 여부만 보므로 threshold와 무관 — 안 변하는 게 정상",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 검증
def verify(breakeven: pd.DataFrame, threshold: float) -> None:
    col = f"threshold({threshold:.0%})충족최소지출S(원)"
    row = breakeven[
        (breakeven["영역명"] == VERIFY["영역명"])
        & (breakeven["상품유형"] == VERIFY["상품유형"])
        & (breakeven["실적구간"] == VERIFY["실적구간"])
    ]
    if len(row) != 1:
        raise SystemExit(f"[중단] 검증 케이스 행을 찾지 못함: {VERIFY}")
    got = round(float(row[col].iloc[0]))
    exp = VERIFY["expected_S"]
    if abs(got - exp) > 1:
        raise SystemExit(
            f"[중단] 검증 실패. 마트/특화형/실적120만 S = {got:,}원 (기대 {exp:,}원). 계산이 틀렸다."
        )
    print(f"[검증 통과] 마트/특화형 10%/실적120만(L=12,000, f=1.15%) S = {got:,}원 (기대 {exp:,}원)")


def report_missing(benefits: pd.DataFrame) -> None:
    """CSV 결측 보고 — 추측해서 채우지 않았음을 명시 (CLAUDE.md 규칙 1)."""
    print("\n=== CSV 결측값 보고 (추정하지 않고 빈칸으로 둠) ===")
    cols = list(PRODUCT_TYPES.values()) + [c for _, (_, c) in PERF_TIERS.items()]
    any_missing = False
    for col in cols:
        blank = benefits[col].isna() | (benefits[col].astype(str).str.strip() == "")
        miss = benefits[blank]
        if len(miss):
            any_missing = True
            print(f"  - {col}: {len(miss)}건 — {', '.join(miss['영역명'].tolist())}")
    if not any_missing:
        print("  - 혜택률/월한도 결측 없음")
    unmapped = benefits[benefits["업종매핑(내가정)"] == UNMAPPABLE]
    print(f"  - 업종매핑 불가('{UNMAPPABLE}'): {len(unmapped)}건 — {', '.join(unmapped['영역명'].tolist())}")
    print("  - 전월실적 조건은 CSV에 40/80/120만원 3구간만 있음. 그 사이 구간·상위 구간은 추정하지 않음.")


def main() -> None:
    cfg = load_config()
    fee_rates = cfg["merchant_fee_rate"]
    threshold = cfg["g_star_threshold"]
    benefits = load_benefits()
    print(f"[로드] {BENEFITS_PATH} — {len(benefits)}개 영역, 상품유형 {len(PRODUCT_TYPES)}종")
    print(
        f"[로드] {CONFIG_PATH} — g_star_threshold={threshold:.0%} (재정의 없음), "
        f"merchant_fee_rate {len(fee_rates)}개 업종"
    )
    report_missing(benefits)

    OUT_NOMINAL.parent.mkdir(parents=True, exist_ok=True)

    # ---- 계산 1
    nominal = build_nominal(benefits, fee_rates)
    nominal.to_csv(OUT_NOMINAL, index=False, encoding="utf-8-sig")
    valid_n = nominal[nominal["g*(명목)"].notna()]
    print(f"\n[완료] {OUT_NOMINAL} ({len(nominal)} rows)")
    print("=== 계산1. 명목 혜택률 기준 g* ===")
    print(f"  계산 가능 조합 {len(valid_n)}건 / 전체 {len(nominal)}건 (나머지는 혜택률 결측 또는 업종매핑 불가)")
    n_inf = int((valid_n["g*(명목)"] == np.inf).sum())
    print(f"  명목 r >= f 라 성립 자체가 불가(g*=inf): {n_inf}건 = {n_inf / len(valid_n):.1%}")
    for ptype in PRODUCT_TYPES:
        n_t = valid_n[valid_n["상품유형"] == ptype]
        k = int((n_t["g*(명목)"] == np.inf).sum())
        print(f"    - {ptype}: {k}/{len(n_t)}건 = {k / len(n_t):.1%}")
    finite = valid_n[np.isfinite(valid_n["g*(명목)"].astype(float))]
    print(f"  g*가 유한한 조합: {len(finite)}건")
    for _, r in finite.iterrows():
        print(
            f"    {r['영역명']}/{r['상품유형']} r={r['r(명목최대혜택률)']:.1%} "
            f"f={r['f(수수료율,가정)']:.2%} g*={r['g*(명목)']:.1%}"
        )

    # ---- 계산 2
    breakeven = build_breakeven(benefits, fee_rates, threshold)
    breakeven.to_csv(OUT_BREAKEVEN, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {OUT_BREAKEVEN} ({len(breakeven)} rows)")
    verify(breakeven, threshold)

    s_col = f"threshold({threshold:.0%})충족최소지출S(원)"
    valid_b = breakeven[breakeven["실적대비배수_S/실적하한"].notna()]
    over = valid_b["배수>1(단독성립불가)"].astype(bool)
    print("\n=== 계산2. 월 한도 반영 ===")
    print(f"  계산 가능 조합 {len(valid_b)}건 / 전체 {len(breakeven)}건")
    print(f"  실적대비배수 > 1 (그 영역 단독으로는 어떤 경우에도 성립 불가): {int(over.sum())}건 = {over.mean():.1%}")
    print(f"  배수 <= 1 (단독 성립 가능): {int((~over).sum())}건")
    if (~over).any():
        ok = valid_b[~over].sort_values("실적대비배수_S/실적하한")
        print("  성립 가능 조합:")
        for _, r in ok.iterrows():
            print(
                f"    {r['영역명']}/{r['상품유형']}/{r['실적구간']} r={r['r(명목최대혜택률)']:.1%} "
                f"f={r['f(수수료율,가정)']:.2%} L={int(r['월한도L(원)']):,}원 "
                f"S={r[s_col]:,.0f}원 배수={r['실적대비배수_S/실적하한']:.2f}"
            )
    print("\n  상품유형별 실적대비배수 (범용형은 CSV에 실적별 한도가 없어 배수 계산 대상 아님):")
    for ptype in PRODUCT_TYPES:
        v = valid_b[valid_b["상품유형"] == ptype]["실적대비배수_S/실적하한"]
        if not len(v):
            print(f"    {ptype}: 계산 대상 없음 (CSV상 통합한도/제한없음 — 실적구간별 한도 부재)")
            continue
        print(
            f"    {ptype}: 최소 {v.min():.2f}배 / 중앙값 {v.median():.2f}배 / 최대 {v.max():.2f}배, "
            f"배수<=1 {int((v <= 1).sum())}/{len(v)}건"
        )

    uni = breakeven[(breakeven["상품유형"] == "범용형") & breakeven[s_col].notna()]
    print("\n  범용형(통합한도 5만원 / 국내외 가맹점은 제한없음) — 별도 체계:")
    fin = uni[np.isfinite(uni[s_col].astype(float))]
    if len(fin):
        print(
            f"    통합한도를 한 영역에서 전부 소진한다고 볼 때 threshold 충족 지출액 S: "
            f"{fin[s_col].min():,.0f}원 ~ {fin[s_col].max():,.0f}원 "
            f"(f에만 의존: 3*50,000/f). 대상 {len(fin)}건"
        )
    nofin = uni[~np.isfinite(uni[s_col].astype(float))]
    if len(nofin):
        print(f"    한도 없음(제한없음)이라 지출로 희석 불가 -> 성립 불가: {len(nofin)}건")
    print("\n  실적구간별 실적대비배수(특화형):")
    for tier in PERF_TIERS:
        v = valid_b[(valid_b["실적구간"] == tier) & (valid_b["상품유형"] == "특화형")]["실적대비배수_S/실적하한"]
        print(f"    {tier}: 최소 {v.min():.2f}배 / 중앙값 {v.median():.2f}배 / 최대 {v.max():.2f}배")

    # ---- 계산 3
    families = build_limit_families(benefits, breakeven)
    families.to_csv(OUT_FAMILIES, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {OUT_FAMILIES} ({len(families)} rows)")
    print("=== 계산3. 한도 계열별 한도/전월실적 비율 ===")
    print(
        families[
            [
                "한도계열(40/80/120만)",
                "실적구간",
                "월한도L(원)",
                "한도/전월실적 비율",
                "소속영역수",
                "특화형_실적대비배수_중앙값",
            ]
        ].to_string(index=False)
    )

    # ---- 계산 4
    sens = build_sensitivity(benefits, fee_rates, threshold)
    sens.to_csv(OUT_SENSITIVITY, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {OUT_SENSITIVITY} ({len(sens)} rows)")
    print("=== 계산4. 민감도 ===")
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
