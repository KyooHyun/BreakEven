# 입력: data/processed/02_cleaned.parquet, outputs/tables/segment_industry_amt.csv,
#       outputs/tables/product_proposals.csv, outputs/tables/bep_heatmap.csv,
#       outputs/tables/segment_profile.csv, config/assumptions.yaml
# 출력: outputs/tables/sensitivity.csv
# 목적: L4 민감도와 반증. (PROJECT_SPEC.md L4)
#   1. 수수료율(f) 매핑을 한 단계 낙관/비관으로 흔들었을 때 "할인 성립 여부"가 뒤집히는 업종 탐색
#   2. 유입형 상권 제외 전후로 세그먼트 소비 구성이 얼마나 바뀌는지 비교
#   3. 클러스터 개수 k를 바꿨을 때 "소매/유통만 확실히 성립" 결론이 유지되는지 확인
#   4. 세그먼트별 탄력성 가정(H7)을 흔들었을 때 "탄력성 기준 성립" 건수가 어떻게 바뀌는지,
#      그리고 각 세그먼트 1위 업종이 탄력성 기준으로도 성립하려면 필요한 최소 탄력성은
#      얼마인지 역산 (H9) — H7의 "486개 중 6개만 성립"이 가정치의 과보수성 때문인지
#      확인하기 위함.
#   5. 실제 판매 중인 카드 상품(대형마트 10% 할인, 월 한도 1.5만원, 웹 검색으로 확인)의
#      명목 혜택률을 본 모델에 넣어 검증 (H10) — 실제 시장에 혜택 상품이 존재한다는
#      사실 자체로 이 프로젝트의 비관적 결론이 흔들리는지 확인.
#   6. 탄력성 기준 판정(H7·H9)도 업종->수수료구간 매핑 가정 위에 있는지 교차 확인
#      (docs/limitations.md 0번 표의 미검증 칸을 채우기 위함)
#   결론이 무너지는 조건은 숨기지 않고 그대로 기록한다.

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

CLEANED_PATH = Path("data/processed/02_cleaned.parquet")
SEGMENT_INDUSTRY_PATH = Path("outputs/tables/segment_industry_amt.csv")
BEP_HEATMAP_PATH = Path("outputs/tables/bep_heatmap.csv")
SEGMENT_PROFILE_PATH = Path("outputs/tables/segment_profile.csv")
CONFIG_PATH = Path("config/assumptions.yaml")
OUT_SENSITIVITY = Path("outputs/tables/sensitivity.csv")

FEE_TIERS = [0.0040, 0.0100, 0.0115, 0.0145, 0.0208]  # config/assumptions.yaml fee_rate_by_revenue_tier 순서
EXCLUDED_AGES = {1, 11}
REF_R = 0.003  # 최소 유의미 혜택률(0.3%) 기준으로 성립 여부 판정
ELASTICITY_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]  # 현재 가정치 대비 배수
# 6번 부분 매핑 시나리오 대상 — 1번에서 판정을 뒤집었던 영세구간(0.40%) 업종
# (src/08_woori_dual_network.py LOW_FEE_INDUSTRIES와 동일)
LOW_FEE_INDUSTRIES = ["음식", "생활서비스", "여가/오락"]

# H10: 실제 시장 카드 상품 대비 검증. 웹 검색(2026-08-27)으로 확인한 실제 판매 상품 —
# 롯데카드 LOCA CLASSIC: 전월실적 150만원 이상 시 대형마트 10% 할인, 월 한도 15,000원.
# 출처: https://namu.wiki/w/롯데카드/카드%20상품
# 명목 혜택률(10%)은 본 프로젝트의 소매/유통 가정 수수료율(1.15%)보다 훨씬 커서 그대로는
# 모델에 넣을 수 없다 — 월 한도가 "실효 혜택률"을 낮추는 역할을 한다는 가설(H10)을 검증.
REAL_PRODUCT_NOMINAL_R = 0.10
REAL_PRODUCT_MONTHLY_CAP = 15_000
REAL_PRODUCT_SPEND_LEVELS = [300_000, 500_000, 1_000_000, 1_500_000]  # 월 대형마트 지출 가정(예시)


def g_star(f: float, r: float) -> float:
    if f <= r:
        return np.inf
    return r / (f - r)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fee_sensitivity(cfg: dict) -> pd.DataFrame:
    """업종별 수수료율 가정을 한 단계 위/아래 구간으로 흔췄을 때 r=0.3% 기준 성립 여부 변화."""
    threshold = cfg["g_star_threshold"]
    rows = []
    for industry, f in cfg["merchant_fee_rate"].items():
        idx = FEE_TIERS.index(f)
        pessimistic = FEE_TIERS[max(idx - 1, 0)]
        optimistic = FEE_TIERS[min(idx + 1, len(FEE_TIERS) - 1)]
        for label, fv in [("비관(한 단계 아래)", pessimistic), ("기본(현재 가정)", f), ("낙관(한 단계 위)", optimistic)]:
            g = g_star(fv, REF_R)
            rows.append(
                {
                    "업종": industry,
                    "시나리오": label,
                    "f": fv,
                    "g*(r=0.3%)": g,
                    "성립(g*<=threshold)": g <= threshold,
                }
            )
    df = pd.DataFrame(rows)
    flips = (
        df.pivot(index="업종", columns="시나리오", values="성립(g*<=threshold)")
        .assign(결론_뒤집힘=lambda d: d.nunique(axis=1) > 1)
    )
    print("=== 1. 수수료율 매핑 민감도 (r=0.3% 기준 성립 여부) ===")
    print(flips.to_string())
    return df


def inflow_sensitivity() -> pd.DataFrame:
    """유입형 상권 포함/제외 시 연령대별 업종 소비 구성이 얼마나 달라지는지 비교."""
    df = pd.read_parquet(CLEANED_PATH)
    df = df[~df["age"].isin(EXCLUDED_AGES)]

    def industry_share(sub: pd.DataFrame) -> pd.Series:
        return sub.groupby("card_tpbuz_nm_1", observed=True)["amt"].sum().pipe(lambda s: s / s.sum())

    incl = industry_share(df) * 100
    excl = industry_share(df[~df["is_inflow_area"]]) * 100
    compare = pd.DataFrame({"포함(유입형 상권 O, %)": incl, "제외(유입형 상권 X, 본 분석 기준, %)": excl})
    compare["차이(%p)"] = compare["포함(유입형 상권 O, %)"] - compare["제외(유입형 상권 X, 본 분석 기준, %)"]
    compare = compare.sort_values("차이(%p)", key=abs, ascending=False)
    print("\n=== 2. 유입형 상권 포함/제외 — 업종별 매출 비중 차이 ===")
    print(compare.round(2).to_string())
    return compare


def k_sensitivity(cfg: dict) -> pd.DataFrame:
    """k=3(본 분석) vs k=4, k=5일 때 '소매/유통만 확실히 성립' 결론이 유지되는지 확인."""
    df = pd.read_parquet(CLEANED_PATH)
    df = df[(~df["is_inflow_area"]) & (~df["age"].isin(EXCLUDED_AGES))]

    group_cols = ["sex", "age_label"]
    vector_cols = ["card_tpbuz_nm_2", "hour_label", "day_label"]
    pivot = df.groupby(group_cols + vector_cols, observed=True)["amt"].sum().unstack(vector_cols, fill_value=0)
    vectors = pivot.div(pivot.sum(axis=1), axis=0)

    rows = []
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=cfg["random_seed"], n_init=10)
        labels = pd.Series(km.fit_predict(vectors), index=vectors.index, name="cluster")
        merged = df.merge(labels, left_on=group_cols, right_index=True)
        for cid, g in merged.groupby("cluster"):
            top1 = g.groupby("card_tpbuz_nm_1", observed=True)["amt"].sum().idxmax()
            members = sorted({f"{s}-{a}" for s, a in g[group_cols].drop_duplicates().itertuples(index=False)})
            rows.append({"k": k, "cluster": cid, "구성원수": len(members), "1위업종": top1})
    result = pd.DataFrame(rows)
    print("\n=== 3. k별 클러스터 1위 업종 (전부 소매/유통이면 결론 유지) ===")
    print(result.to_string(index=False))
    all_retail = (result["1위업종"] == "소매/유통").all()
    print(f"\n모든 k에서 모든 클러스터의 1위 업종이 소매/유통인가? -> {all_retail}")
    return result


def elasticity_sensitivity(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H9: 탄력성 가정치(H7)를 배수로 흔들었을 때 (a) 탄력성 기준 성립 건수가 어떻게
    바뀌는지, (b) 각 세그먼트 1위 업종이 성립하려면 필요한 최소 탄력성 배수는 얼마인지."""
    if not BEP_HEATMAP_PATH.exists():
        raise SystemExit(f"[중단] {BEP_HEATMAP_PATH} 없음. 먼저 04_economics.py 실행할 것.")
    heatmap = pd.read_csv(BEP_HEATMAP_PATH)
    threshold = cfg["g_star_threshold"]
    elasticity_cfg = cfg["segment_response_elasticity"]

    # (a) 배수 스윕 — 배수가 커질수록 탄력성 기준 성립 건수가 몇 개까지 늘어나는지
    finite = heatmap[np.isfinite(heatmap["g*(필요증분이용률)"])].copy()
    sweep_rows = []
    for mult in ELASTICITY_MULTIPLIERS:
        scaled_expected = finite["세그먼트예상반응g(탄력성가정)"] * mult
        elastic_ok = scaled_expected >= finite["g*(필요증분이용률)"]
        both_ok = elastic_ok & finite["threshold기준_성립"]
        sweep_rows.append(
            {
                "탄력성_배수": mult,
                "탄력성기준_성립_건수": int(elastic_ok.sum()),
                "두기준_모두_성립_건수": int(both_ok.sum()),
                "전체_조합수": len(finite),
            }
        )
    sweep_df = pd.DataFrame(sweep_rows)
    print("=== 4a. 탄력성 배수 스윕 — 배수가 커질수록 성립 건수가 몇 개까지 느는가 ===")
    print(sweep_df.to_string(index=False))

    # (b) 세그먼트 1위 업종이 탄력성 기준으로 성립하려면 필요한 최소 탄력성(r=r_min 기준)
    if not SEGMENT_PROFILE_PATH.exists():
        raise SystemExit(f"[중단] {SEGMENT_PROFILE_PATH} 없음. 먼저 03_segment.py 실행할 것.")
    profile = pd.read_csv(SEGMENT_PROFILE_PATH).set_index("segment")
    r_min = cfg["benefit_rate_sweep"]["min"]

    required_rows = []
    for seg in ["cluster0", "cluster1", "cluster2"]:
        top_industry = profile.loc[seg, "주력업종_top3"].split(" / ")[0].split("(")[0]
        row = heatmap[
            (heatmap["세그먼트"] == seg)
            & (heatmap["업종"] == top_industry)
            & np.isclose(heatmap["r(혜택률)"], r_min)
        ].iloc[0]
        g_req = row["g*(필요증분이용률)"]
        assumed_coef = elasticity_cfg[seg]
        # r=r_min에서 expected_g = coef이므로 필요 coef = g_req 그대로
        required_multiple = g_req / assumed_coef if np.isfinite(g_req) else np.inf
        required_rows.append(
            {
                "세그먼트": seg,
                "1위업종": top_industry,
                "r": r_min,
                "g*(필요증분이용률)": g_req,
                "가정_탄력성계수": assumed_coef,
                "성립에_필요한_탄력성계수": g_req,
                "필요배수(가정치대비)": required_multiple,
            }
        )
    required_df = pd.DataFrame(required_rows)
    print("\n=== 4b. 세그먼트 1위 업종이 탄력성 기준으로 성립하려면 필요한 배수 (r=0.3% 기준) ===")
    print(required_df.to_string(index=False))
    print(
        "\n실제 시장에 해당 업종 할인 카드 상품이 다수 존재한다면, 그 상품들이 전제하는 "
        "탄력성은 최소 위 '성립에 필요한 탄력성계수' 수준일 것이라고 역산할 수 있다 — "
        "본 프로젝트의 가정치(0.05~0.20)와 비교해 몇 배 차이나는지가 이 가정이 얼마나 "
        "보수적인지를 보여주는 지표다."
    )
    return sweep_df, required_df


def elasticity_mapping_cross(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """6. 탄력성 기준 판정(H7·H9)도 업종->수수료구간 매핑 가정 위에 있는가.

    docs/limitations.md 0번은 "서로 다른 결론들이 이 가정 하나에 공통으로 기대고 있다"를
    표로 정리하는데, 탄력성 항목만 '구조적으로 같은 가정 위에 있으나 미검증'으로 비어
    있었다. 비교 대상이 g*이고 g*는 f에서 나오므로 영향이 있는 건 분명하지만, 실제로
    판정이 뒤집히는지는 계산해봐야 안다.

    expected_g = coef * (r / r_min) 은 f와 무관하므로 매핑을 바꿔도 그대로다.
    g*만 새 매핑으로 다시 계산해 (a) 성립 건수, (b) 세그먼트 1위 업종 필요 배수를 비교한다.
    매핑 시나리오는 1번(fee_sensitivity)·08 계산5와 같은 규칙을 쓴다."""
    heatmap = pd.read_csv(BEP_HEATMAP_PATH)
    profile = pd.read_csv(SEGMENT_PROFILE_PATH).set_index("segment")
    fee_rates = cfg["merchant_fee_rate"]
    threshold = cfg["g_star_threshold"]
    elasticity_cfg = cfg["segment_response_elasticity"]
    r_min = cfg["benefit_rate_sweep"]["min"]

    scenarios = {}
    for label, shift in [("비관(한 단계 아래)", -1), ("기본(현재 가정)", 0), ("낙관(한 단계 위)", 1)]:
        scenarios[label] = {
            ind: FEE_TIERS[min(max(FEE_TIERS.index(f) + shift, 0), len(FEE_TIERS) - 1)]
            for ind, f in fee_rates.items()
        }
    partial = dict(fee_rates)
    for ind in LOW_FEE_INDUSTRIES:
        partial[ind] = FEE_TIERS[min(FEE_TIERS.index(fee_rates[ind]) + 1, len(FEE_TIERS) - 1)]
    scenarios["부분(저수수료 3개 업종만 한 단계 위)"] = partial

    count_rows, required_rows = [], []
    for label, mapping in scenarios.items():
        h = heatmap.copy()
        h["f_new"] = h["업종"].map(mapping)
        h["g*_new"] = [g_star(f, r) for f, r in zip(h["f_new"], h["r(혜택률)"])]
        finite = h[np.isfinite(h["g*_new"])]
        elastic_ok = finite["세그먼트예상반응g(탄력성가정)"] >= finite["g*_new"]
        threshold_ok = finite["g*_new"] <= threshold
        count_rows.append(
            {
                "매핑시나리오": label,
                "g*가 유한한 조합수": len(finite),
                "threshold기준 성립": int(threshold_ok.sum()),
                "탄력성기준 성립": int(elastic_ok.sum()),
                "두 기준 모두 성립": int((elastic_ok & threshold_ok).sum()),
            }
        )
        for seg in ["cluster0", "cluster1", "cluster2"]:
            top_industry = profile.loc[seg, "주력업종_top3"].split(" / ")[0].split("(")[0]
            row = h[
                (h["세그먼트"] == seg) & (h["업종"] == top_industry) & np.isclose(h["r(혜택률)"], r_min)
            ].iloc[0]
            g_req = row["g*_new"]
            coef = elasticity_cfg[seg]
            required_rows.append(
                {
                    "매핑시나리오": label,
                    "세그먼트": seg,
                    "1위업종": top_industry,
                    "f(시나리오)": row["f_new"],
                    "g*(r=0.3%)": g_req,
                    "가정_탄력성계수": coef,
                    "필요배수(가정치대비)": g_req / coef if np.isfinite(g_req) else np.inf,
                    "가정치_그대로_성립": bool(np.isfinite(g_req) and coef >= g_req),
                }
            )

    count_df = pd.DataFrame(count_rows)
    required_df = pd.DataFrame(required_rows)
    print("\n=== 6a. 매핑을 흔들었을 때 탄력성 기준 성립 건수 ===")
    print(count_df.to_string(index=False))
    print("\n=== 6b. 매핑을 흔들었을 때 세그먼트 1위 업종의 필요 배수 (r=0.3%) ===")
    print(required_df.round(4).to_string(index=False))
    flips = required_df.groupby("세그먼트")["가정치_그대로_성립"].nunique()
    print(
        "\n매핑 시나리오에 따라 '가정치 그대로 성립' 판정이 갈리는 세그먼트: "
        + (", ".join(flips[flips > 1].index) if (flips > 1).any() else "없음")
    )
    return count_df, required_df


def real_product_check(cfg: dict) -> pd.DataFrame:
    """H10: 실제 판매 중인 카드 상품(대형마트 10% 할인, 월 한도 15,000원)의 명목
    혜택률을 그대로 넣으면 모델과 안 맞는다(r=10% >> f=1.15%) — 월 한도가 지출액에
    따라 '실효 혜택률'을 얼마나 낮추는지, 그 실효 혜택률이 본 모델의 f보다 낮아지는
    지점(=g*가 유한해지는 지점)이 실제로 있는지를 확인한다."""
    f = cfg["merchant_fee_rate"]["소매/유통"]
    threshold = cfg["g_star_threshold"]

    rows = []
    for spend in REAL_PRODUCT_SPEND_LEVELS:
        capped = spend * REAL_PRODUCT_NOMINAL_R > REAL_PRODUCT_MONTHLY_CAP
        effective_r = REAL_PRODUCT_MONTHLY_CAP / spend if capped else REAL_PRODUCT_NOMINAL_R
        g = g_star(f, effective_r)
        rows.append(
            {
                "월_대형마트_지출_가정": spend,
                "명목혜택률": REAL_PRODUCT_NOMINAL_R,
                "한도_적용됨": capped,
                "실효혜택률": effective_r,
                "f(소매유통_가정)": f,
                "g*(실효혜택률_기준)": g,
                "threshold이하": bool(np.isfinite(g) and g <= threshold),
            }
        )
    result = pd.DataFrame(rows)

    spend_for_r_eq_f = REAL_PRODUCT_MONTHLY_CAP / f  # 실효r == f가 되는 지출액(g* 발산 경계)

    print("=== 5. 실제 카드 상품(대형마트 10% 할인, 월 한도 1.5만원) 대비 검증 (H10) ===")
    print(result.to_string(index=False))
    print(
        f"\n실효 혜택률이 f({f:.2%})와 같아지는(=g* 발산 경계) 월 대형마트 지출액: "
        f"약 {spend_for_r_eq_f:,.0f}원. 이보다 적게 쓰면 실효 혜택률이 f를 넘어 g*=inf(성립 "
        f"불가), 이보다 많이 써야 g*가 유한해진다 — 그런데 유한해져도(예: 150만원 지출 시 "
        f"g*={g_star(f, REAL_PRODUCT_MONTHLY_CAP/1_500_000):.1%}) threshold({threshold:.0%})는 "
        f"훨씬 못 미친다. 즉 월 한도가 '실효 혜택률을 낮춰 성립시키는' 장치라기보다, 모델이 "
        f"가정하지 않은 다른 방식(전월실적 조건으로 다른 업종 소비까지 끌어들이는 락인, 절대"
        f"손실액 자체를 한도로 캡핑해 세그먼트 단위 감당 가능한 수준으로 묶는 것)으로 "
        f"손익을 맞추고 있을 가능성을 시사한다."
    )
    return result


def main() -> None:
    if not CLEANED_PATH.exists():
        raise SystemExit(f"[중단] {CLEANED_PATH} 없음. 먼저 02_clean.py 실행할 것.")
    cfg = load_config()

    fee_df = fee_sensitivity(cfg)
    inflow_df = inflow_sensitivity()
    k_df = k_sensitivity(cfg)
    elasticity_sweep_df, elasticity_required_df = elasticity_sensitivity(cfg)
    elasticity_cross_count_df, elasticity_cross_required_df = elasticity_mapping_cross(cfg)
    real_product_df = real_product_check(cfg)

    OUT_SENSITIVITY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SENSITIVITY, "w", encoding="utf-8-sig") as f:
        f.write("## 1. 수수료율 매핑 민감도\n")
        fee_df.to_csv(f, index=False)
        f.write("\n## 2. 유입형 상권 포함/제외 비교\n")
        inflow_df.to_csv(f)
        f.write("\n## 3. k(클러스터 수) 민감도\n")
        k_df.to_csv(f, index=False)
        f.write("\n## 4a. 탄력성 배수 스윕\n")
        elasticity_sweep_df.to_csv(f, index=False)
        f.write("\n## 4b. 세그먼트 1위 업종 성립에 필요한 탄력성 배수\n")
        elasticity_required_df.to_csv(f, index=False)
        f.write("\n## 5. 실제 카드 상품(대형마트 10% 할인) 대비 검증\n")
        real_product_df.to_csv(f, index=False)
        f.write("\n## 6a. 매핑 x 탄력성 교차 — 성립 건수\n")
        elasticity_cross_count_df.to_csv(f, index=False)
        f.write("\n## 6b. 매핑 x 탄력성 교차 — 세그먼트 1위 업종 필요 배수\n")
        elasticity_cross_required_df.to_csv(f, index=False)
    print(f"\n[완료] {OUT_SENSITIVITY}")


if __name__ == "__main__":
    main()
