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


def main() -> None:
    if not CLEANED_PATH.exists():
        raise SystemExit(f"[중단] {CLEANED_PATH} 없음. 먼저 02_clean.py 실행할 것.")
    cfg = load_config()

    fee_df = fee_sensitivity(cfg)
    inflow_df = inflow_sensitivity()
    k_df = k_sensitivity(cfg)
    elasticity_sweep_df, elasticity_required_df = elasticity_sensitivity(cfg)

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
    print(f"\n[완료] {OUT_SENSITIVITY}")


if __name__ == "__main__":
    main()
