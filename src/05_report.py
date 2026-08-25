# 입력: data/processed/02_cleaned.parquet, outputs/tables/segment_industry_amt.csv,
#       outputs/tables/product_proposals.csv, config/assumptions.yaml
# 출력: outputs/tables/sensitivity.csv
# 목적: L4 민감도와 반증. (PROJECT_SPEC.md L4)
#   1. 수수료율(f) 매핑을 한 단계 낙관/비관으로 흔들었을 때 "할인 성립 여부"가 뒤집히는 업종 탐색
#   2. 유입형 상권 제외 전후로 세그먼트 소비 구성이 얼마나 바뀌는지 비교
#   3. 클러스터 개수 k를 바꿨을 때 "소매/유통만 확실히 성립" 결론이 유지되는지 확인
#   결론이 무너지는 조건은 숨기지 않고 그대로 기록한다.

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

CLEANED_PATH = Path("data/processed/02_cleaned.parquet")
SEGMENT_INDUSTRY_PATH = Path("outputs/tables/segment_industry_amt.csv")
CONFIG_PATH = Path("config/assumptions.yaml")
OUT_SENSITIVITY = Path("outputs/tables/sensitivity.csv")

FEE_TIERS = [0.0040, 0.0100, 0.0115, 0.0145, 0.0208]  # config/assumptions.yaml fee_rate_by_revenue_tier 순서
EXCLUDED_AGES = {1, 11}
REF_R = 0.003  # 최소 유의미 혜택률(0.3%) 기준으로 성립 여부 판정


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


def main() -> None:
    if not CLEANED_PATH.exists():
        raise SystemExit(f"[중단] {CLEANED_PATH} 없음. 먼저 02_clean.py 실행할 것.")
    cfg = load_config()

    fee_df = fee_sensitivity(cfg)
    inflow_df = inflow_sensitivity()
    k_df = k_sensitivity(cfg)

    OUT_SENSITIVITY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SENSITIVITY, "w", encoding="utf-8-sig") as f:
        f.write("## 1. 수수료율 매핑 민감도\n")
        fee_df.to_csv(f, index=False)
        f.write("\n## 2. 유입형 상권 포함/제외 비교\n")
        inflow_df.to_csv(f)
        f.write("\n## 3. k(클러스터 수) 민감도\n")
        k_df.to_csv(f, index=False)
    print(f"\n[완료] {OUT_SENSITIVITY}")


if __name__ == "__main__":
    main()
