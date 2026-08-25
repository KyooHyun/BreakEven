# 입력: data/processed/02_cleaned.parquet, config/assumptions.yaml (random_seed)
# 출력: outputs/tables/segment_profile.csv, outputs/tables/segment_industry_amt.csv,
#       outputs/figures/segment_radar.png
#       (segment_industry_amt.csv: 04_economics.py의 g* 히트맵 입력)
# 목적: 성연령 조합별 [업종중분류 × 시간대 × 요일] 소비 비중 벡터를 만들고 클러스터링해
#       소비 리듬 세그먼트를 도출한다. (PROJECT_SPEC.md L2)
#       유입형 상권(is_inflow_area=True, docs/limitations.md 2번 항목)은 제외.
#
# 실측 결과 기록 (docs/hypothesis-log.md H4 참고):
#   업종대분류(9개)·업종중분류(85개) 어느 쪽으로 벡터를 만들어도, 시간대·요일 축이
#   독립적인 세그먼트를 만들어내지 못하고 연령대가 지배적인 축으로 나타남 (성별은
#   거의 영향 없음 — 60대만 예외적으로 성별에 따라 다른 클러스터로 갈림).
#   → 사용자 판단(2026-08-25)에 따라 연령대 3단계(청년/중장년/고령)로 확정하고 L3로 진행.
#   0-9세, 100세 이상은 카드 이용량이 극소해 별도 표기(카드 상품 설계 대상 제외).

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = Path("data/processed/02_cleaned.parquet")
CONFIG_PATH = Path("config/assumptions.yaml")
OUT_TABLE = Path("outputs/tables/segment_profile.csv")
OUT_INDUSTRY = Path("outputs/tables/segment_industry_amt.csv")
OUT_FIGURE = Path("outputs/figures/segment_radar.png")

# dataviz 스킬 palette.md 기준 categorical 슬롯 1~3 (all-pairs 검증된 조합)
SEGMENT_COLORS = {"cluster0": "#2a78d6", "cluster1": "#eb6834", "cluster2": "#1baf7a"}
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

GROUP_COLS = ["sex", "age_label"]
VECTOR_COLS = ["card_tpbuz_nm_2", "hour_label", "day_label"]
N_MONTHS = 4  # 2025-12, 2026-01~03 (데이터 기준 시점, docs/limitations.md 3번 항목)
K_FINAL = 3  # 실측 결과에 따라 확정 (스크립트 상단 주석 참고)
EXCLUDED_AGES = {1, 11}  # 0-9세, 100세 이상 — 카드 이용 극소


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rhythm_vectors(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.groupby(GROUP_COLS + VECTOR_COLS, observed=True)["amt"].sum().unstack(VECTOR_COLS, fill_value=0)
    return pivot.div(pivot.sum(axis=1), axis=0)


def profile_group(g: pd.DataFrame) -> dict:
    top3 = g.groupby("card_tpbuz_nm_1", observed=True)["amt"].sum().nlargest(3)
    peak_hour = g.groupby("hour_label", observed=True)["amt"].sum().idxmax()
    peak_day = g.groupby("day_label", observed=True)["amt"].sum().idxmax()
    avg_payment = g["amt"].sum() / g["cnt"].sum()
    monthly_est_cnt = g["cnt"].sum() / N_MONTHS
    return {
        "주력업종_top3": " / ".join(f"{k}({v / g['amt'].sum():.1%})" for k, v in top3.items()),
        "피크시간대": peak_hour,
        "피크요일": peak_day,
        "건당평균결제액": round(avg_payment),
        "월추정이용건수": round(monthly_est_cnt),
    }


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"[중단] {IN_PATH} 없음. 먼저 02_clean.py 실행할 것.")
    cfg = load_config()
    df = pd.read_parquet(IN_PATH)
    df = df[~df["is_inflow_area"]]

    main_df = df[~df["age"].isin(EXCLUDED_AGES)]
    vectors = build_rhythm_vectors(main_df)
    print(f"세그먼트 원소(성연령 조합) 수: {len(vectors)}, 벡터 차원: {vectors.shape[1]}")

    km = KMeans(n_clusters=K_FINAL, random_state=cfg["random_seed"], n_init=10)
    labels = pd.Series(km.fit_predict(vectors), index=vectors.index, name="cluster")
    main_df = main_df.merge(labels, left_on=GROUP_COLS, right_index=True)

    cluster_avg_payment = {cid: g["amt"].sum() / g["cnt"].sum() for cid, g in main_df.groupby("cluster")}
    payment_rank = {
        cid: ("소액" if p == min(cluster_avg_payment.values()) else "고액" if p == max(cluster_avg_payment.values()) else "중간")
        for cid, p in cluster_avg_payment.items()
    }

    rows = []
    for cluster_id, g in main_df.groupby("cluster"):
        members = sorted({f"{s}-{a}" for s, a in g[GROUP_COLS].drop_duplicates().itertuples(index=False)})
        p = profile_group(g)
        rows.append(
            {
                "segment": f"cluster{cluster_id}",
                "구성원(성별-연령)": ", ".join(members),
                **p,
                "제안이름(초안, 사람 검토 필요)": f"{p['피크요일']}·{p['피크시간대']} {payment_rank[cluster_id]}결제형",
            }
        )

    excluded_df = df[df["age"].isin(EXCLUDED_AGES)]
    for (sex, age_label), g in excluded_df.groupby(["sex", "age_label"], observed=True):
        rows.append(
            {
                "segment": "제외(카드이용 극소)",
                "구성원(성별-연령)": f"{sex}-{age_label}",
                **profile_group(g),
            }
        )

    profile = pd.DataFrame(rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_TABLE}")
    print(profile.to_string(index=False))

    # 세그먼트(cluster0~2, 카드 상품 설계 대상만) × 업종대분류 매출/건수 — L3(04_economics.py) 입력
    industry = (
        main_df.groupby(["cluster", "card_tpbuz_nm_1"], observed=True)
        .agg(amt=("amt", "sum"), cnt=("cnt", "sum"))
        .reset_index()
    )
    industry["segment"] = "cluster" + industry["cluster"].astype(str)
    industry["월추정매출액"] = industry["amt"] / N_MONTHS
    industry["월추정이용건수"] = industry["cnt"] / N_MONTHS
    industry = industry[["segment", "card_tpbuz_nm_1", "월추정매출액", "월추정이용건수"]]
    industry.to_csv(OUT_INDUSTRY, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_INDUSTRY}")

    plot_segment_radar(industry)


def plot_segment_radar(industry: pd.DataFrame) -> None:
    """세그먼트별 업종대분류 매출 비중 레이더 차트."""
    categories = sorted(industry["card_tpbuz_nm_1"].unique())
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    for seg in ["cluster0", "cluster1", "cluster2"]:
        seg_df = industry[industry["segment"] == seg].set_index("card_tpbuz_nm_1")
        share = (seg_df["월추정매출액"] / seg_df["월추정매출액"].sum() * 100).reindex(categories, fill_value=0)
        values = share.tolist()
        values += values[:1]
        color = SEGMENT_COLORS[seg]
        ax.plot(angles, values, color=color, linewidth=2, label=seg)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=INK_PRIMARY, fontsize=10)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=8)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID)
    ax.set_title("세그먼트별 업종대분류 매출 비중 (%)", color=INK_PRIMARY, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False, labelcolor=INK_PRIMARY)

    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[완료] {OUT_FIGURE}")


if __name__ == "__main__":
    main()
