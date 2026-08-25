# 입력: data/processed/01_loaded.parquet, config/assumptions.yaml
# 출력: data/processed/02_cleaned.parquet, outputs/figures/consumption_rhythm.png
#       (PROJECT_SPEC.md L1 산출물 — hour/age/day 라벨과 유입형 상권 플래그가 이 단계에서
#       나오므로 여기서 같이 생성한다)
# 목적: 코드값 라벨링(hour/age/day, 경기도 시군 민간데이터 규격서_카드.pdf 기준),
#       유입형 상권(사업자 소재지 집중 행정동) 식별·플래그, 건당 평균 결제액 파생 컬럼 생성
#       (근거: docs/limitations.md 2번 항목 — 성남시 특정 행정동 매출 이상 집중 실측)

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = Path("data/processed/01_loaded.parquet")
CONFIG_PATH = Path("config/assumptions.yaml")
OUT_PATH = Path("data/processed/02_cleaned.parquet")
OUT_FIGURE = Path("outputs/figures/consumption_rhythm.png")

# dataviz 스킬 palette.md 기준 — sequential blue ramp, light-surface 크롬
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


# 경기도 시군 민간데이터 규격서_카드.pdf (data/raw/) 기준 코드값 매핑
HOUR_LABELS = {
    1: "00-07시", 2: "07-09시", 3: "09-11시", 4: "11-13시", 5: "13-15시",
    6: "15-17시", 7: "17-19시", 8: "19-21시", 9: "21-23시", 10: "23-24시",
}
AGE_LABELS = {
    1: "0-9세", 2: "10대", 3: "20대", 4: "30대", 5: "40대", 6: "50대",
    7: "60대", 8: "70대", 9: "80대", 10: "90대", 11: "100세 이상",
}
DAY_LABELS = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}


def load() -> pd.DataFrame:
    if not IN_PATH.exists():
        raise SystemExit(f"[중단] {IN_PATH} 없음. 먼저 01_load.py 실행할 것.")
    return pd.read_parquet(IN_PATH)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_label"] = df["hour"].map(HOUR_LABELS)
    df["age_label"] = df["age"].map(AGE_LABELS)
    df["day_label"] = df["day"].map(DAY_LABELS)
    unmapped = df[["hour_label", "age_label", "day_label"]].isna().sum()
    if unmapped.sum():
        print(f"[경고] 매핑 안 된 코드값 존재: {unmapped[unmapped > 0].to_dict()}")
    return df


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def flag_inflow_areas(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """행정동(admi_cty_no)별 [업종(card_tpbuz_nm_2) 상위 2개 매출 비중]과
    [행정동 총매출 규모]를 기준으로 유입형 상권을 플래그한다.
    기준: config/assumptions.yaml의 inflow_detection 참고.
    """
    params = cfg["inflow_detection"]

    by_category = df.groupby(["admi_cty_no", "card_tpbuz_nm_2"])["amt"].sum().reset_index()
    admi_total = by_category.groupby("admi_cty_no")["amt"].sum().rename("admi_total")
    by_category = by_category.merge(admi_total, on="admi_cty_no")
    by_category["share"] = by_category["amt"] / by_category["admi_total"]

    top2_share = (
        by_category.sort_values("share", ascending=False)
        .groupby("admi_cty_no")
        .head(2)
        .groupby("admi_cty_no")["share"]
        .sum()
        .rename("top2_category_share")
    )

    profile = pd.concat([top2_share, admi_total], axis=1)
    amt_cutoff = profile["admi_total"].quantile(params["total_amt_percentile_threshold"])

    inflow_ids = profile[
        (profile["top2_category_share"] >= params["top2_category_share_threshold"])
        & (profile["admi_total"] >= amt_cutoff)
    ].index

    print(f"[유입형 상권 플래그] {len(inflow_ids)}개 행정동 / 전체 {len(profile)}개")
    print(f"[유입형 상권 매출 비중] {profile.loc[inflow_ids, 'admi_total'].sum() / profile['admi_total'].sum():.1%}")

    df["is_inflow_area"] = df["admi_cty_no"].isin(inflow_ids)
    return df


def add_avg_payment(df: pd.DataFrame) -> pd.DataFrame:
    df["avg_payment"] = df["amt"] / df["cnt"].replace(0, pd.NA)
    return df


def plot_consumption_rhythm(df: pd.DataFrame) -> None:
    """시간대(hour_label) × 요일(day_label) 매출액 비중 히트맵. 유입형 상권 제외 기준."""
    day_order = ["월", "화", "수", "목", "금", "토", "일"]
    hour_order = [HOUR_LABELS[i] for i in range(1, 11)]

    sub = df[~df["is_inflow_area"]]
    pivot = sub.groupby(["day_label", "hour_label"], observed=True)["amt"].sum().unstack("hour_label")
    pivot = pivot.reindex(index=day_order, columns=hour_order)
    share = pivot / pivot.values.sum() * 100  # 전체 대비 비중(%)

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(share.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(hour_order)), hour_order, rotation=45, ha="right", color=INK_PRIMARY, fontsize=9)
    ax.set_yticks(range(len(day_order)), day_order, color=INK_PRIMARY, fontsize=10)
    ax.set_title("시간대 × 요일 소비 비중 (유입형 상권 제외, 4개월 합산)", color=INK_PRIMARY, fontsize=12, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(len(day_order)):
        for j in range(len(hour_order)):
            val = share.values[i, j]
            text_color = INK_PRIMARY if val < share.values.max() * 0.6 else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("매출액 비중 (%)", color=INK_MUTED, fontsize=9)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[완료] {OUT_FIGURE}")


def main() -> None:
    cfg = load_config()
    df = load()
    df = add_labels(df)
    df = flag_inflow_areas(df, cfg)
    df = add_avg_payment(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[완료] {OUT_PATH} ({len(df):,} rows)")

    plot_consumption_rhythm(df)


if __name__ == "__main__":
    main()
