# 입력: outputs/tables/segment_industry_amt.csv, outputs/tables/segment_profile.csv,
#       config/assumptions.yaml
# 출력: outputs/tables/bep_heatmap.csv, outputs/tables/product_proposals.csv,
#       outputs/figures/bep_curve.png
# 목적: 세그먼트 × 업종 조합별로 손익분기 필요 증분 이용률 g*를 역산하고
#       (PROJECT_SPEC.md 2절 핵심 수식), g_star_threshold(config, 사용자 확정 0.50) 이하
#       성립 구간에서 카드 상품 3안 초안을 만든다.
#
#   g* = r / (f - r)   (f > r 일 때만 유효. f <= r이면 할인 자체가 성립 불가.)
#   f는 업종별 가정치(config/assumptions.yaml merchant_fee_rate) — 매출구간별 공시를
#   업종에 매핑한 것이라 프로젝트에서 가장 근거가 약한 가정. L4 민감도 분석 최우선 대상.
#
# product_proposals.csv는 실측 g*/현재 지출 비중을 근거로 한 AI 초안이다. 특히 "전략설명"·
# "리스크"는 데이터가 아니라 해석이므로 사람이 검토·수정할 것 (CLAUDE.md 규칙 2, 4).

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

SEGMENT_INDUSTRY_PATH = Path("outputs/tables/segment_industry_amt.csv")
SEGMENT_PROFILE_PATH = Path("outputs/tables/segment_profile.csv")
CONFIG_PATH = Path("config/assumptions.yaml")
OUT_HEATMAP = Path("outputs/tables/bep_heatmap.csv")
OUT_PROPOSALS = Path("outputs/tables/product_proposals.csv")
OUT_FIGURE = Path("outputs/figures/bep_curve.png")

# dataviz 스킬 palette.md 기준 categorical 슬롯 1~5 — g*는 f에만 의존하므로 업종을
# f(수수료율) 기준으로 묶어 최대 5개 곡선으로 표현(9개 업종이 5개 구간으로 수렴)
CURVE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
Y_CAP = 3.0  # g* 300%까지만 표시(그 이상은 화면 밖으로 발산 — 곡선 방향으로 충분히 전달됨)

# 세그먼트별 전략 서술(해석) — 실측 g*/지출비중을 근거로 한 AI 초안. 사람 검토 필요.
STRATEGY_NOTES = {
    "cluster0": (
        "소매/유통·의료/건강은 할인으로 성립(g*<=50%). 3위 지출인 음식(f=0.40%)은 "
        "할인 자체가 불가해 혜택에서 제외 — 체감 혜택이 낮게 느껴질 리스크 있음.",
        "고령층 방문 채널(오프라인/앱) 확인 필요, 의료비는 계절성(감기철 등) 반영 안 됨",
    ),
    "cluster1": (
        "소매/유통만 즉시할인으로 성립. 2위 지출(음식 29.1%, f=0.40%)은 즉시할인 불가 — "
        "PROJECT_SPEC.md 확장논점대로 피크시간대(토 17-19시)로 좁혀 실적조건부 이연적립"
        "(즉시할인 대신 익월 캐시백 + 최소실적 락인)으로 대체 제안.",
        "이연적립은 즉시 체감 혜택이 약해 유인력이 낮을 수 있음. r을 조금만 올려도 g* 급등(민감도 큼)",
    ),
    "cluster2": (
        "최대 볼륨 세그먼트(월 1.4억 건). 소매/유통만 즉시할인 성립. 미디어/통신·공공/기업/"
        "단체는 g* 여유(40%대)가 있으나 이 세그먼트의 주 지출 업종이 아님 — 통신비 캐시백을 "
        "락인용 결합 혜택으로 걸고, 생활서비스·음식은 연회비 선회수로 비용 일부를 상쇄하는 구조 제안.",
        "통신 캐시백이 매력적이지 않으면 결합 유인 약함. 연회비 선회수는 초기 가입 장벽 리스크",
    ),
}


def g_star(f: float, r: float) -> float:
    """손익분기 필요 증분 이용률. f<=r이면 성립 불가이므로 inf 반환."""
    if f <= r:
        return np.inf
    return r / (f - r)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    if not SEGMENT_INDUSTRY_PATH.exists():
        raise SystemExit(f"[중단] {SEGMENT_INDUSTRY_PATH} 없음. 먼저 03_segment.py 실행할 것.")
    cfg = load_config()
    fee_rates = cfg["merchant_fee_rate"]
    if all(v is None for v in fee_rates.values()):
        raise SystemExit(
            "[중단] config/assumptions.yaml의 merchant_fee_rate가 전부 null이다. "
            "여신금융협회 공시를 확인해 실제 값을 채운 뒤 다시 실행할 것."
        )

    seg_industry = pd.read_csv(SEGMENT_INDUSTRY_PATH)
    sweep = cfg["benefit_rate_sweep"]
    r_values = np.round(np.arange(sweep["min"], sweep["max"] + sweep["step"], sweep["step"]), 4)

    rows = []
    for _, row in seg_industry.iterrows():
        industry = row["card_tpbuz_nm_1"]
        f = fee_rates.get(industry)
        if f is None:
            continue
        for r in r_values:
            g = g_star(f, r)
            rows.append(
                {
                    "세그먼트": row["segment"],
                    "업종": industry,
                    "f(수수료율)": f,
                    "r(혜택률)": r,
                    "g*(필요증분이용률)": g,
                    "현재월매출액": round(row["월추정매출액"]),
                    "필요증분매출액": round(row["월추정매출액"] * g) if np.isfinite(g) else None,
                    "월혜택지급액(현재기준)": round(row["월추정매출액"] * r),
                }
            )

    heatmap = pd.DataFrame(rows)
    OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    heatmap.to_csv(OUT_HEATMAP, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_HEATMAP} ({len(heatmap):,} rows)")

    print("\n=== r=0.5%, r=1.0% 스냅샷 (세그먼트×업종별 g*) ===")
    for r_snap in [0.005, 0.010]:
        snap = heatmap[np.isclose(heatmap["r(혜택률)"], r_snap)]
        pivot = snap.pivot(index="세그먼트", columns="업종", values="g*(필요증분이용률)")
        print(f"\n--- r={r_snap:.1%} ---")
        print(pivot.round(2).to_string())

    build_proposals(heatmap, cfg["g_star_threshold"])
    plot_bep_curve(fee_rates, cfg["g_star_threshold"], sweep)


def plot_bep_curve(fee_rates: dict, threshold: float, sweep: dict) -> None:
    """혜택률(r) 대비 필요 증분 이용률(g*) 곡선. g*는 f에만 의존하므로 업종을 f로 묶는다."""
    tiers: dict[float, list[str]] = {}
    for industry, f in fee_rates.items():
        tiers.setdefault(f, []).append(industry)

    r_fine = np.linspace(sweep["min"], sweep["max"], 200)
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (f, industries) in enumerate(sorted(tiers.items())):
        g_values = np.array([g_star(f, r) for r in r_fine])
        g_capped = np.clip(g_values, None, Y_CAP)
        color = CURVE_COLORS[i % len(CURVE_COLORS)]
        label = f"f={f:.2%} ({'/'.join(industries)})"
        ax.plot(r_fine * 100, g_capped * 100, color=color, linewidth=2, label=label)
        # f<=r을 넘어 발산하는 지점부터는 점선으로 화면 상단까지 표시
        diverged = g_values >= Y_CAP
        if diverged.any():
            ax.plot(
                r_fine[diverged] * 100, np.full(diverged.sum(), Y_CAP * 100),
                color=color, linewidth=2, linestyle=":",
            )

    ax.axhline(threshold * 100, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(
        sweep["max"] * 100, threshold * 100, f" g*={threshold:.0%} 임계선",
        color=INK_MUTED, fontsize=9, va="bottom", ha="right",
    )

    ax.set_xlabel("혜택률 r (%)", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("필요 증분 이용률 g* (%, 300%에서 상단 절단)", color=INK_PRIMARY, fontsize=10)
    ax.set_title("혜택률 대비 필요 증분 이용률 (업종을 수수료율 구간으로 묶음)", color=INK_PRIMARY, fontsize=12)
    ax.set_ylim(0, Y_CAP * 100 * 1.05)
    ax.grid(color=GRID, linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_PRIMARY, loc="upper left")

    fig.tight_layout()
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURE, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"[완료] {OUT_FIGURE}")


def build_proposals(heatmap: pd.DataFrame, threshold: float) -> None:
    if not SEGMENT_PROFILE_PATH.exists():
        raise SystemExit(f"[중단] {SEGMENT_PROFILE_PATH} 없음. 먼저 03_segment.py 실행할 것.")
    profile = pd.read_csv(SEGMENT_PROFILE_PATH).set_index("segment")

    feasible = heatmap[heatmap["g*(필요증분이용률)"] <= threshold]
    rows = []
    for seg in ["cluster0", "cluster1", "cluster2"]:
        seg_feasible = feasible[feasible["세그먼트"] == seg]
        best_per_industry = seg_feasible.loc[seg_feasible.groupby("업종")["r(혜택률)"].idxmax()]
        # 이 세그먼트의 top3 지출 업종 중 실제로 할인이 성립하는 업종만 혜택 대상으로 채택
        top3_industries = [x.split("(")[0] for x in profile.loc[seg, "주력업종_top3"].split(" / ")]
        chosen = best_per_industry[best_per_industry["업종"].isin(top3_industries)]

        strategy, risk = STRATEGY_NOTES[seg]
        rows.append(
            {
                "상품안": seg,
                "제안이름(초안)": profile.loc[seg, "제안이름(초안, 사람 검토 필요)"],
                "타겟세그먼트": f"{seg} ({profile.loc[seg, '구성원(성별-연령)']})",
                "혜택업종·혜택률": "; ".join(f"{r['업종']} {r['r(혜택률)']:.1%}" for _, r in chosen.iterrows()),
                "g*(해당업종)": "; ".join(f"{r['업종']} {r['g*(필요증분이용률)']:.1%}" for _, r in chosen.iterrows()),
                "월혜택지급액합계(현재기준)": int(chosen["월혜택지급액(현재기준)"].sum()),
                "혜택 적용 피크시간대": f"{profile.loc[seg, '피크요일']}·{profile.loc[seg, '피크시간대']}",
                "전략설명(AI초안, 검토필요)": strategy,
                "리스크(AI초안, 검토필요)": risk,
            }
        )

    proposals = pd.DataFrame(rows)
    OUT_PROPOSALS.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(OUT_PROPOSALS, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {OUT_PROPOSALS}")
    print(proposals.to_string(index=False))


if __name__ == "__main__":
    main()
