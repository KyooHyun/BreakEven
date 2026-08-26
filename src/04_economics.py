# 입력: outputs/tables/segment_industry_amt.csv, outputs/tables/segment_profile.csv,
#       config/assumptions.yaml
# 출력: outputs/tables/bep_heatmap.csv, outputs/tables/product_proposals.csv,
#       outputs/tables/structural_alternatives.csv, outputs/figures/bep_curve.png
# 목적: 세그먼트 × 업종 조합별로 손익분기 필요 증분 이용률 g*를 역산하고
#       (PROJECT_SPEC.md 2절 핵심 수식), g_star_threshold(config, 사용자 확정 0.50) 이하
#       성립 구간에서 카드 상품 3안 초안을 만든다.
#
#   g* = r / (f - r)   (f > r 일 때만 유효. f <= r이면 할인 자체가 성립 불가.)
#   f는 업종별 가정치(config/assumptions.yaml merchant_fee_rate) — 매출구간별 공시를
#   업종에 매핑한 것이라 프로젝트에서 가장 근거가 약한 가정. L4 민감도 분석 최우선 대상.
#
# g*는 f·r에만 의존하고 세그먼트 변수가 없다(H6, docs/hypothesis-log.md) — "성립 구간"이
# threshold 기준으로는 세그먼트 무관하게 동일하게 나온다는 구조적 한계가 있다. 이를
# 보완하기 위해 두 가지를 추가한다(H7·H8, docs/hypothesis-log.md):
#   1. 세그먼트별 혜택 반응 탄력성 가정(segment_response_elasticity)으로 "이 세그먼트가
#      실제로 달성 가능한 증분 이용률"을 별도 추정해 threshold 기준 성립 여부와 대조.
#   2. 즉시할인 구조를 이연적립(breakage)·연회비 선회수(annual fee)로 바꿨을 때 g*가
#      얼마나 내려오는지 계산 — 즉시할인으로는 불가능했던 2위 지출 업종(음식)에 적용.
# 두 가정 모두 근거가 얇은 판단치이며(config/assumptions.yaml 주석 참고) L4 민감도 대상.
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
OUT_STRUCTURAL = Path("outputs/tables/structural_alternatives.csv")
OUT_FIGURE = Path("outputs/figures/bep_curve.png")

# 구조 실험 대상 업종 — cluster1·cluster2 모두 2위 지출 업종(음식)이 즉시할인으로는
# 성립 불가(H6)했으므로, 이연적립·연회비 선회수 구조를 이 업종에 적용해본다.
STRUCTURAL_TARGET_INDUSTRY = "음식"

# dataviz 스킬 palette.md 기준 categorical 슬롯 1~5 — g*는 f에만 의존하므로 업종을
# f(수수료율) 기준으로 묶어 최대 5개 곡선으로 표현(9개 업종이 5개 구간으로 수렴)
CURVE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
Y_CAP = 3.0  # g* 300%까지만 표시(그 이상은 화면 밖으로 발산 — 곡선 방향으로 충분히 전달됨)

# 세그먼트별 전략 서술(해석) — 실측 g*/지출비중과 structural_alternatives.csv(H8) 결과를
# 근거로 한 AI 초안. 카드 마케팅은 상품마케팅이 아니라 서비스마케팅이라 "매 결제 순간
# 선택받아야 한다"는 관점, 그리고 "고객이 원하는 혜택 vs 카드사 수익성"의 균형점을 찾는다는
# 관점으로 서술했다. 여전히 해석이므로 사람이 검토·수정할 것 (CLAUDE.md 규칙 2, 4).
STRATEGY_NOTES = {
    "cluster0": (
        "고령층(60대 이상)은 소매/유통·의료/건강 지출 비중이 커 이 두 업종만으로도 즉시할인이 "
        "성립(g*<=50%). 이 세그먼트는 threshold 기준으로 이미 균형점을 찾은 경우라 구조를 "
        "복잡하게 가져갈 이유가 없음 — 다만 3위 지출인 음식(f=0.40%)은 할인 자체가 불가능해 "
        "혜택 대상에서 빠지므로, '매 결제 순간 선택받는다'는 관점에서는 체감 혜택의 폭이 좁게 "
        "느껴질 수 있음.",
        "고령층 방문 채널(오프라인/앱) 확인 필요, 의료비는 계절성(감기철 등) 반영 안 됨. "
        "탄력성 가정(H7)을 적용하면 이 세그먼트는 소매/유통조차 실제 달성 가능성이 낮게 나와 "
        "(필요 배수 7.06배, H9) threshold 통과를 곧바로 '고객이 실제로 반응할 것'으로 읽으면 안 됨.",
    ),
    "cluster1": (
        "청년층(10~20대)은 소매/유통만 즉시할인으로 성립하고, 2위 지출인 음식(29.1%, f=0.40%)은 "
        "즉시할인으로는 불가능함. 이연적립(익월 캐시백 + 최소실적 락인)으로 실효 비용을 낮춰봤지만 "
        "가정 소멸률(30%) 기준 g*=110.5%로 threshold(50%)를 못 넘었음(충족에 필요한 소멸률은 "
        "55.6%, H8) — 지금 가정치로는 음식 혜택까지 얹은 균형점을 찾지 못했다는 뜻이라, 음식 "
        "혜택을 포기하거나 소멸률을 더 뒷받침할 근거를 찾는 것 중 하나를 택해야 함. 다만 이 "
        "세그먼트는 세 세그먼트 중 탄력성 가정이 가장 높아(0.20) 소매/유통 성립에 필요한 배수가 "
        "가장 작음(1.76배, H9) — '혜택에 조금만 유리한 가정이 붙어도 성립 가능성이 가장 높은 "
        "세그먼트'라는 점이 청년층 대상 상품의 실질적 차별점.",
        "이연적립은 즉시 체감 혜택이 약해 유인력이 낮을 수 있음. r을 조금만 올려도 g* 급등"
        "(민감도 큼). 음식 혜택을 포기하면 '2위 지출 업종은 혜택 없음'이 서비스 경험의 공백으로 "
        "느껴질 리스크가 있음.",
    ),
    "cluster2": (
        "최대 볼륨 세그먼트(월 1.4억 건, 중장년 30~60대). 소매/유통만 즉시할인 성립하지만, 2위 "
        "지출인 음식은 연회비 선회수 구조를 적용하면 세그먼트 카드수(약 946만 장 추정) 기준 "
        "연회비 수입 풀의 약 20%만 배정해도 threshold를 충족함(H8) — cluster1과 같은 '대안'을 "
        "검토했는데 이 세그먼트에서는 실제로 균형점이 성립한다는 게 계산으로 확인됨. 볼륨이 가장 "
        "크기 때문에 연회비 풀 자체도 커서(월 약 118억원 추정) 구조 여력이 생긴 것으로 보임.",
        "이 계산은 연회비 풀 전체를 음식 한 카테고리에만 배정 가능하다고 가정한 것이라(다른 "
        "혜택과의 경쟁 관계 미반영) 낙관적일 수 있음(docs/limitations.md 7번). 탄력성 가정(H7)을 "
        "적용하면 이 세그먼트도 소매/유통 성립에 3.53배가 필요해(H9), threshold 통과를 곧바로 "
        "'고객이 실제로 반응할 것'으로 읽으면 안 됨.",
    ),
}


def g_star(f: float, r: float) -> float:
    """손익분기 필요 증분 이용률. f<=r이면 성립 불가이므로 inf 반환."""
    if f <= r:
        return np.inf
    return r / (f - r)


def expected_g(r: float, r_min: float, elasticity_coef: float) -> float:
    """세그먼트별 혜택 반응 탄력성 가정 하 '실제 달성 가능한' 증분 이용률.
    (config/assumptions.yaml segment_response_elasticity — 근거 얇은 판단치, 선형 가정)"""
    return elasticity_coef * (r / r_min)


def g_star_deferred(f: float, r: float, breakage_rate: float) -> float:
    """이연적립 구조. 소멸률만큼 실효 혜택률이 낮아져 카드사 실효 비용이 줄어든다."""
    effective_r = r * (1 - breakage_rate)
    return g_star(f, effective_r)


def required_breakage_rate(f: float, r: float, threshold: float) -> float:
    """g*_deferred가 threshold와 같아지려면 필요한 소멸률. r 자체가 이미 threshold를
    만족하면 0. f<=r이면(할인 자체가 불가한 업종) 소멸률로는 구제 불가 -> inf."""
    if f <= r:
        return np.inf
    effective_r_required = threshold * f / (1 + threshold)
    if effective_r_required >= r:
        return 0.0
    return 1 - effective_r_required / r


def required_annual_fee_share(f: float, r: float, a_monthly: float, threshold: float, annual_fee_pool_monthly: float) -> float:
    """연회비 수입 중 이 업종 혜택 보전에 배정해야 하는 비율. g*_base가 이미
    threshold 이하면 0. f<=r이면(할인 자체가 불가) 연회비로는 구제 불가 -> inf.
    필요 배정액이 전체 연회비 풀을 넘으면(=100% 배정해도 부족) 그 비율(>1)을 그대로 반환."""
    if f <= r:
        return np.inf
    g_base = g_star(f, r)
    if g_base <= threshold:
        return 0.0
    required_af_monthly = (g_base - threshold) * a_monthly * (f - r)
    if annual_fee_pool_monthly <= 0:
        return np.inf
    return required_af_monthly / annual_fee_pool_monthly


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
    elasticity_cfg = cfg["segment_response_elasticity"]

    rows = []
    for _, row in seg_industry.iterrows():
        industry = row["card_tpbuz_nm_1"]
        f = fee_rates.get(industry)
        if f is None:
            continue
        seg = row["segment"]
        elasticity_coef = elasticity_cfg.get(seg)
        for r in r_values:
            g = g_star(f, r)
            g_expected = expected_g(r, sweep["min"], elasticity_coef) if elasticity_coef is not None else None
            rows.append(
                {
                    "세그먼트": seg,
                    "업종": industry,
                    "f(수수료율)": f,
                    "r(혜택률)": r,
                    "g*(필요증분이용률)": g,
                    "현재월매출액": round(row["월추정매출액"]),
                    "필요증분매출액": round(row["월추정매출액"] * g) if np.isfinite(g) else None,
                    "월혜택지급액(현재기준)": round(row["월추정매출액"] * r),
                    "threshold기준_성립": bool(np.isfinite(g) and g <= cfg["g_star_threshold"]),
                    "세그먼트예상반응g(탄력성가정)": g_expected,
                    "탄력성기준_성립": bool(g_expected is not None and np.isfinite(g) and g_expected >= g),
                }
            )

    heatmap = pd.DataFrame(rows)
    OUT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    heatmap.to_csv(OUT_HEATMAP, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_HEATMAP} ({len(heatmap):,} rows)")

    both = heatmap[heatmap["threshold기준_성립"] & heatmap["탄력성기준_성립"]]
    threshold_only = heatmap[heatmap["threshold기준_성립"] & ~heatmap["탄력성기준_성립"]]
    print(
        f"\n=== threshold 기준 vs 탄력성 기준 성립 판정 비교 (세그먼트×업종×r 조합 전체 {len(heatmap):,}건 중) ===\n"
        f"두 기준 모두 성립: {len(both):,}건 / threshold만 성립(탄력성 기준으로는 달성 어려움): {len(threshold_only):,}건"
    )

    print("\n=== r=0.5%, r=1.0% 스냅샷 (세그먼트×업종별 g*) ===")
    for r_snap in [0.005, 0.010]:
        snap = heatmap[np.isclose(heatmap["r(혜택률)"], r_snap)]
        pivot = snap.pivot(index="세그먼트", columns="업종", values="g*(필요증분이용률)")
        print(f"\n--- r={r_snap:.1%} ---")
        print(pivot.round(2).to_string())

    build_proposals(heatmap, cfg["g_star_threshold"])
    build_structural_alternatives(seg_industry, cfg)
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


def build_structural_alternatives(seg_industry: pd.DataFrame, cfg: dict) -> None:
    """즉시할인으로는 성립 불가한 2위 지출 업종(음식)에 이연적립(cluster1)·연회비
    선회수(cluster2) 구조를 적용했을 때 g*가 얼마나 내려오는지 계산한다 (H8).
    r은 즉시할인 상품안과 동일하게 benefit_rate_sweep 최솟값(0.3%)을 기준으로 잡는다."""
    threshold = cfg["g_star_threshold"]
    r = cfg["benefit_rate_sweep"]["min"]
    breakage = cfg["deferred_reward_breakage_rate"]
    annual_fee_monthly = cfg["annual_fee_krw"] / 12
    avg_tx_per_card = cfg["avg_monthly_transactions_per_card"]
    f = cfg["merchant_fee_rate"][STRUCTURAL_TARGET_INDUSTRY]

    seg_totals = seg_industry.groupby("segment")["월추정이용건수"].sum()

    target_rows = seg_industry[seg_industry["card_tpbuz_nm_1"] == STRUCTURAL_TARGET_INDUSTRY].set_index("segment")

    rows = []
    for seg, structure in [("cluster1", "이연적립"), ("cluster2", "연회비선회수")]:
        a_monthly = target_rows.loc[seg, "월추정매출액"]
        g_base = g_star(f, r)

        n_cards = seg_totals.loc[seg] / avg_tx_per_card
        annual_fee_pool_monthly = annual_fee_monthly * n_cards

        row = {
            "세그먼트": seg,
            "적용업종": STRUCTURAL_TARGET_INDUSTRY,
            "f(수수료율)": f,
            "r(혜택률)": r,
            "즉시할인_g*": g_base,
            "즉시할인_threshold이하": bool(np.isfinite(g_base) and g_base <= threshold),
            "구조": structure,
        }
        if structure == "이연적립":
            g_deferred = g_star_deferred(f, r, breakage)
            b_required = required_breakage_rate(f, r, threshold)
            row.update(
                {
                    "가정_소멸률": breakage,
                    "적용후_g*": g_deferred,
                    "적용후_threshold이하": bool(np.isfinite(g_deferred) and g_deferred <= threshold),
                    "threshold충족_필요소멸률": b_required,
                    "가정치로_충분한가": bool(breakage >= b_required),
                }
            )
        else:
            n_cards_round = round(n_cards)
            share_required = required_annual_fee_share(f, r, a_monthly, threshold, annual_fee_pool_monthly)
            row.update(
                {
                    "추정_세그먼트카드수": n_cards_round,
                    "연회비수입_월(전체풀)": round(annual_fee_pool_monthly),
                    "threshold충족_필요배정비율": share_required,
                    "가정치로_충분한가": bool(0 <= share_required <= 1),
                }
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    OUT_STRUCTURAL.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_STRUCTURAL, index=False, encoding="utf-8-sig")
    print(f"\n[완료] {OUT_STRUCTURAL}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
