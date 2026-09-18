# 입력: config/assumptions.yaml (merchant_fee_rate, benefit_rate_sweep, g_star_threshold,
#         woori_dual_network — 전부 로드만 하고 재정의하지 않는다)
#       outputs/tables/segment_industry_amt.csv (세그먼트 x 업종 월추정매출액)
#       src/04_economics.py (g_star 수식 import)
# 출력: outputs/tables/woori_c_implied.csv        (계산0: 대행수수료율 c 역산과 함축 결제액)
#       outputs/tables/woori_feff_gstar.csv       (계산1: 업종 x c x a x r -> f_eff, g*)
#       outputs/tables/woori_alpha_frontier.csv   (계산2: 성립에 필요한 독자망 비중 a*)
#       outputs/tables/woori_segment_impact.csv   (계산3: 세그먼트별 독자망 전환 효과)
#       outputs/tables/woori_industry_priority.csv(계산4: 어느 업종부터 독자망으로 끌어올 것인가)
#       outputs/tables/woori_leak_rank_robustness.csv (계산5: 세그먼트 잠식률 순위의 견고성)
#       outputs/tables/woori_threshold_sweep.csv  (계산6: threshold 45~60% 스윕)
#       outputs/figures/woori_alpha_frontier.png  (a 대비 g* 곡선, 실측 a 표시)
# 목적: 기존 모델의 f(가맹점 수수료율 하나)를 우리카드의 이원 결제망 구조로 확장한다(H19~H21).
#       같은 결제라도 독자가맹점에서 발생하면 BC카드 대행(프로세싱) 수수료가 붙지 않고,
#       BC 매입망을 타면 붙는다 -> 카드사가 실제로 쥐는 실효 수수료율이 결제망에 따라 갈린다.
#
#   f_own = f                                   (독자망: 대행 수수료 없음)
#   f_bc  = f - c                               (BC 대행망: 대행 수수료율 c 차감)
#   f_eff = a*f_own + (1-a)*f_bc = f - (1-a)*c  (a = 독자망 결제 비중)
#   g*    = r / (f_eff - r)                     (수식 자체는 04_economics.g_star 그대로)
#
# 핵심 질문: "혜택률 r이 손익분기를 넘으려면 독자망 전환율 a가 몇 %여야 하는가"
#   - 할인이 이론상 가능해지는 문턱:  f_eff >  r        -> a > 1 - (f - r)/c
#   - threshold(g*<=t)를 넘는 문턱:  f_eff >= r(1+t)/t -> a >= 1 - (f - r(1+t)/t)/c
#
# 주의(CLAUDE.md 규칙 1·3) — 이 스크립트가 기대는 가정 3개를 표 컬럼명에도 전부 남긴다:
#   (가정1) a의 실측 대리지표. 공개 수치는 "독자카드 매출 비중"(2026 1Q 37.8%)이지
#           "독자가맹점에서 발생한 결제 비중"이 아니다. 후자가 모델이 필요로 하는 값이고,
#           독자카드 결제도 매입이 BC망을 타면 대행 수수료가 붙을 수 있으므로 실측 대리값은
#           실제 a의 상한일 가능성이 높다 -> 실측 a는 "현재 위치 표시"로만 쓰고 결론은
#           a 전 구간 스윕으로 낸다.
#   (가정2) c(대행 수수료율)는 공개되지 않는다. 보도된 프로세싱 수수료 총액(연 800~1000억원)을
#           결제액으로 나눠 역산하는데, **이 역산값은 상한도 하한도 아니다** — 방향이 반대인
#           편향 두 개가 섞여 있고 서로 상쇄되지 않는다:
#             (하향 편향) 분자 800~1000억원은 대행 수수료가 고정비로 나가던 시절, 즉 a가 0에
#               가깝던 때의 부담인데 분모 51조4984억원은 2025년 실적이다. 분자는 전환 전,
#               분모는 전환 후를 쓴 셈이라 당시 결제액이 지금보다 작았다면 실제 c는 더 크다.
#             (상향 편향) 대행 수수료는 법인·체크를 포함한 매입 전체에 붙는데 분모는 개인
#               신용판매만이다. 분모를 작게 잡았으므로 c는 실제보다 크게 나온다.
#           따라서 c는 "상하한을 특정할 수 없는 추정치"로 다루고, 하나의 값에 결론을 걸지
#           않고 격자로 흔든다. 좁히려면 사업보고서 주석의 수수료비용 세부 내역과 같은
#           사업연도의 매입액(신용판매+현금서비스)이 필요하다 — 미수행(후속 최우선 과제).
#   (가정3) 업종별 결제액 구성은 본 프로젝트의 경기도 카드 소비 데이터에서 온 것이지
#           우리카드 자사 결제 구성이 아니다. 우선순위 계산(계산3·4)의 금액은 전부
#           "경기도 데이터 구성을 대입했을 때"라는 조건부 수치다.

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CONFIG_PATH = Path("config/assumptions.yaml")
SEGMENT_INDUSTRY_PATH = Path("outputs/tables/segment_industry_amt.csv")
ECONOMICS_PATH = Path("src/04_economics.py")

OUT_C_IMPLIED = Path("outputs/tables/woori_c_implied.csv")
OUT_FEFF = Path("outputs/tables/woori_feff_gstar.csv")
OUT_FRONTIER = Path("outputs/tables/woori_alpha_frontier.csv")
OUT_SEGMENT = Path("outputs/tables/woori_segment_impact.csv")
OUT_PRIORITY = Path("outputs/tables/woori_industry_priority.csv")
OUT_RANK = Path("outputs/tables/woori_leak_rank_robustness.csv")
OUT_THRESHOLD = Path("outputs/tables/woori_threshold_sweep.csv")
OUT_FIGURE = Path("outputs/figures/woori_alpha_frontier.png")

# dataviz 스킬 palette.md categorical 슬롯 (04_economics.py와 동일 팔레트)
CURVE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
Y_CAP = 3.0  # g* 300%에서 상단 절단 (04_economics.py와 동일)

# 계산5 부분 매핑 시나리오 대상 — L4에서 판정을 뒤집었던 영세구간(0.40%) 업종
# (docs/limitations.md 4번, outputs/tables/sensitivity.csv 1번)
LOW_FEE_INDUSTRIES = ["음식", "생활서비스", "여가/오락"]


def load_g_star():
    """src/04_economics.py의 g_star()를 그대로 쓴다 (숫자 접두사 때문에 importlib 경유)."""
    spec = importlib.util.spec_from_file_location("economics_04", ECONOMICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.g_star


g_star = load_g_star()


def f_effective(f: float, alpha: float, c: float) -> float:
    """이원 결제망 실효 수수료율. a=1이면 f(독자망 전량), a=0이면 f-c(BC망 전량)."""
    return f - (1 - alpha) * c


def required_alpha(f: float, r_target: float, c: float) -> float:
    """f_eff >= r_target 을 만족하는 최소 독자망 비중 a.
    반환값이 <=0 이면 BC망 100%에서도 이미 충족, >1 이면 독자망 100%로도 불가.
    (클리핑하지 않고 원값을 그대로 돌려준다 — '얼마나 모자라는가'가 정보다.)"""
    if c <= 0:
        return np.nan
    return 1 - (f - r_target) / c


def dgstar_dalpha(f: float, r: float, alpha: float, c: float) -> float:
    """현재 a에서 a를 1%p 올릴 때 g*가 얼마나 변하는가 (음수면 개선).
    g* = r/(f_eff - r), df_eff/da = c  ->  dg*/da = -r*c/(f_eff - r)^2, 여기에 1%p=0.01 적용."""
    fe = f_effective(f, alpha, c)
    if fe <= r:
        return np.nan  # 성립 불가 구간에서는 g*가 무한대라 기울기가 정의되지 않는다
    return -r * c / (fe - r) ** 2 * 0.01


def verdict(a_required: float, alpha_now: float) -> str:
    if not np.isfinite(a_required):
        return "판정불가(c=0)"
    if a_required <= 0:
        return "결제망 무관하게 이미 성립"
    if a_required > 1:
        return "독자망 100%로도 불가"
    if a_required <= alpha_now:
        return "현재 독자망 비중으로 성립"
    return "독자망 추가 전환 필요"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_c_implied(w: dict, c_grid: list) -> pd.DataFrame:
    """계산0. c는 공개되지 않으므로 두 방향으로 다룬다.
    (a) 역산: 보도된 프로세싱 수수료 총액 / 결제액 -> c 구간. 단 이 값은 상한도 하한도
        아니다 — 분자(전환 전 시점)와 분모(2025년 실적)의 시점이 달라 c를 작게 만드는
        편향과, 분모가 개인 신용판매만이라 c를 크게 만드는 편향이 섞여 상쇄되지 않는다.
    (b) 스윕 격자의 각 c에 대해 '그 c가 성립하려면 연간 결제액이 얼마여야 하는가'를 역으로
        제시해, 독자가 51조4984억원(개인 신용판매 실측)과 비교해 타당성을 직접 판단하게 한다."""
    v_personal = w["personal_credit_sales_2025_krw"]
    fee_min = w["processing_fee_annual_krw_min"]
    fee_max = w["processing_fee_annual_krw_max"]

    rows = []
    for label, fee in [
        ("프로세싱수수료 800억(보도 하한)", fee_min),
        ("프로세싱수수료 1000억(보도 상한)", fee_max),
    ]:
        rows.append(
            {
                "구분": "역산(c = 프로세싱수수료 / 개인신용판매)",
                "기준": label,
                "c(대행수수료율)": fee / v_personal,
                "함축_연간결제액(조원)": round(v_personal / 1e12, 1),
                "개인신용판매(51.5조) 대비 배수": 1.0,
                "비고": (
                    "상한 아님 — 편향 2개가 상쇄되지 않음: "
                    "[분자=전환 전 시점 / 분모=2025년 실적]이라 c를 과소평가할 수 있고, "
                    "[분모가 개인 신용판매만(법인·체크 제외)]이라 c를 과대평가한다"
                ),
            }
        )
    for c in c_grid:
        implied_v_min = fee_min / c
        implied_v_max = fee_max / c
        implied_mid = (implied_v_min + implied_v_max) / 2
        rows.append(
            {
                "구분": "스윕 격자(c 가정 -> 함축 결제액 역제시)",
                "기준": f"c={c:.2%}",
                "c(대행수수료율)": c,
                "함축_연간결제액(조원)": round(implied_mid / 1e12, 1),
                "개인신용판매(51.5조) 대비 배수": round(implied_mid / v_personal, 2),
                "비고": (
                    f"프로세싱수수료 800~1000억원을 c로 나누면 연 결제액 "
                    f"{implied_v_min / 1e12:.1f}~{implied_v_max / 1e12:.1f}조원이 필요"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_feff_gstar(fee_rates: dict, c_grid: list, alpha_grid: np.ndarray, r_values: np.ndarray,
                     threshold: float) -> pd.DataFrame:
    """계산1. 업종 x c x a x r 전 조합의 f_eff와 g*. 명목 f 기준 g*(=기존 모델 결과)를
    나란히 둬서 '결제망을 넣으면 판정이 어떻게 달라지는가'를 한 표에서 대조할 수 있게 한다."""
    rows = []
    for industry, f in fee_rates.items():
        for c in c_grid:
            for alpha in alpha_grid:
                fe = f_effective(f, alpha, c)
                for r in r_values:
                    g_nominal = g_star(f, r)
                    g_eff = g_star(fe, r) if fe > 0 else np.inf
                    rows.append(
                        {
                            "업종": industry,
                            "f(명목 가맹점수수료율)": f,
                            "c(대행수수료율, 가정)": c,
                            "a(독자망결제비중)": round(float(alpha), 4),
                            "f_eff(실효수수료율)": fe,
                            "r(혜택률)": r,
                            "g*_명목f기준(기존모델)": g_nominal,
                            "g*_f_eff기준(이원결제망)": g_eff,
                            "성립_유한(f_eff>r)": bool(np.isfinite(g_eff)),
                            "성립_threshold": bool(np.isfinite(g_eff) and g_eff <= threshold),
                        }
                    )
    return pd.DataFrame(rows)


def build_alpha_frontier(fee_rates: dict, c_grid: list, r_values: np.ndarray, threshold: float,
                         alpha_now: float, alpha_prior: float) -> pd.DataFrame:
    """계산2 — 이 스크립트의 핵심 산출물.
    업종·혜택률·c마다 '독자망 전환율이 몇 %여야 성립하는가'(a*)를 역산하고, 실측 대리값
    (2026 1Q 37.8%)과의 거리를 %p로 남긴다. a*가 1을 넘으면 독자망 100%로도 불가다."""
    rows = []
    for industry, f in fee_rates.items():
        for c in c_grid:
            for r in r_values:
                a_finite = required_alpha(f, r, c)
                a_thr = required_alpha(f, r * (1 + threshold) / threshold, c)
                rows.append(
                    {
                        "업종": industry,
                        "f(명목 가맹점수수료율)": f,
                        "c(대행수수료율, 가정)": c,
                        "r(혜택률)": r,
                        "a*_할인가능(f_eff>r)": a_finite,
                        "a*_threshold충족": a_thr,
                        "판정_할인가능": verdict(a_finite, alpha_now),
                        "판정_threshold": verdict(a_thr, alpha_now),
                        "실측a(37.8%)와의 격차_할인가능(%p)": (a_finite - alpha_now) * 100,
                        "실측a(37.8%)와의 격차_threshold(%p)": (a_thr - alpha_now) * 100,
                        "전년a(16.2%)에서_할인가능했나": bool(a_finite <= alpha_prior),
                        "현재a(37.8%)에서_할인가능한가": bool(a_finite <= alpha_now),
                    }
                )
    return pd.DataFrame(rows)


def build_segment_impact(seg_industry: pd.DataFrame, fee_rates: dict, c_grid: list,
                         alpha_now: float) -> pd.DataFrame:
    """계산3. 세그먼트별 지출 구성으로 가중한 실효 수수료율과, 현재 a에서 새어나가는
    대행수수료 금액. 업종 구성이 세그먼트마다 다르므로 '독자망 전환의 수혜 세그먼트'가
    갈린다 — g*가 세그먼트와 무관했던 기존 모델(H6)에 세그먼트 축이 다시 생기는 지점이다.
    주의(가정3): 금액은 경기도 카드 소비 데이터 구성을 대입한 조건부 수치다."""
    rows = []
    for c in c_grid:
        for seg, grp in seg_industry.groupby("segment"):
            grp = grp[grp["card_tpbuz_nm_1"].isin(fee_rates)]
            amt = grp["월추정매출액"].to_numpy()
            f_vec = grp["card_tpbuz_nm_1"].map(fee_rates).to_numpy()
            total = amt.sum()
            f_weighted = float((f_vec * amt).sum() / total)
            fe_now = f_effective(f_weighted, alpha_now, c)
            leak_now = c * (1 - alpha_now) * total
            rows.append(
                {
                    "세그먼트": seg,
                    "c(대행수수료율, 가정)": c,
                    "월추정매출액합(경기도데이터 대입)": round(total),
                    "지출가중 f(명목)": f_weighted,
                    "f_eff(a=0, BC망전량)": f_weighted - c,
                    "f_eff(a=37.8%, 실측대리)": fe_now,
                    "f_eff(a=100%, 독자망전량)": f_weighted,
                    "현재a에서 월 대행수수료 유출액(원)": round(leak_now),
                    "a를 100%로 올릴 때 월 절감액(원)": round(leak_now),
                    "명목f 대비 실효f 잠식률(현재a)": (f_weighted - fe_now) / f_weighted,
                }
            )
    return pd.DataFrame(rows)


def build_leak_rank_robustness(seg_industry: pd.DataFrame, fee_rates: dict, c_grid: list,
                               fee_tiers: list, alpha_now: float) -> pd.DataFrame:
    """계산5. 계산3의 세그먼트 잠식률 순위가 어느 가정에 견고한지 확인한다.

    잠식률 = (1-a)·c / f_seg 이므로 c는 모든 세그먼트에 똑같이 곱해지는 상수다 ->
    c를 어떻게 잡아도 순위는 f_seg(세그먼트 지출가중 수수료율)의 역순으로 고정된다.
    반면 f_seg는 업종->수수료구간 매핑에서 나오므로 매핑을 흔들면 바뀔 수 있다.
    c 격자 x 매핑 시나리오(05_report.py fee_sensitivity와 같은 '한 단계 위/아래' 규칙)
    전부에 대해 순위를 실제로 계산해 확인한다 — 수식으로 자명한 쪽도 계산으로 닫는다."""
    scenarios = {}
    for label, shift in [("비관(한 단계 아래)", -1), ("기본(현재 가정)", 0), ("낙관(한 단계 위)", 1)]:
        mapped = {}
        for industry, f in fee_rates.items():
            idx = fee_tiers.index(f)
            mapped[industry] = fee_tiers[min(max(idx + shift, 0), len(fee_tiers) - 1)]
        scenarios[label] = mapped
    # 위 세 시나리오는 9개 업종을 한꺼번에 같은 방향으로 옮기므로 f_seg의 상대 순서가
    # 보존되기 쉽다 — 순위 견고성의 실제 시험대가 아니다. L4에서 결론을 뒤집었던 바로 그
    # 가정(저수수료 3개 업종만 한 단계 위로, docs/limitations.md 4번)을 따로 넣는다.
    # cluster1은 이 3개 업종 지출 비중이 가장 높으므로 f_seg가 가장 크게 올라간다.
    partial = dict(fee_rates)
    for industry in LOW_FEE_INDUSTRIES:
        idx = fee_tiers.index(fee_rates[industry])
        partial[industry] = fee_tiers[min(idx + 1, len(fee_tiers) - 1)]
    scenarios["부분(저수수료 3개 업종만 한 단계 위)"] = partial

    rows = []
    for scenario, mapping in scenarios.items():
        for c in c_grid:
            for seg, grp in seg_industry.groupby("segment"):
                grp = grp[grp["card_tpbuz_nm_1"].isin(mapping)]
                amt = grp["월추정매출액"].to_numpy()
                f_vec = grp["card_tpbuz_nm_1"].map(mapping).to_numpy()
                f_seg = float((f_vec * amt).sum() / amt.sum())
                rows.append(
                    {
                        "매핑시나리오": scenario,
                        "c(대행수수료율, 가정)": c,
                        "세그먼트": seg,
                        "지출가중 f_seg": f_seg,
                        "잠식률": (1 - alpha_now) * c / f_seg,
                    }
                )
    df = pd.DataFrame(rows)
    df["잠식률순위(1=가장큼)"] = (
        df.groupby(["매핑시나리오", "c(대행수수료율, 가정)"])["잠식률"].rank(ascending=False).astype(int)
    )
    order = (
        df.sort_values("잠식률순위(1=가장큼)")
        .groupby(["매핑시나리오", "c(대행수수료율, 가정)"])["세그먼트"]
        .apply(lambda s: " > ".join(s))
        .rename("순위")
        .reset_index()
    )
    return df.merge(order, on=["매핑시나리오", "c(대행수수료율, 가정)"])


def build_threshold_sweep(fee_rates: dict, c_grid: list, r_ref: float, alpha_now: float,
                          threshold_grid: np.ndarray, alpha_base: float = 1.0) -> pd.DataFrame:
    """계산6. 대표 결과("대행 수수료 항을 넣으면 의료/건강이 42.9%->51.8%로 threshold를
    넘는다")는 두 층으로 되어 있다:
      (1) 잠식의 크기 — 명목 f가 얼마나 깎이는가. threshold와 무관하게 성립하고 c에만 걸린다.
      (2) 판정의 뒤집힘 — 그 크기가 threshold 선을 넘는가. threshold 가정에 추가로 걸린다.
    threshold(=50%)는 사람이 정한 판단치이므로(config/assumptions.yaml 주석), 그 선을
    45~60%로 흔들었을 때 (2)가 유지되는지 확인한다. 흔들어도 뒤집히는 업종이 남아 있으면
    대표 결과가 threshold 값 하나에 얹혀 있지 않다는 뜻이고, 특정 구간에서만 뒤집히면
    그 구간을 명시해야 한다.

    alpha_base는 '항을 넣기 전' 상태다 — 기존 모델은 대행 수수료를 0으로 두었으므로
    a=1(f_eff=f)이 그 상태에 해당한다(H19)."""
    rows = []
    for c in c_grid:
        for t in threshold_grid:
            flipped = []
            for industry, f in fee_rates.items():
                g_before = g_star(f_effective(f, alpha_base, c), r_ref)  # 항 넣기 전 = 명목 f
                g_after = g_star(f_effective(f, alpha_now, c), r_ref)
                ok_before = bool(np.isfinite(g_before) and g_before <= t)
                ok_after = bool(np.isfinite(g_after) and g_after <= t)
                if ok_before and not ok_after:
                    flipped.append(industry)
            rows.append(
                {
                    "c(대행수수료율, 가정)": c,
                    "threshold": round(float(t), 4),
                    "판정이 뒤집힌 업종 수": len(flipped),
                    "판정이 뒤집힌 업종": "; ".join(flipped) if flipped else "(없음)",
                }
            )
    return pd.DataFrame(rows)


def build_industry_priority(seg_industry: pd.DataFrame, fee_rates: dict, c_grid: list,
                            r_ref: float, threshold: float, alpha_now: float) -> pd.DataFrame:
    """계산4 — "어느 업종을 먼저 독자망으로 끌어와야 혜택 설계의 여유가 생기는가".
    두 축을 같이 놓는다:
      (금액축) 현재 a에서 그 업종에서 새어나가는 대행수수료 = c*(1-a)*매출
      (판정축) 그 업종의 혜택(r_ref)이 성립하기까지 남은 전환율 격차 a* - a
    금액이 커도 이미 성립한 업종은 우선순위가 아니고, 격차가 작아도 금액이 미미하면
    전환 노력 대비 효과가 없다. 두 축을 하나의 점수로 합치지 않고 나란히 남겨 사람이
    판단하게 한다 (CLAUDE.md 규칙 4)."""
    amt_by_industry = seg_industry.groupby("card_tpbuz_nm_1")["월추정매출액"].sum()
    rows = []
    for c in c_grid:
        for industry, f in fee_rates.items():
            amt = float(amt_by_industry.get(industry, 0.0))
            a_finite = required_alpha(f, r_ref, c)
            a_thr = required_alpha(f, r_ref * (1 + threshold) / threshold, c)
            fe_now = f_effective(f, alpha_now, c)
            rows.append(
                {
                    "업종": industry,
                    "c(대행수수료율, 가정)": c,
                    "r(기준혜택률)": r_ref,
                    "월추정매출액(경기도데이터 대입)": round(amt),
                    "현재a에서 월 대행수수료 유출액(원)": round(c * (1 - alpha_now) * amt),
                    "f(명목)": f,
                    "f_eff(현재a)": fe_now,
                    "a*_할인가능": a_finite,
                    "남은격차_할인가능(%p)": (a_finite - alpha_now) * 100,
                    "a*_threshold충족": a_thr,
                    "남은격차_threshold(%p)": (a_thr - alpha_now) * 100,
                    "g*(현재a)": g_star(fe_now, r_ref),
                    "a 1%p 전환당 g* 변화(%p)": dgstar_dalpha(f, r_ref, alpha_now, c) * 100,
                    "판정_할인가능": verdict(a_finite, alpha_now),
                    "판정_threshold": verdict(a_thr, alpha_now),
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["c(대행수수료율, 가정)", "남은격차_할인가능(%p)", "현재a에서 월 대행수수료 유출액(원)"],
        ascending=[True, True, False],
    )


def plot_alpha_frontier(fee_rates: dict, c_ref: float, r_ref: float, threshold: float,
                        alpha_now: float, alpha_prior: float) -> None:
    """a(독자망 결제 비중) 대비 g* 곡선. 업종을 f로 묶고(같은 f면 같은 곡선), 실측 대리값
    16.2%->37.8% 구간을 음영으로 표시해 '지금 어디쯤 와 있는가'를 그림에서 바로 읽게 한다."""
    tiers: dict[float, list[str]] = {}
    for industry, f in fee_rates.items():
        tiers.setdefault(f, []).append(industry)

    alphas = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (f, industries) in enumerate(sorted(tiers.items())):
        g_values = np.array([g_star(f_effective(f, a, c_ref), r_ref) for a in alphas])
        color = CURVE_COLORS[i % len(CURVE_COLORS)]
        label = f"f={f:.2%} ({'/'.join(industries)})"
        visible = np.isfinite(g_values) & (g_values < Y_CAP)
        ax.plot(alphas[visible] * 100, g_values[visible] * 100, color=color, linewidth=2, label=label)
        capped = ~visible
        if capped.any():
            ax.plot(alphas[capped] * 100, np.full(capped.sum(), Y_CAP * 100),
                    color=color, linewidth=2, linestyle=":")

        # 임계 a를 점으로 찍는다 — f=0.40% 구간은 곡선 전체가 상단 절단선에 붙어 있어
        # (a=100%에서도 g*=300%) 곡선만으로는 "할인이 가능해지는 지점"이 보이지 않는다.
        a_fin = required_alpha(f, r_ref, c_ref)
        a_thr = required_alpha(f, r_ref * (1 + threshold) / threshold, c_ref)
        if 0 < a_fin < 1:
            ax.plot([a_fin * 100], [Y_CAP * 100], marker="o", color=color, markersize=6, zorder=5)
            ax.annotate(f"a*={a_fin:.1%}\n여기서부터 할인 가능\n(단, g*는 여전히 300%)",
                        (a_fin * 100, Y_CAP * 100), textcoords="offset points", xytext=(8, -34),
                        fontsize=8, color=color)
        if 0 < a_thr < 1:
            ax.plot([a_thr * 100], [threshold * 100], marker="o", color=color, markersize=6, zorder=5)
            ax.annotate(f"a*={a_thr:.1%}\n여기서부터 threshold 충족",
                        (a_thr * 100, threshold * 100), textcoords="offset points", xytext=(8, 12),
                        fontsize=8, color=color)

    ax.axhline(threshold * 100, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.text(100, threshold * 100, f" g*={threshold:.0%} 임계선", color=INK_MUTED,
            fontsize=9, va="bottom", ha="right")
    ax.axvspan(alpha_prior * 100, alpha_now * 100, color=INK_MUTED, alpha=0.12)
    ax.axvline(alpha_now * 100, color=INK_PRIMARY, linewidth=1)
    # 임계 a 주석(오른쪽 위)과 겹치지 않도록 실측선 라벨은 선 왼쪽 아래에 붙인다.
    ax.text(alpha_now * 100 - 2, Y_CAP * 100 * 0.72,
            f"실측 대리값 {alpha_prior:.1%} → {alpha_now:.1%}\n(독자카드 매출 비중, 2025→2026 1Q)",
            color=INK_PRIMARY, fontsize=8, va="top", ha="right")

    ax.set_xlabel("독자망 결제 비중 a (%)", color=INK_PRIMARY, fontsize=10)
    ax.set_ylabel("필요 증분 이용률 g* (%, 300%에서 상단 절단)", color=INK_PRIMARY, fontsize=10)
    ax.set_title(
        f"독자망 전환율 대비 손익분기 필요 증분 이용률 (r={r_ref:.1%}, c={c_ref:.2%} 가정)",
        color=INK_PRIMARY, fontsize=12,
    )
    ax.set_xlim(0, 100)
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


def main() -> None:
    if not SEGMENT_INDUSTRY_PATH.exists():
        raise SystemExit(f"[중단] {SEGMENT_INDUSTRY_PATH} 없음. 먼저 03_segment.py 실행할 것.")
    cfg = load_config()
    w = cfg.get("woori_dual_network")
    if w is None:
        raise SystemExit("[중단] config/assumptions.yaml에 woori_dual_network 블록이 없다.")

    fee_rates = cfg["merchant_fee_rate"]
    threshold = cfg["g_star_threshold"]
    sweep = cfg["benefit_rate_sweep"]
    r_values = np.round(np.arange(sweep["min"], sweep["max"] + sweep["step"], sweep["step"]), 4)
    # 기준 혜택률은 04_economics.py의 상품 3안과 같은 0.3%(스윕 최솟값)를 쓴다 —
    # "가장 작은 혜택률에서도 성립하는가"가 이 프로젝트의 관문이었다.
    r_ref = sweep["min"]

    alpha_now = w["own_network_share_observed"]
    alpha_prior = w["own_network_share_prior"]
    c_grid = list(w["agency_fee_rate_sweep"])
    # 역산값(프로세싱수수료 보도 상한 / 개인신용판매)도 격자에 넣어 기준 c로 쓴다.
    # 주의: '보도 상한'은 분자(800~1000억원)의 상한이라는 뜻이지 c의 상한이 아니다 —
    # c 자체는 상하한을 특정할 수 없다(상단 주석 가정2).
    c_ref = round(w["processing_fee_annual_krw_max"] / w["personal_credit_sales_2025_krw"], 6)
    if c_ref not in c_grid:
        c_grid = sorted(c_grid + [c_ref])

    alpha_grid = np.round(np.arange(0.0, 1.0 + w["own_share_step"], w["own_share_step"]), 4)
    alpha_grid = np.unique(np.concatenate([alpha_grid, [alpha_prior, alpha_now]]))

    seg_industry = pd.read_csv(SEGMENT_INDUSTRY_PATH)
    OUT_C_IMPLIED.parent.mkdir(parents=True, exist_ok=True)

    print("=== 계산0. 대행수수료율 c 역산 (c는 비공개 -> 역산 + 스윕으로만 다룬다) ===")
    c_implied = build_c_implied(w, list(w["agency_fee_rate_sweep"]))
    c_implied.to_csv(OUT_C_IMPLIED, index=False, encoding="utf-8-sig")
    print(c_implied.to_string(index=False))
    print(
        f"\n기준 c(역산 = 프로세싱수수료 1000억 / 개인신용판매 51조4984억) = {c_ref:.4%}"
        "\n  -> 이 값은 c의 상한이 아니다. 시점 불일치(분자=전환 전, 분모=2025년 실적)는 c를 "
        "과소평가하는 쪽, 분모를 개인 신용판매로만 잡은 것은 c를 과대평가하는 쪽이며 두 편향이 "
        "상쇄되지 않는다 — 아래 결과는 전부 c 격자와 함께 읽어야 한다."
    )
    print(f"[완료] {OUT_C_IMPLIED}")

    print("\n=== 계산1. 업종 x c x a x r -> f_eff, g* ===")
    feff = build_feff_gstar(fee_rates, c_grid, alpha_grid, r_values, threshold)
    feff.to_csv(OUT_FEFF, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_FEFF} ({len(feff):,} rows)")
    snap = feff[
        np.isclose(feff["c(대행수수료율, 가정)"], c_ref)
        & np.isclose(feff["r(혜택률)"], r_ref)
        & feff["a(독자망결제비중)"].isin([0.0, round(alpha_prior, 4), round(alpha_now, 4), 1.0])
    ]
    pivot = snap.pivot(index="업종", columns="a(독자망결제비중)", values="g*_f_eff기준(이원결제망)")
    print(f"\n--- r={r_ref:.1%}, c={c_ref:.4%} 에서 a별 g* (inf = 할인 자체 불가) ---")
    print(pivot.round(3).to_string())

    print("\n=== 계산2. 성립에 필요한 독자망 전환율 a* ===")
    frontier = build_alpha_frontier(fee_rates, c_grid, r_values, threshold, alpha_now, alpha_prior)
    frontier.to_csv(OUT_FRONTIER, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_FRONTIER} ({len(frontier):,} rows)")
    key = frontier[
        np.isclose(frontier["c(대행수수료율, 가정)"], c_ref) & np.isclose(frontier["r(혜택률)"], r_ref)
    ]
    print(f"\n--- r={r_ref:.1%}, c={c_ref:.4%} 기준 업종별 a* ---")
    print(
        key[["업종", "f(명목 가맹점수수료율)", "a*_할인가능(f_eff>r)", "판정_할인가능",
             "실측a(37.8%)와의 격차_할인가능(%p)", "a*_threshold충족", "판정_threshold"]]
        .round(4).to_string(index=False)
    )
    flipped = frontier[
        np.isclose(frontier["c(대행수수료율, 가정)"], c_ref)
        & (~frontier["전년a(16.2%)에서_할인가능했나"])
        & frontier["현재a(37.8%)에서_할인가능한가"]
    ]
    print(
        f"\n실측 a가 16.2%->37.8%로 오르면서 '할인 자체 불가'에서 '가능'으로 뒤집힌 "
        f"업종x혜택률 조합(c={c_ref:.4%}): {len(flipped)}건"
    )
    if len(flipped):
        print(flipped[["업종", "r(혜택률)", "a*_할인가능(f_eff>r)"]].round(4).to_string(index=False))

    print("\n=== 계산3. 세그먼트별 독자망 전환 효과 ===")
    segment_impact = build_segment_impact(seg_industry, fee_rates, c_grid, alpha_now)
    segment_impact.to_csv(OUT_SEGMENT, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_SEGMENT}")
    print(
        segment_impact[np.isclose(segment_impact["c(대행수수료율, 가정)"], c_ref)]
        .drop(columns=["c(대행수수료율, 가정)"]).round(6).to_string(index=False)
    )

    print("\n=== 계산4. 업종별 독자망 전환 우선순위 ===")
    priority = build_industry_priority(seg_industry, fee_rates, c_grid, r_ref, threshold, alpha_now)
    priority.to_csv(OUT_PRIORITY, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_PRIORITY}")
    print(
        priority[np.isclose(priority["c(대행수수료율, 가정)"], c_ref)]
        .drop(columns=["c(대행수수료율, 가정)", "r(기준혜택률)"]).round(4).to_string(index=False)
    )

    print("\n=== 계산5. 세그먼트 잠식률 순위의 견고성 (c 격자 x 업종매핑 시나리오) ===")
    fee_tiers = sorted(cfg["fee_rate_by_revenue_tier"].values())
    rank = build_leak_rank_robustness(seg_industry, fee_rates, c_grid, fee_tiers, alpha_now)
    rank.to_csv(OUT_RANK, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_RANK}")
    summary = rank[["매핑시나리오", "c(대행수수료율, 가정)", "순위"]].drop_duplicates()
    print(summary.to_string(index=False))
    by_c = summary[summary["매핑시나리오"] == "기본(현재 가정)"]["순위"].nunique()
    by_map = summary["순위"].nunique()
    print(
        f"\nc를 {len(c_grid)}개 값으로 흔들었을 때 서로 다른 순위: {by_c}가지 "
        f"(잠식률=(1-a)*c/f_seg 에서 c는 모든 세그먼트에 곱해지는 상수이므로 순위 불변이 수식상 예상값)"
        f"\n업종매핑까지 함께 흔들었을 때 서로 다른 순위: {by_map}가지 "
        f"(전체 이동 3종 + 저수수료 3개 업종만 이동 1종)"
    )

    print("\n=== 계산6. threshold를 흔들었을 때 '항을 넣어서 뒤집힌 판정'이 유지되는가 ===")
    threshold_grid = np.round(np.arange(0.45, 0.601, 0.01), 4)
    tsweep = build_threshold_sweep(fee_rates, c_grid, r_ref, alpha_now, threshold_grid)
    tsweep.to_csv(OUT_THRESHOLD, index=False, encoding="utf-8-sig")
    print(f"[완료] {OUT_THRESHOLD}")
    print(
        tsweep[np.isclose(tsweep["c(대행수수료율, 가정)"], c_ref)]
        .drop(columns=["c(대행수수료율, 가정)"]).to_string(index=False)
    )
    base = tsweep[np.isclose(tsweep["c(대행수수료율, 가정)"], c_ref)]
    hit = base[base["판정이 뒤집힌 업종 수"] > 0]["threshold"]
    print(
        f"\nthreshold 45~60% 중 항 추가로 판정이 뒤집히는 구간: "
        + (f"{hit.min():.0%}~{hit.max():.0%} ({len(hit)}/{len(base)}개 지점)" if len(hit) else "없음")
        + f"  (c={c_ref:.4%} 기준)"
    )

    plot_alpha_frontier(fee_rates, c_ref, r_ref, threshold, alpha_now, alpha_prior)

    print(
        "\n[주의] 위 결과는 세 가지 가정 위에 있다 (스크립트 상단 주석 참고): "
        "(1) a의 실측 대리지표는 '독자카드 매출 비중'이지 '독자가맹점 결제 비중'이 아니다, "
        "(2) c는 비공개라 보도 총액을 결제액으로 나눈 역산값이다, "
        "(3) 업종별 금액 구성은 경기도 카드 소비 데이터이지 우리카드 자사 결제 구성이 아니다."
    )


if __name__ == "__main__":
    main()
