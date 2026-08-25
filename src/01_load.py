# 입력: data/raw/tbsh_gyeonggi_day_YYYYMM_시군구명.csv (경기도 카드 소비 데이터 원본)
# 출력: data/processed/01_loaded.parquet
# 목적: 원본 데이터를 읽고 CLAUDE.md 데이터 검증 체크리스트를 수행한다.
#       (결측치, 건당 평균 결제액 상식 범위, 업종/시간대 극단값, 기준 시점, 코드값 매칭)

import re
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/01_loaded.parquet")

# 실제 원본 컬럼명 (utf-8-sig로 확인, 2026-08-25):
# ta_ymd(기준년월일), cty_rgn_no(시군구코드), admi_cty_no(행정동코드),
# card_tpbuz_cd(업종코드), card_tpbuz_nm_1(업종대분류), card_tpbuz_nm_2(업종중분류),
# hour(시간대), sex(성별), age(연령코드), day(요일), amt(매출금액), cnt(매출건수)
DTYPES = {
    "ta_ymd": "int32",
    "cty_rgn_no": "int32",
    "admi_cty_no": "int32",
    "card_tpbuz_cd": "category",
    "card_tpbuz_nm_1": "category",
    "card_tpbuz_nm_2": "category",
    "hour": "int8",
    "sex": "category",
    "age": "int8",
    "day": "int8",
    "amt": "int64",
    "cnt": "int32",
}

FNAME_RE = re.compile(r"tbsh_gyeonggi_day_(\d{6})_(.+)\.csv$")


def find_raw_files() -> list[Path]:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        sys.exit(
            f"[중단] {RAW_DIR}에 원본 CSV가 없다. "
            "공공데이터포털에서 경기도 카드 소비 데이터를 받아 배치할 것."
        )
    return files


def load_raw(files: list[Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        m = FNAME_RE.search(f.name)
        yyyymm, city = m.group(1), m.group(2) if m else (None, None)
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=DTYPES)
        df["file_yyyymm"] = yyyymm
        df["file_city"] = city
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def validate(df: pd.DataFrame) -> None:
    print("=== 파일 커버리지 (월별 시군구 수) ===")
    print(df.groupby("file_yyyymm")["file_city"].nunique())

    print("\n=== 시군구코드-이름 매핑 일관성 (코드 1개당 파일명이 여러 개면 문제) ===")
    mapping = df.groupby("cty_rgn_no")["file_city"].nunique()
    inconsistent = mapping[mapping > 1]
    print("불일치 건수:", len(inconsistent))
    if len(inconsistent):
        print(inconsistent)

    print("\n=== 결측치 비율 ===")
    print(df.isna().mean().sort_values(ascending=False))

    avg_pay = df["amt"] / df["cnt"].replace(0, pd.NA)
    print("\n=== 건당 평균 결제액(amt/cnt) 분포 ===")
    print(avg_pay.describe())

    print("\n=== 데이터 기준 시점 범위 (ta_ymd) ===")
    print(df["ta_ymd"].min(), "~", df["ta_ymd"].max())

    print("\n=== 코드값 유니크 목록 (규격서 대조용) ===")
    for c in ["card_tpbuz_cd", "hour", "sex", "age", "day"]:
        print(f"{c}: {sorted(df[c].astype(str).unique().tolist())}")

    print("\n=== amt/cnt 극단값 ===")
    print("amt 0 이하:", (df["amt"] <= 0).sum(), " / cnt 0 이하:", (df["cnt"] <= 0).sum())
    print("amt top5:")
    print(df.nlargest(5, "amt")[["file_yyyymm", "file_city", "card_tpbuz_nm_1", "amt", "cnt"]])


def main() -> None:
    files = find_raw_files()
    df = load_raw(files)
    validate(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n[완료] {OUT_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
