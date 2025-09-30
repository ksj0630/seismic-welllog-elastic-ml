import os
import re
import unicodedata

# ====== 설정 ======
DIR = r"./"   # ← 스크린샷의 폴더 경로로 교체
DRY_RUN = False                # True: 미리보기, False: 실제 이름 변경
PAD_PREFIX = 4                    # 숫자 프리픽스 자리수 (예: 3 → 086_)
# 유지할(진짜) 확장자 목록: 필요하면 추가
KEEP_EXTENSIONS = {
    ".py", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".png", ".jpg", ".jpeg", ".pdf",
    ".npy", ".npz", ".pkl", ".pt", ".h5", ".ckpt", ".bin",
    ".log"
}
# ===================

def normalize_name(name: str) -> str:
    """
    파일/폴더 이름을 깔끔한 스네이크케이스로 정리.
    - 확장자는 지정 목록일 때만 유지
    - 시작 숫자 프리픽스는 0패딩
    """
    # 디렉토리/확장자 분리
    stem, ext = os.path.splitext(name)
    # 확장자가 '진짜'가 아니면 확장자 제거 처리(중간 점(.)을 구분자로 본다)
    if ext.lower() not in KEEP_EXTENSIONS:
        stem, ext = name, ""

    # 유니코드 정규화(한/영 혼용, 전각/반각 등)
    s = unicodedata.normalize("NFKC", stem).strip()

    # 구분자 통일: 영숫자/밑줄 외는 전부 '_' 로
    # (점은 이미 ext 분리에서 빠졌으니 stem 내부 점도 '_'로)
    s = re.sub(r"[^\w]+", "_", s)

    # 소문자 통일
    s = s.lower()

    # 맨 앞 숫자 프리픽스가 있으면 0패딩
    m = re.match(r"^(\d+)_+(.*)$", s)
    if m:
        num = m.group(1).zfill(PAD_PREFIX)
        rest = m.group(2)
        s = f"{num}_{rest}"

    # 연속 밑줄 정리 및 양끝 밑줄 제거
    s = re.sub(r"_+", "_", s).strip("_")

    # 빈 이름 방지
    if not s:
        s = "unnamed"

    return s + ext.lower()

def unique_path(dst_dir: str, fname: str) -> str:
    """dst_dir 안에서 fname 충돌 시 _2, _3… 붙여 고유 경로 반환"""
    base, ext = os.path.splitext(fname)
    cand = fname
    i = 2
    while os.path.exists(os.path.join(dst_dir, cand)):
        cand = f"{base}_{i}{ext}"
        i += 1
    return cand

def main():
    entries = sorted(os.listdir(DIR))
    if not entries:
        print("⚠️ 대상 폴더가 비어 있습니다.")
        return

    print(f"📂 Target directory: {DIR}")
    print(f"🔎 DRY_RUN = {DRY_RUN}\n")

    changes = []
    for old in entries:
        old_path = os.path.join(DIR, old)
        new_name = normalize_name(old)
        if new_name == old:
            continue
        # 충돌 처리
        new_name = unique_path(DIR, new_name)
        new_path = os.path.join(DIR, new_name)
        changes.append((old_path, new_path))

    if not changes:
        print("✅ 변경할 파일명이 없습니다. (이미 깔끔한 상태)")
        return

    # 미리보기
    print("=== Rename preview ===")
    for old_path, new_path in changes:
        print(f"{os.path.basename(old_path)}  ->  {os.path.basename(new_path)}")
    print("======================\n")

    if DRY_RUN:
        print("📝 DRY_RUN 모드입니다. 실제 변경은 하지 않았습니다.")
        print("👉 이름을 실제로 바꾸려면 DRY_RUN=False 로 설정 후 다시 실행하세요.")
        return

    # 실제 변경
    renamed, failed = 0, 0
    for old_path, new_path in changes:
        try:
            os.rename(old_path, new_path)
            renamed += 1
        except Exception as e:
            print(f"❌ 실패: {os.path.basename(old_path)} → {os.path.basename(new_path)} ({e})")
            failed += 1

    print(f"\n✅ 완료: {renamed}개 변경, ❗실패: {failed}개")

if __name__ == "__main__":
    main()
