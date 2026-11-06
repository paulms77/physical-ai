"""음성 발화를 '명령'과 '검출 대상'으로 분리한다.

지금까지는 앱이 보낸 텍스트 전체를 검출 대상으로 해석했다. 그래서 "포커스"라고
말하면 ``포커스``라는 이름의 물체를 찾으려 들었다. 이 모듈이 그 앞단에서 명령어를
걷어내고, 남은 말만 대상 어휘로 넘긴다.

지원하는 명령
-------------
    "컵 말고 펜"                 → 컵을 버리고 펜만 (교체)
    "포커스" / "확대"            → 포커스 패널 켜기
    "포커스 꺼" / "포커스 그만"   → 포커스 패널 끄기
    "다음" / "다른 거"           → 같은 라벨의 다른 인스턴스로 전환
    "가까운 거" / "제일 가까운"   → 최근접 인스턴스로 전환

한계(의도적)
-----------
규칙 기반이라 "A 말고 B" 같은 정형 교체는 처리하지만, 복문이나 자유로운 부정
("컵은 별로고 펜이 나을 것 같아")은 해석하지 못한다. 문장 이해가 필요한 부분은
이 계층의 범위 밖이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import normalize_prompt

# 포커스 기능을 가리키는 말.
_FOCUS_WORDS = ("포커스", "확대")

# 포커스를 "끄는" 쪽 표현. **토큰 전체**가 일치할 때만 부정으로 본다.
# 부분 문자열로 검사하면 "마우스"의 "마"를 부정어로 오인한다(실측).
_FOCUS_OFF_TOKENS = frozenset({
    "꺼", "꺼줘", "꺼라", "끄기", "끄고", "끄자", "끌래",
    "싫어", "싫다", "싫은", "싫", "말고", "말아", "마",
    "그만", "그만해", "해제", "안해", "off",
    "필요없어", "필요없다", "하지마", "하지마라", "빼",
})

# 위 목록으로 못 잡는 활용형을 어간·어미로 보완한다.
_FOCUS_OFF_PREFIXES = ("끄", "꺼", "싫")
_FOCUS_OFF_SUFFIXES = ("지마", "지마라")


def _is_off_token(token: str) -> bool:
    """토큰이 '포커스를 끈다'는 뜻인지."""
    return (
        token in _FOCUS_OFF_TOKENS
        or token.startswith(_FOCUS_OFF_PREFIXES)
        or token.endswith(_FOCUS_OFF_SUFFIXES)
    )

# 인스턴스 전환 명령. 긴 표현을 먼저 검사해야 부분 일치로 잘리지 않는다.
_SELECT_NEAREST = (
    "제일 가까운 거", "가장 가까운 거", "제일 가까운", "가장 가까운",
    "가까운 거", "가까운거", "가까운 것", "가까운", "최근접",
)
_SELECT_NEXT = (
    "다른 거", "다른거", "다른 것", "딴 거", "딴거",
    "다음 거", "다음거", "다음", "바꿔", "넘겨",
)

# 명령 해석 후 잔여 문장에서 지워둘 부스러기(대상어가 아님).
_RESIDUAL_NOISE = ("패널", "화면", "보기", "모드", "기능")

# 교체 표현. "A 말고 B"는 A를 버리고 B만 남긴다. 이게 없으면 A와 B를 둘 다
# 찾아버려서 "컵 말고 펜"이 정확히 반대로 동작한다.
_REPLACE_TOKENS = ("말고", "말구", "대신", "대신에", "빼고", "아니고", "아니라", "아니야")


def _apply_replacement(text: str) -> str:
    """'A 말고 B' 꼴이면 마지막 교체어 뒤쪽만 남긴다.

    붙여 쓴 형태("컵말고 펜")도 처리한다. 교체어가 없으면 원문 그대로.
    """
    tokens = text.split()
    cut = -1
    for i, token in enumerate(tokens):
        low = token.lower()
        if low in _REPLACE_TOKENS or any(
            low.endswith(op) and len(low) > len(op) for op in _REPLACE_TOKENS
        ):
            cut = i  # 마지막 것을 기준으로 삼는다
    if cut < 0:
        return text
    return " ".join(tokens[cut + 1:])


@dataclass
class Intent:
    """한 발화에서 뽑아낸 의도.

    Attributes
    ----------
    raw:
        원본 발화(로그·디버깅용).
    targets:
        검출 대상 영어 라벨. **빈 리스트면 대상을 바꾸지 않는다**("포커스 꺼"처럼
        명령만 있는 발화와, 대상을 지운 발화를 구분하기 위함).
    focus:
        True=켜기, False=끄기, None=변경 없음.
    select:
        ``"next"`` | ``"nearest"`` | None. 포커스 대상 전환 요청.
    """

    raw: str
    targets: list[str] = field(default_factory=list)
    focus: bool | None = None
    select: str | None = None

    def is_noop(self) -> bool:
        """아무 의미도 없는 발화인지."""
        return not self.targets and self.focus is None and self.select is None


def _extract_focus(text: str) -> tuple[bool | None, str]:
    """포커스 명령을 뽑고, 그 부분만 제거한 잔여 문장을 돌려준다.

    토큰 단위로 처리하므로 어순에 자유롭다. "포커스 컵", "컵을 포커스 해줘",
    "컵 포커스" 모두 대상은 컵, 포커스는 켜기로 해석된다.
    """
    tokens = text.split()
    kept: list[str] = []
    found = False
    off = False

    for token in tokens:
        low = token.lower()
        hit = next((w for w in _FOCUS_WORDS if w in low), None)
        if hit is not None:
            # "포커스꺼"처럼 한 토큰에 붙어 있는 경우 뒤쪽 어간으로 끄기를 판정한다.
            tail = low[low.find(hit) + len(hit):]
            if tail and _is_off_token(tail):
                off = True
            found = True
            continue  # 포커스/확대 토큰 자체는 대상어가 아니다
        if found and _is_off_token(low):
            off = True
            continue  # 포커스 뒤에 온 부정어 — 명령의 일부
        kept.append(token)

    if not found:
        return None, text
    return (not off), " ".join(kept)


def _extract_select(text: str) -> tuple[str | None, str]:
    """인스턴스 전환 명령을 뽑고, 그 부분을 제거한 잔여 문장을 돌려준다."""
    lowered = text.lower()
    for phrase in _SELECT_NEAREST:
        idx = lowered.find(phrase)
        if idx >= 0:
            return "nearest", text[:idx] + " " + text[idx + len(phrase):]
    for phrase in _SELECT_NEXT:
        idx = lowered.find(phrase)
        if idx >= 0:
            return "next", text[:idx] + " " + text[idx + len(phrase):]
    return None, text


def parse_command(text: str) -> Intent:
    """발화를 명령과 대상으로 분리한다.

    Parameters
    ----------
    text:
        앱이 보낸 받아쓰기 결과(한국어 또는 영어).

    Returns
    -------
    Intent
        명령(포커스/전환)과 검출 대상 라벨.

    Examples
    --------
    >>> parse_command("컵 찾아줘").targets
    ['cup']
    >>> parse_command("포커스 꺼").focus
    False
    >>> parse_command("가까운 거").select
    'nearest'
    """
    raw = (text or "").strip()
    if not raw:
        return Intent(raw=raw)

    focus, residual = _extract_focus(raw)
    select, residual = _extract_select(residual)

    for noise in _RESIDUAL_NOISE:
        residual = residual.replace(noise, " ")

    # "컵 말고 펜" — 교체어 앞은 버린다. normalize_prompt보다 먼저 해야 한다
    # (normalize_prompt는 "말고"를 불용어로 그냥 지워버린다).
    residual = _apply_replacement(residual)

    targets = normalize_prompt(residual) if residual.strip() else []
    return Intent(raw=raw, targets=targets, focus=focus, select=select)
