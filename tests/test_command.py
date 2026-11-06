"""음성 명령 파싱 — 명령과 대상의 분리.

앱은 받아쓴 한국어 문장을 그대로 보낸다. parse_command가 여기서 '포커스/다음/
가까운 거' 같은 명령을 걷어내고 남은 말만 검출 대상으로 넘긴다.
"""

from __future__ import annotations

import pytest

from pysical.perception.command import parse_command


# --- 대상만 있는 발화 ---


def test_target_only():
    intent = parse_command("컵 찾아줘")
    assert intent.targets == ["cup"]
    assert intent.focus is None
    assert intent.select is None


def test_multiple_targets():
    assert parse_command("컵과 펜").targets == ["cup", "pen"]


def test_conjunction_does_not_split_object_name():
    """'사과'의 '과'가 접속사로 오인돼 잘리면 안 된다."""
    assert parse_command("사과와 컵").targets == ["apple", "cup"]


# --- 교체: "A 말고 B" ---


@pytest.mark.parametrize(
    "utterance",
    ["컵 말고 펜", "컵말고 펜", "컵 대신 펜", "컵 빼고 펜 찾아줘", "컵 아니고 펜"],
)
def test_replacement_keeps_only_the_second(utterance):
    """'컵 말고 펜'이 둘 다 찾으면 정확히 반대로 동작한다(회귀 방지)."""
    assert parse_command(utterance).targets == ["pen"]


def test_replacement_with_trailing_verb():
    assert parse_command("컵 말고 펜을 잡아줘").targets == ["pen"]


def test_replacement_combines_with_commands():
    intent = parse_command("컵 말고 마우스 가까운 거")
    assert intent.targets == ["mouse"]
    assert intent.select == "nearest"


def test_conjunction_is_not_replacement():
    """'컵과 펜'은 교체가 아니라 둘 다 찾는 것이다."""
    assert parse_command("컵과 펜").targets == ["cup", "pen"]


# --- 포커스 on/off ---


@pytest.mark.parametrize("utterance", ["포커스", "확대해줘", "포커스 켜"])
def test_focus_on(utterance):
    assert parse_command(utterance).focus is True


@pytest.mark.parametrize(
    "utterance",
    ["포커스 꺼", "포커스꺼", "포커스 그만", "포커스를 하기 싫어", "포커스 하지마"],
)
def test_focus_off(utterance):
    assert parse_command(utterance).focus is False


def test_focus_word_is_not_a_detection_target():
    """'포커스'를 물체 이름으로 넘기면 안 된다(그러면 포커스를 찾으려 든다)."""
    assert parse_command("포커스").targets == []


# --- 어순 자유도: 포커스가 대상 앞에 와도 대상이 살아 있어야 한다 ---


@pytest.mark.parametrize("utterance", ["포커스 컵", "컵 포커스", "컵을 포커스 해줘"])
def test_focus_with_target_any_order(utterance):
    intent = parse_command(utterance)
    assert intent.targets == ["cup"]
    assert intent.focus is True


def test_object_name_containing_negation_letter_is_not_off():
    """'마우스'의 '마'가 부정어로 오인되면 포커스가 꺼진다(회귀 방지)."""
    intent = parse_command("포커스 마우스")
    assert intent.targets == ["mouse"]
    assert intent.focus is True


# --- 인스턴스 전환 ---


@pytest.mark.parametrize("utterance", ["다음", "다른 거", "딴거"])
def test_select_next(utterance):
    assert parse_command(utterance).select == "next"


@pytest.mark.parametrize("utterance", ["가까운 거", "제일 가까운", "최근접"])
def test_select_nearest(utterance):
    assert parse_command(utterance).select == "nearest"


# --- 복합 발화 ---


def test_target_focus_and_select_together():
    intent = parse_command("펜 포커스 가까운 거")
    assert intent.targets == ["pen"]
    assert intent.focus is True
    assert intent.select == "nearest"


def test_target_with_focus_off():
    intent = parse_command("컵 포커스 꺼")
    assert intent.targets == ["cup"]
    assert intent.focus is False


# --- 빈 발화 ---


@pytest.mark.parametrize("utterance", ["", "   "])
def test_empty_is_noop(utterance):
    assert parse_command(utterance).is_noop()
