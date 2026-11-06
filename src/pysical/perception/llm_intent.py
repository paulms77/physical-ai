"""로컬 LLM 의도 해석 — 대화형 발화를 Intent로 바꾼다.

**왜 필요한가**
--------------
규칙+사전은 정형 명령("컵 찾아줘", "포커스", "컵 말고 펜")에는 충분하지만, 망설임과
자기정정이 섞인 실제 발화는 해석하지 못한다::

    "나는 선풍기를 잡을까 말까 안 잡을래 나는 컵을 잡고 싶어"
      규칙 → ['fan', '잡을까', '말까', ...]   ← 거절한 선풍기를 찾으러 간다
      LLM  → ['cup']

또 사전에 없는 한국어는 영어로 바뀌지 않아 검출기에 무의미한 벡터가 들어간다.
LLM은 어휘 제한 없이 적절한 영어 라벨을 만들어 준다.

**속도 원칙**
------------
LLM은 **발화할 때만** 돈다(수십 초에 한 번). 프레임 경로(15fps)와 검출 루프는
전혀 거치지 않는다. 그래서 실시간성에 영향이 없다. 게다가 짧은 명령은 규칙이
0ms에 처리하고 LLM을 아예 부르지 않는다.

**실패해도 멈추지 않는다**
-----------------------
Ollama가 꺼져 있거나, 타임아웃이거나, 응답이 깨지면 규칙 파서 결과로 되돌아간다.
데모 중 LLM이 죽어도 지금까지의 동작이 그대로 유지된다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..utils.logging import get_logger
from .command import Intent, parse_command
from .ko_labels import KO_EN_LABELS

log = get_logger(__name__)

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b"

#: 응답 형식을 강제한다. 작은 모델이 형식을 깨는 것을 막는 핵심 장치.
_SCHEMA = {
    "type": "object",
    "properties": {
        "targets": {"type": "array", "items": {"type": "string"}},
        "focus": {"type": ["boolean", "null"]},
        "select": {"type": ["string", "null"], "enum": ["next", "nearest", None]},
    },
    "required": ["targets", "focus", "select"],
}

_EXAMPLES = """예시:
발화: "컵 찾아줘"
{"targets": ["cup"], "focus": null, "select": null}

발화: "포커스"
{"targets": [], "focus": true, "select": null}

발화: "포커스 꺼줘"
{"targets": [], "focus": false, "select": null}

발화: "다음"
{"targets": [], "focus": null, "select": "next"}

발화: "제일 가까운 거"
{"targets": [], "focus": null, "select": "nearest"}

발화: "컵 말고 볼펜"
{"targets": ["pen"], "focus": null, "select": null}

발화: "나는 의자를 볼까 말까 안 볼래 나는 책을 보고 싶어"
{"targets": ["book"], "focus": null, "select": null}

발화: "가방은 됐고 시계 확대해줘"
{"targets": ["clock"], "focus": true, "select": null}

발화: "선풍기는 됐고 볼펜 보여줘"
{"targets": ["pen"], "focus": null, "select": null}

발화: "가방은 됐고 컵 확대해줘"
{"targets": ["cup"], "focus": true, "select": null}

발화: "미니 휴대용 선풍기 잡아줘"
{"targets": ["fan"], "focus": null, "select": null}

발화: "선풍기는 됐고 볼펜 확대해줘"
{"targets": ["pen"], "focus": true, "select": null}

발화: "컵은 별로고 마우스가 나을 것 같아"
{"targets": ["mouse"], "focus": null, "select": null}

발화: "아 아니다 그냥 노트북"
{"targets": ["laptop computer"], "focus": null, "select": null}

발화: "물병이랑 컵 둘 다"
{"targets": ["water bottle", "cup"], "focus": null, "select": null}
"""


def _system_prompt() -> str:
    vocab = ", ".join(sorted(set(KO_EN_LABELS.values())))
    return f"""너는 로봇 비전 시스템의 음성 명령 해석기다.
한국어 발화에서 사용자가 **최종적으로** 원하는 것만 뽑아 JSON으로 답한다.

필드:
- targets: 최종적으로 찾기 원하는 물체의 영어 라벨 배열.
  **중요**: 거절 표현("됐고", "말고", "안 잡을래", "별로고")은 그 **앞의 물체 하나만**
  버린다는 뜻이다. 문장에 남은 다른 물체는 **반드시 targets에 넣는다**.
  거절 표현이 있다고 해서 배열을 비우면 안 된다. 물체 언급이 전혀 없을 때만 빈 배열.
- focus: **"포커스" 또는 "확대"라는 말이 있을 때만** true/false로 채운다.
  "잡아줘", "보여줘", "찾아줘"는 포커스가 아니다 — 이때 focus는 반드시 null이다.
- select: "다음/다른 거"면 "next", "가까운 거/제일 가까운"이면 "nearest", 없으면 null.

혼동하기 쉬운 대응(반드시 지킨다):
  노트북=laptop computer, 공책/수첩=notebook
  선풍기/팬=fan, 에어컨=air conditioner
  물병=water bottle, 컵=cup, 머그컵=mug
  볼펜/펜=pen, 연필=pencil
수식어("미니", "휴대용", "무선", "작은")는 빼고 사물 이름만 남긴다.
거절 표현("됐고", "말고", "안 잡을래", "별로고")이 있어도, 같은 문장에 확대·포커스
요청이 있으면 focus는 그대로 반영한다.

영어 라벨은 되도록 다음에서 고른다. 없으면 적절한 일반 영어 명사를 쓴다:
{vocab}

{_EXAMPLES}
JSON만 출력한다."""


class OllamaIntentParser:
    """로컬 Ollama로 발화를 Intent로 바꾼다.

    Parameters
    ----------
    model:
        Ollama 모델 태그(예: ``qwen2.5:3b``).
    host:
        Ollama 서버 주소.
    timeout:
        응답 대기 상한(초). 넘기면 규칙 파서로 폴백한다.
    keep_alive:
        모델을 메모리에 유지하는 시간. 짧으면 메모리를 빨리 돌려주지만 다음
        호출에서 다시 로드하느라 느려진다.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 6.0,
        keep_alive: str = "30m",
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._system = _system_prompt()

    def available(self) -> bool:
        """서버가 떠 있고 이 모델이 있는지 확인한다."""
        try:
            with urllib.request.urlopen(f"{self._host}/api/tags", timeout=2.0) as r:
                tags = json.load(r)
        except Exception as e:
            log.warning("Ollama에 연결할 수 없다(규칙 파서만 사용): %s", e)
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self._model not in names:
            log.warning("Ollama에 모델 %s 가 없다 (있는 것: %s)", self._model, ", ".join(sorted(names)) or "없음")
            return False
        return True

    def warmup(self) -> None:
        """첫 호출이 느린 것을 피하려고 모델을 미리 올린다."""
        try:
            self.parse("컵")
        except Exception:
            pass

    def parse(self, text: str) -> Intent | None:
        """발화를 Intent로 바꾼다. 실패하면 None(호출자가 규칙으로 폴백)."""
        body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": text},
            ],
            "format": _SCHEMA,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{self._host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                payload = json.load(r)
            data = json.loads(payload["message"]["content"])
        except Exception as e:
            log.warning("LLM 해석 실패(규칙으로 대체): %s", e)
            return None

        targets = [str(t).strip().lower() for t in data.get("targets", []) if str(t).strip()]
        focus = data.get("focus")
        select = data.get("select")
        if focus is not None and not isinstance(focus, bool):
            focus = None
        if select not in ("next", "nearest"):
            select = None
        return Intent(raw=text, targets=targets, focus=focus, select=select)


def _rule_result_is_clean(intent: Intent) -> bool:
    """규칙 결과가 믿을 만한지.

    한글이 그대로 남은 라벨이 있으면 사전에서 못 찾은 것이므로 믿을 수 없다.
    """
    return bool(intent.targets) and not any(
        any("가" <= ch <= "힣" for ch in t) for t in intent.targets
    )


def resolve_intent(text: str, llm: OllamaIntentParser | None = None) -> Intent:
    """발화를 Intent로 바꾼다 — 규칙 우선, 필요할 때만 LLM.

    짧고 명확한 명령("컵", "포커스", "다음")은 규칙이 0ms에 처리하고 LLM을 부르지
    않는다. 사전에서 못 찾은 말이 있거나 문장이 길면 LLM으로 넘긴다.
    """
    rule = parse_command(text)

    if llm is None:
        return rule

    short = len(text.split()) <= 4
    if short and not rule.is_noop() and (_rule_result_is_clean(rule) or not rule.targets):
        return rule  # 규칙으로 충분하다

    llm_intent = llm.parse(text)
    if llm_intent is None or llm_intent.is_noop():
        return rule  # LLM 실패·빈 응답 → 규칙 결과 사용
    return llm_intent
