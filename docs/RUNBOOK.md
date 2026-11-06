# 운영 매뉴얼 — 띄우고, 확인하고, 끄기

> 데모를 돌릴 때 실제로 치는 명령들. macOS(M1) 수신기 기준이다.
> 음성으로 무엇을 말하는지는 [VOICE_CONTROL.md](VOICE_CONTROL.md) 참고.

---

## 1. IP 확인

앱에 수동 입력할 주소다. **Wi-Fi가 바뀌면 달라진다.**

```bash
ipconfig getifaddr en0
```

`en0`이 Wi-Fi다. 유선 어댑터를 쓰면 `en1`~`en4`도 확인한다.

```bash
# 활성 인터페이스 전부
for i in en0 en1 en2 en3 en4; do
  ip=$(ipconfig getifaddr $i 2>/dev/null) && echo "$i: $ip"
done
```

> `hostname -I`는 **Linux 전용**이라 macOS에서는 `illegal option` 에러가 난다.
> A6000 서버에서는 그쪽을 쓴다.

iPhone 쪽 IP는 설정 → Wi-Fi → 연결된 네트워크 ⓘ 에서 본다. 앞 세 자리가
Mac과 같으면 같은 서브넷이고, 그때만 Bonjour 자동 연결이 동작한다.

## 2. 실행

대화형 음성 이해를 쓰려면 Ollama를 먼저 띄운다(선택 — 없어도 규칙 파서로 동작).

```bash
ollama serve &                 # 이미 떠 있으면 생략
```

그다음 수신기를 띄운다.

```bash
cd ~/pysical
source .venv/bin/activate
python scripts/run_live.py --prompt "컵" --viz --focus
```

`source`를 빼먹으면 `.venv/bin`이 PATH에 없어 `--viz`가 실패한다
(`Failed to find Rerun Viewer executable in PATH`).

### 실행 변형

| 목적 | 명령 |
| --- | --- |
| 대상을 미리 지정 | `python scripts/run_live.py --prompt "컵" --viz --focus` |
| 음성으로만 지정 | `python scripts/run_live.py --voice-only --viz --focus` |
| 검출 없이 궤적만 | `python scripts/run_live.py --no-detector --viz` |
| 장치 강제 | `--device cpu` 또는 `--device mps` |
| 검출 주기 조정 | `--detect-every 3` (기본 5, 낮을수록 자주 검출) |
| 신뢰도 임계 | `--conf 0.3` (기본 0.4, 낮출수록 많이 잡되 오검출↑) |
| 음성 이해 끄기 | `--llm none` (규칙 파서만 — Ollama 불필요) |
| 모델 바꾸기 | `--llm-model qwen2.5:1.5b` (더 가벼움) |

`--voice-only`는 첫 음성이 올 때까지 검출을 돌리지 않는다. 카메라 궤적만 그려진다.

### 시작 로그에서 확인할 것

```text
검출 장치: mps                                  ← CPU면 약 3배 느리다
의도 해석: Ollama qwen2.5:3b                     ← 없으면 "규칙 파서만"
초기 대상: ['cup']
수신기 시작 — ws://0.0.0.0:8765
Bonjour 광고 시작: PysicalReceiver (192.168.0.10:8765)
```

`YOLOE 로드` 줄이 없으면 `--no-detector`로 켠 것이다.

## 3. 상태 확인

```bash
# 수신기가 돌고 있는지 (실행 중인 옵션까지 보인다)
ps aux | grep run_live | grep -v grep

# 포트가 열려 있는지 / 누가 붙어 있는지
lsof -nP -iTCP:8765

# rerun 뷰어가 떠 있는지
lsof -nP -iTCP:9876

# Ollama 서버와 적재된 모델
lsof -nP -iTCP:11434
ollama ps
```

`ollama ps`는 모델이 GPU에 올라가 있는지와 언제 내려가는지를 보여준다.

```text
NAME          SIZE      PROCESSOR    UNTIL
qwen2.5:3b    2.1 GB    100% GPU     18 minutes from now
```

`8765`에 `ESTABLISHED` 줄이 보이고 상대 IP가 iPhone이면 연결된 것이다.

```text
TCP 192.168.0.10:8765->192.168.0.23:58074 (ESTABLISHED)
                        ^^^^^^^^^^^^ iPhone
```

## 4. 종료

### 정상 종료

수신기를 띄운 터미널에서 **Ctrl+C**. rerun 창은 따로 ⌘Q로 닫는다.

### 강제 종료

```bash
pkill -f run_live.py
```

### Ollama 끄기

로그인 시 자동 실행으로 등록하지 않았으므로, **재부팅하면 저절로 꺼진다.**
직접 끄려면:

```bash
pkill ollama
```

모델만 메모리에서 내리려면(서버는 유지):

```bash
ollama stop qwen2.5:3b
```

기본 설정으로 30분 동안 요청이 없으면 모델이 알아서 내려간다. 디스크의
가중치(1.9GB)는 그대로 남아 다음에 바로 쓸 수 있다.

### venv 빠져나오기

```bash
deactivate
```

프롬프트 앞의 `(.venv)`가 사라진다. 터미널을 닫아도 자동으로 해제되므로 굳이
칠 필요는 없다. 실행 중인 수신기를 끄지는 않는다 — 그건 Ctrl+C로 따로 끈다.

### 완전 초기화

레이아웃이 꼬였거나 recording이 여러 개 쌓였을 때.

```bash
pkill -f run_live.py          # 수신기 종료
# rerun 창을 ⌘Q로 완전히 종료
lsof -nP -iTCP:8765 -iTCP:9876   # 둘 다 비었는지 확인
```

**rerun 창을 닫지 않고 수신기만 재시작하면** 기존 창에 recording이 하나 더
쌓이고, 창은 이전 recording을 계속 보여준다. 화면이 안 바뀌면 이 상태다.

## 5. 자주 막히는 지점

| 증상 | 확인 |
| --- | --- |
| `rerun` 실행 파일 없음 | `source .venv/bin/activate` 했는지 |
| `address already in use` | 이전 수신기가 남아 있다 → `pkill -f run_live.py` |
| 앱이 자동 연결 안 됨 | 서브넷이 다르다 → 앱에 IP 수동 입력 |
| 화면이 안 바뀜 | rerun 창이 낡은 recording을 보고 있다 → 창을 닫고 재시작 |
| 검출이 안 됨 | `--no-detector`거나 `--voice-only`인데 대상 미지정 |
| 검출이 느림 | 로그에 `검출 장치: cpu`면 MPS가 안 잡힌 것 |
| 긴 문장을 못 알아들음 | 로그에 `의도 해석: 규칙 파서만`이면 Ollama가 안 떠 있다 |
| 첫 발화만 느림 | 모델 최초 적재(~4초). 이후 ~800ms |

## 6. mock으로 확인 (iPhone 없이)

파이프라인만 검증할 때. 앱과 **같은 와이어 포맷**으로 가짜 프레임을 보낸다.

```bash
# 터미널 1
cd ~/pysical && source .venv/bin/activate
python scripts/run_live.py --no-detector --viz

# 터미널 2
cd ~/pysical && source .venv/bin/activate
python scripts/run_mock_sender.py
```

rerun에 가상 카메라가 원을 그리며 돌면 수신·디코드·좌표변환·시각화가 정상이다.

## 7. 음성 이해 LLM (Ollama)

긴 대화형 발화를 해석하는 데만 쓴다. 짧은 명령은 규칙 파서가 0ms에 처리하므로
LLM을 부르지 않는다. 자세한 동작은 [VOICE_CONTROL.md](VOICE_CONTROL.md) 참고.

```bash
ollama serve &                 # 서버 기동
ollama list                    # 받아둔 모델
ollama ps                      # 지금 메모리에 올라간 모델
ollama stop qwen2.5:3b         # 모델 내리기
pkill ollama                   # 서버 종료
```

| | |
| --- | --- |
| 모델 | `qwen2.5:3b` (디스크 1.9GB, 실행 2.1GB, 100% GPU) |
| 지연 | 최초 적재 ~4초, 이후 ~800ms |
| 없어도 되나 | 된다 — 자동으로 규칙 파서로 되돌아간다 |
| 자동 실행 | 등록 안 함. 재부팅하면 직접 띄워야 한다 |

모델을 다시 받으려면 네트워크가 필요하다. **데모 장소의 망이 다를 수 있으므로
미리 받아두는 것이 안전하다**(한 번 받으면 `~/.ollama`에 남는다).

## 8. 테스트

```bash
cd ~/pysical && source .venv/bin/activate
python -m pytest tests/ -q
```

## 9. iOS 앱 빌드 (Mac)

```bash
cd ~/Downloads/ios_app
xcodegen                          # project.yml → .xcodeproj 재생성
open PysicalSensor.xcodeproj
```

Organizer(⇧⌘O) → Archives → 최신 선택 → Distribute App → TestFlight Internal Only.

**업로드할 때마다** `project.yml`의 `CURRENT_PROJECT_VERSION`을 올려야 한다.
같은 빌드 번호는 App Store Connect가 거부한다.

> 권한·아이콘·버전은 전부 `project.yml`에서 생성된다. `Info.plist`를 직접
> 고치면 다음 `xcodegen`에서 사라진다.
