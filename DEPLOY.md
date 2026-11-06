# 배포 가이드 — 노트북(수신기)과 Mac(iOS 빌드) 셋업

> 세 배포 패키지를 각 기기로 옮겨 설치하는 법. 역할은 memory: pysical-3device-arch.
>
> - 📱 iPhone = 눈(센서)  · 🍎 Mac = iOS 앱 빌드  · 💻 노트북 = 파이썬 수신기(두뇌+화면)

---

## 패키지 3개

| 파일 | 크기 | 어디로 | 용도 |
| --- | --- | --- | --- |
| `ios_app.tar.gz` | 12KB | 🍎 Mac | iOS 앱 소스(빌드용) |
| `pysical_code.tar.gz` | ~150KB | 💻 노트북 | 파이썬 코드 |
| `pysical_weights.tar.gz` | 642MB | 💻 노트북 | 모델 가중치(YOLOE/depth/ESPCN) |

---

## 💻 노트북 셋업 (수신기)

### 1. 코드 + 모델 받기

```bash
# A6000에서 노트북으로 (노트북 터미널에서, SSH config에 gpu-server 있으면)
scp gpu-server:~/pysical/dist/pysical_code.tar.gz ~/
scp gpu-server:~/pysical/dist/pysical_weights.tar.gz ~/

# 압축 해제 (한 폴더로)
mkdir -p ~/pysical && cd ~/pysical
tar -xzf ~/pysical_code.tar.gz
tar -xzf ~/pysical_weights.tar.gz    # weights/ 가 생김
```

### 2. Python 3.12 venv 구축

```bash
cd ~/pysical
python3.12 -m venv .venv          # 3.12가 없으면 먼저 설치
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements-phase1.txt
```

> `requirements-phase1.txt`가 opencv-contrib·onnxruntime·ultralytics·zeroconf 등을
> 전부 설치한다. (opencv-python이 아니라 opencv-contrib — dnn_superres 필요)

### 3. 수신기 실행

```bash
# 화면 있는 노트북이면 GUI 뷰어로:
.venv/bin/python scripts/run_live.py --prompt "컵" --viz --focus

# 검출 없이 궤적만 먼저 확인(가벼움):
.venv/bin/python scripts/run_live.py --no-detector --viz
```

실행하면 Bonjour로 서비스를 광고하고 WebSocket 서버가 뜬다. iPhone 앱이 자동 연결한다.
연결이 안 되면 앱에서 **노트북 IP를 수동 입력**한다.
IP 확인: macOS는 `ipconfig getifaddr en0`, Linux는 `hostname -I`.
(`hostname -I`는 Linux 전용이라 macOS에서는 에러가 난다)

---

## 🍎 Mac 셋업 (iOS 앱 빌드)

### 1. 앱 소스 받기

```bash
scp gpu-server:~/pysical/dist/ios_app.tar.gz ~/Desktop/
cd ~/Desktop && tar -xzf ios_app.tar.gz    # ios_app/ 생김
```

### 2. Xcode 프로젝트 생성 (XcodeGen)

```bash
brew install xcodegen              # 최초 1회
cd ~/Desktop/ios_app && xcodegen   # PysicalSensor.xcodeproj 생성
open PysicalSensor.xcodeproj
```

### 3. Xcode에서

- Signing & Capabilities → **Team** 본인 계정 선택
- iPhone 연결 → 상단에서 실기기 선택 → **Run(▶)**
- 앱에서 권한 허용(카메라/마이크/음성/로컬네트워크)

자세한 내용은 `ios_app/BUILD.md`.

---

## 연결 확인 (완료 조건)

1. 노트북: `run_live.py` 실행 중
2. iPhone: 앱 실행 → 자동 연결(초록) 또는 노트북 IP 수동 입력
3. **iPhone과 노트북이 같은 Wi-Fi**여야 함
4. 폰을 들고 걸으면 노트북 rerun에서 카메라가 따라 움직인다 → 성공
5. 음성 "컵 찾아줘" → 노트북이 컵을 검출·추적

## 문제 해결

- **헤드리스 서버에서 `--viz` 에러**("Rerun Viewer executable 없음"): 화면 없는 서버다.
  `--save out.rrd`로 저장 후 로컬에서 `rerun out.rrd`로 열거나, `--serve-web`으로 브라우저 접속.
- **연결 안 됨**: 다른 Wi-Fi거나 로컬네트워크 권한 거부. 수동 IP 시도.
- **depth 모델 다운로드 실패**(사내망): weights를 통째로 복사했으면 문제없음(자동 다운로드 불필요).
