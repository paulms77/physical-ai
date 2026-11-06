"""한국어 → 영어 라벨 사전.

**왜 필요한가**
--------------
YOLOE의 텍스트 인코더(MobileCLIP)는 영어 전용이다. 한국어를 그대로 넣으면 의미가
담기지 않을 뿐 아니라, 모든 한국어 단어가 서로 거의 같은 벡터가 되어 구분조차
되지 않는다(실측)::

    pen ↔ 펜      +0.244        pen ↔ cup     +0.303   ← '컵'이 '펜'보다 가깝다
    펜  ↔ 팬      +0.968        펜  ↔ 선풍기   +0.922   ← 한국어끼리 뭉친다

그래서 한국어는 **반드시** 영어로 바꿔서 넘겨야 한다. 이 파일이 그 대응표다.

**영어 라벨 선택 기준**
--------------------
YOLOE는 open-vocabulary라 고정 클래스 목록이 없다(``model.names``는 자리표시자).
다만 LVIS 1203개 클래스로 평가됐으므로, 그 어휘에 있는 표현을 쓰면 검출이 더
안정적이다. 가능한 한 LVIS 표기를 따랐고, 없는 것은 :data:`EXPERIMENTAL`로 표시해
둔다(안 잡혔을 때 모델 한계인지 사전 문제인지 구분하기 위함).

**늘리는 법**
------------
새 단어는 여기에 한 줄 추가하면 된다. 영어 라벨은 LVIS에 있는 표현을 우선한다.
"""

from __future__ import annotations

KO_EN_LABELS: dict[str, str] = {
    # --- 필기구 ---
    "펜": "pen",
    "볼펜": "pen",
    "연필": "pencil",
    "샤프": "pencil",
    "마커": "marker",
    "매직": "marker",
    "형광펜": "marker",
    "지우개": "eraser",
    "자": "ruler",
    "가위": "scissors",
    "칼": "knife",
    "커터": "knife",
    "스테이플러": "stapler",
    "호치키스": "stapler",
    "테이프": "tape",
    "필통": "pencil box",
    "연필깎이": "pencil sharpener",
    "붓": "paintbrush",
    "크레용": "crayon",
    # --- 종이류 ---
    "책": "book",
    "교재": "hardback book",
    "노트": "notebook",
    "공책": "notebook",
    "수첩": "notebook",
    "종이": "paper towel",
    "봉투": "envelope",
    "서류철": "binder",
    "바인더": "binder",
    "클립보드": "clipboard",
    "명함": "business card",
    "카드": "card",
    "엽서": "postcard",
    "신문": "newspaper",
    "잡지": "booklet",
    "책갈피": "bookmark",
    # --- 컵·식기 ---
    "컵": "cup",
    "머그컵": "mug",
    "머그": "mug",
    "찻잔": "teacup",
    "텀블러": "thermos bottle",
    "보온병": "thermos bottle",
    "물병": "water bottle",
    "병": "bottle",
    "와인병": "wine bottle",
    "맥주병": "beer bottle",
    "캔": "can",
    "음료수캔": "beer can",
    "유리컵": "glass",
    "와인잔": "wineglass",
    "그릇": "bowl",
    "접시": "plate",
    "쟁반": "tray",
    "숟가락": "spoon",
    "젓가락": "chopstick",
    "포크": "fork",
    "주전자": "kettle",
    "티팟": "teapot",
    "냅킨": "napkin",
    "물컵": "cup",
    # --- 전자기기 ---
    "노트북": "laptop computer",
    "랩탑": "laptop computer",
    "컴퓨터": "laptop computer",
    "모니터": "monitor",
    "화면": "monitor",
    "티비": "television set",
    "텔레비전": "television set",
    "키보드": "computer keyboard",
    "마우스": "mouse",
    "마우스패드": "mousepad",
    "휴대폰": "cellular telephone",
    "핸드폰": "cellular telephone",
    "폰": "cellular telephone",
    "스마트폰": "cellular telephone",
    "전화기": "telephone",
    "태블릿": "laptop computer",
    "아이패드": "laptop computer",
    "카메라": "camera",
    "렌즈": "camera lens",
    "이어폰": "earphone",
    "헤드폰": "earphone",
    "스피커": "speaker",
    "마이크": "microphone",
    "리모컨": "remote control",
    "배터리": "battery",
    "건전지": "battery",
    "충전기": "battery charger",
    "케이블": "cable",
    "선": "cable",
    "콘센트": "wall socket",
    # --- 가구·공간 ---
    "의자": "chair",
    "안락의자": "armchair",
    "소파": "sofa",
    "스툴": "stool",
    "책상": "desk",
    "테이블": "table",
    "식탁": "dining table",
    "책장": "bookcase",
    "선반": "bookcase",
    "서랍": "drawer",
    "캐비닛": "file cabinet",
    "침대": "bed",
    "베개": "pillow",
    "쿠션": "pillow",
    "담요": "blanket",
    "이불": "bedspread",
    "커튼": "curtain",
    "거울": "mirror",
    "액자": "mirror",
    # --- 조명·가전 ---
    "램프": "lamp",
    "스탠드": "table lamp",
    "전등": "lamp",
    "선풍기": "fan",
    "팬": "fan",
    "에어컨": "air conditioner",
    "히터": "heater",
    "온도계": "thermometer",
    "시계": "clock",
    "알람시계": "alarm clock",
    "벽시계": "wall clock",
    "손목시계": "watch",
    # --- 가방·소지품 ---
    "가방": "backpack",
    "백팩": "backpack",
    "핸드백": "handbag",
    "지갑": "wallet",
    "캐리어": "suitcase",
    "여행가방": "suitcase",
    "쇼핑백": "shopping bag",
    "비닐봉지": "plastic bag",
    "우산": "umbrella",
    "열쇠": "key",
    "키": "key",
    "안경": "sunglasses",
    "선글라스": "sunglasses",
    "모자": "hat",
    "야구모자": "baseball cap",
    "신발": "shoe",
    "양말": "sock",
    "장갑": "glove",
    "마스크": "mask",
    # --- 용기·정리 ---
    "상자": "box",
    "박스": "box",
    "택배": "carton",
    "바구니": "basket",
    "양동이": "bucket",
    "쓰레기통": "trash can",
    "휴지통": "trash can",
    "공구함": "toolbox",
    "통": "canister",
    # --- 위생·의료 ---
    "휴지": "toilet tissue",
    "티슈": "tissue paper",
    "수건": "towel",
    "타월": "hand towel",
    "칫솔": "toothbrush",
    "비누": "soap",
    "샴푸": "shampoo",
    "빗": "hairbrush",
    "주사기": "syringe",
    "약": "medicine",
    "구급상자": "first-aid kit",
    "체온계": "thermometer",
    "저울": "scale",
    "휠체어": "wheelchair",
    "계산기": "calculator",
    # --- 식물·음식 ---
    "화분": "flowerpot",
    "식물": "flowerpot",
    "꽃병": "vase",
    "사과": "apple",
    "바나나": "banana",
    "오렌지": "orange",
    "귤": "mandarin orange",
    "양초": "candle",
    # --- 사람 ---
    "사람": "person",
    "손": "person",
}

#: LVIS 1203 클래스 밖이라 검출이 보장되지 않는 영어 라벨.
#: (CLIP 임베딩에만 기대므로 잡힐 수도, 안 잡힐 수도 있다)
EXPERIMENTAL: frozenset[str] = frozenset({
    "monitor", "battery charger", "cable", "medicine",
})
