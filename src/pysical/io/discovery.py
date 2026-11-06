"""Bonjour(mDNS) 서비스 광고 — iOS 앱이 노트북 수신기를 자동 탐색하게 한다.

iOS 앱의 WebSocketClient는 `_pysical._tcp` 서비스를 NWBrowser로 찾는다. 이 모듈이
노트북 쪽에서 그 서비스를 광고해, 데모마다 IP를 손으로 넣는 수고를 없앤다(README §5).

zeroconf 라이브러리를 쓴다. 광고는 선택적이며, 실패해도 수동 IP 연결은 가능하다.
"""

from __future__ import annotations

import socket
import threading

from ..utils.logging import get_logger

log = get_logger(__name__)

SERVICE_TYPE = "_pysical._tcp.local."


class ServiceAdvertiser:
    """수신기 서비스를 Bonjour로 광고한다(컨텍스트 매니저)."""

    def __init__(self, port: int, name: str = "PysicalReceiver") -> None:
        self._port = port
        self._name = name
        self._zc = None
        self._info = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ServiceAdvertiser":
        # register_service는 mDNS 이름 충돌 검사를 기다리느라 블로킹한다. mDNS가
        # 막힌 망에서는 타임아웃까지 ~18초가 걸려, 그동안 WebSocket 서버가 뜨지
        # 못한다(실측). 광고는 있으면 좋은 부가 기능이므로 뒤에서 돌린다.
        self._thread = threading.Thread(target=self._register, daemon=True)
        self._thread.start()
        return self

    def _register(self) -> None:
        try:
            from zeroconf import ServiceInfo, Zeroconf

            ip = _local_ip()
            zc = Zeroconf()
            info = ServiceInfo(
                SERVICE_TYPE,
                f"{self._name}.{SERVICE_TYPE}",
                addresses=[socket.inet_aton(ip)],
                port=self._port,
                properties={"role": "perception"},
            )
            zc.register_service(info)
            self._zc, self._info = zc, info
            log.info("Bonjour 광고 시작: %s (%s:%d)", self._name, ip, self._port)
        except Exception as e:  # zeroconf 없거나 네트워크 문제여도 계속 진행
            log.warning("Bonjour 광고 실패(수동 IP로 연결 가능): %s", e)

    def __exit__(self, *exc) -> None:
        if self._zc is not None and self._info is not None:
            try:
                self._zc.unregister_service(self._info)
                self._zc.close()
            except Exception:
                pass


def _local_ip() -> str:
    """외부로 나가는 인터페이스의 로컬 IP를 추정한다."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # 실제 연결은 안 함, 라우팅만 확인
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()
