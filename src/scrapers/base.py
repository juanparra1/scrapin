"""Cliente HTTP base con emulación de TLS (curl_cffi), caché en disco y rate-limiting."""

import hashlib
import json
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from curl_cffi import requests as cffi_requests
from src.config import CACHE_DIR, CACHE_TTL_HOURS, DEFAULT_RATE_LIMIT_SECONDS


class BaseScraperClient:
    """Cliente base con huella TLS de navegador real, thread-safe, disyuntor anti-bloqueo y caché inteligente."""

    _sofascore_blocked_until: float = 0.0
    _sofascore_ever_succeeded: bool = False
    _sofascore_consecutive_403: int = 0

    def __init__(
        self,
        rate_limit_seconds: float = DEFAULT_RATE_LIMIT_SECONDS,
        use_cache: bool = True,
        impersonate: str = "firefox133",
    ) -> None:
        self.rate_limit_seconds = rate_limit_seconds
        self.use_cache = use_cache
        self.impersonate = impersonate
        self._lock = threading.Lock()
        self._local = threading.local()
        self._last_request_ts: float = 0.0

    @classmethod
    def is_sofascore_blocked(cls) -> bool:
        """Indica si el disyuntor (Circuit Breaker) de Sofascore está abierto por bloqueo de IP en la nube."""
        return time.time() < cls._sofascore_blocked_until

    def _get_thread_session(self) -> cffi_requests.Session:
        """Proporciona una sesión TLS aislada y thread-safe por cada hilo de ejecución."""
        if not hasattr(self._local, "session") or self._local.session is None:
            s = cffi_requests.Session(impersonate=self.impersonate)
            s.headers.update(
                {
                    "Accept": "*/*",
                    "Accept-Language": "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Cache-Control": "no-cache",
                    "Origin": "https://www.sofascore.com",
                    "Referer": "https://www.sofascore.com/",
                }
            )
            self._local.session = s
        return self._local.session

    @property
    def session(self) -> cffi_requests.Session:
        return self._get_thread_session()

    def _wait_rate_limit(self) -> None:
        wait_time = 0.0
        with self._lock:
            now = time.time()
            target_delay = self.rate_limit_seconds + random.uniform(0.05, 0.20)
            next_allowed = self._last_request_ts + target_delay
            if now < next_allowed:
                wait_time = next_allowed - now
                self._last_request_ts = next_allowed
            else:
                self._last_request_ts = now
        if wait_time > 0:
            time.sleep(wait_time)

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return CACHE_DIR / f"{digest}.json"

    def get_json(
        self,
        url: str,
        ttl_hours: Optional[float] = CACHE_TTL_HOURS,
        allow_404: bool = True,
        retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """Obtiene JSON desde caché o red aplicando límite de velocidad y disyuntor rápido."""
        cache_file = self._cache_path(url)

        if self.use_cache and ttl_hours is not None and cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600.0
            if age_hours <= ttl_hours:
                try:
                    return json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

        is_sofascore = "sofascore.com" in url
        if is_sofascore and self.is_sofascore_blocked():
            return None

        for attempt in range(1, retries + 1):
            self._wait_rate_limit()
            try:
                session = self._get_thread_session()
                response = session.get(url, timeout=8)

                if response.status_code == 404 and allow_404:
                    if is_sofascore:
                        BaseScraperClient._sofascore_ever_succeeded = True
                        BaseScraperClient._sofascore_consecutive_403 = 0
                    return None

                if response.status_code == 200:
                    if is_sofascore:
                        BaseScraperClient._sofascore_ever_succeeded = True
                        BaseScraperClient._sofascore_consecutive_403 = 0
                    data = response.json()
                    if self.use_cache and ttl_hours is not None:
                        cache_file.write_text(
                            json.dumps(data, ensure_ascii=False), encoding="utf-8"
                        )
                    return data

                if response.status_code == 403:
                    if is_sofascore:
                        BaseScraperClient._sofascore_consecutive_403 += 1
                        # Si nunca ha conectado (IP de Render Cloud bloqueada) o lleva 3 bloqueos 403 seguidos, abrir disyuntor
                        if not BaseScraperClient._sofascore_ever_succeeded or BaseScraperClient._sofascore_consecutive_403 >= 3:
                            BaseScraperClient._sofascore_blocked_until = time.time() + 600.0
                            return None
                    backoff = (1.5 ** attempt) + random.uniform(0.3, 0.8)
                    time.sleep(backoff)
                    continue

                if response.status_code in (429, 503):
                    backoff = (1.8 ** attempt) + random.uniform(0.4, 1.0)
                    time.sleep(backoff)
                    continue

                response.raise_for_status()
            except Exception:
                if is_sofascore and not BaseScraperClient._sofascore_ever_succeeded and attempt >= 1:
                    BaseScraperClient._sofascore_blocked_until = time.time() + 300.0
                    return None
                if attempt == retries:
                    return None
                time.sleep(0.6 + random.uniform(0.2, 0.5))

        return None
