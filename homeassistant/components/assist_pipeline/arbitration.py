"""Decide which satellite answers when several hear the same wake word."""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from functools import partial
import logging
import math
import time
from typing import Any

import audioop  # pylint: disable=deprecated-module

from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.storage import Store
from homeassistant.util.hass_dict import HassKey

from .const import DOMAIN, SAMPLE_WIDTH, WAKE_WORD_COOLDOWN
from .error import DuplicateWakeUpDetectedError

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = f"{DOMAIN}.wake_word_arbitration"
STORAGE_VERSION = 1

DEFAULT_WINDOW_MS = 400
MAX_WINDOW_MS = 2000

_FULL_SCALE = 32768
_SAVE_DELAY = 1  # seconds


class WakeWordArbitrationMode(StrEnum):
    """How to pick between satellites that heard the same wake word."""

    FIRST = "first"
    """The first satellite to report wins at once."""

    LOUDEST = "loudest"
    """The satellite that heard the wake word loudest wins."""


@dataclass(frozen=True, slots=True, kw_only=True)
class WakeWordArbitrationSettings:
    """Settings for picking between satellites."""

    mode: WakeWordArbitrationMode = WakeWordArbitrationMode.FIRST

    window_ms: int = DEFAULT_WINDOW_MS
    """How long to wait for other satellites after the first one reports."""

    satellite_offsets_db: Mapping[str, float] = field(default_factory=dict)
    """Correction added to a satellite's level, since microphones differ."""

    def as_dict(self) -> dict[str, Any]:
        """Return the settings as stored and sent to the frontend."""
        return {
            "mode": self.mode.value,
            "window_ms": self.window_ms,
            "satellite_offsets_db": dict(self.satellite_offsets_db),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class WakeWordAudioLevel:
    """What a satellite's wake word audio says about where the person is."""

    level_db: float
    """Loudness in dBFS."""

    tone_ratio: float
    """Sample-to-sample change against loudness. Higher has more treble."""


def measure_wake_word_audio(audio: bytes) -> WakeWordAudioLevel | None:
    """Measure 16-bit mono audio of a wake word."""
    audio = audio[: len(audio) - (len(audio) % SAMPLE_WIDTH)]
    if not audio:
        return None

    level = audioop.rms(audio, SAMPLE_WIDTH)
    # Halved so the difference between neighbouring samples can't clip
    halved = audioop.mul(audio, SAMPLE_WIDTH, 0.5)
    difference = audioop.add(
        halved[SAMPLE_WIDTH:],
        audioop.mul(halved[:-SAMPLE_WIDTH], SAMPLE_WIDTH, -1),
        SAMPLE_WIDTH,
    )
    halved_level = audioop.rms(halved, SAMPLE_WIDTH)
    return WakeWordAudioLevel(
        level_db=20 * math.log10(max(level, 1) / _FULL_SCALE),
        tone_ratio=(
            audioop.rms(difference, SAMPLE_WIDTH) / halved_level
            if halved_level
            else 0.0
        ),
    )


@dataclass(slots=True, kw_only=True)
class _Candidate:
    """A satellite waiting to hear whether it answers."""

    satellite_id: str | None
    level: WakeWordAudioLevel | None
    adjusted_level_db: float | None
    arrival: float
    accepted: asyncio.Future[dict[str, Any]]


@dataclass(slots=True, kw_only=True)
class _Window:
    """The satellites that reported one wake word."""

    opened: float
    cancel: CALLBACK_TYPE
    candidates: list[_Candidate] = field(default_factory=list)


class WakeWordArbiter:
    """Pick the satellite that answers a wake word."""

    def __init__(self, hass: HomeAssistant, last_wake_up: dict[str, float]) -> None:
        """Initialize.

        last_wake_up maps a wake word phrase to when it was last accepted.
        """
        self.hass = hass
        self.store: Store[dict[str, Any]] = Store(hass, STORAGE_VERSION, STORAGE_KEY)
        self.settings = WakeWordArbitrationSettings()
        self._last_wake_up = last_wake_up
        self._windows: dict[str, _Window] = {}
        self._listeners: list[Callable[[], None]] = []

    async def async_load(self) -> None:
        """Load the stored settings."""
        if (data := await self.store.async_load()) is None:
            return
        self.settings = WakeWordArbitrationSettings(
            mode=WakeWordArbitrationMode(data["mode"]),
            window_ms=data["window_ms"],
            satellite_offsets_db=data["satellite_offsets_db"],
        )

    async def async_update_settings(
        self,
        *,
        mode: WakeWordArbitrationMode | None = None,
        window_ms: int | None = None,
        satellite_offsets_db: Mapping[str, float] | None = None,
    ) -> None:
        """Change the given settings and tell listeners."""
        changes: dict[str, Any] = {}
        if mode is not None:
            changes["mode"] = mode
        if window_ms is not None:
            changes["window_ms"] = window_ms
        if satellite_offsets_db is not None:
            changes["satellite_offsets_db"] = dict(satellite_offsets_db)
        self.settings = replace(self.settings, **changes)
        self.store.async_delay_save(self.settings.as_dict, _SAVE_DELAY)
        for listener in list(self._listeners):
            listener()

    @callback
    def async_listen(self, listener: Callable[[], None]) -> CALLBACK_TYPE:
        """Call listener when the settings change."""
        self._listeners.append(listener)
        return partial(self._listeners.remove, listener)

    @property
    def wants_wake_word_audio(self) -> bool:
        """Return whether satellites are compared by their wake word audio."""
        return self.settings.mode is not WakeWordArbitrationMode.FIRST

    async def async_accept(
        self,
        wake_word_phrase: str,
        satellite_id: str | None,
        level: WakeWordAudioLevel | None,
    ) -> dict[str, Any] | None:
        """Return once this satellite is the one to answer the wake word.

        Returns how the satellites compared, or None if the first one to
        report wins. Raises DuplicateWakeUpDetectedError for the others.
        """
        if self.settings.mode is WakeWordArbitrationMode.FIRST:
            self._async_check_cooldown(wake_word_phrase)
            self._last_wake_up[wake_word_phrase] = time.monotonic()
            return None

        now = time.monotonic()
        if (window := self._windows.get(wake_word_phrase)) is None:
            self._async_check_cooldown(wake_word_phrase)
            window = self._windows[wake_word_phrase] = _Window(
                opened=now,
                cancel=async_call_later(
                    self.hass,
                    self.settings.window_ms / 1000,
                    partial(self._async_decide, wake_word_phrase),
                ),
            )

        offset = self.settings.satellite_offsets_db.get(satellite_id or "", 0.0)
        candidate = _Candidate(
            satellite_id=satellite_id,
            level=level,
            adjusted_level_db=None if level is None else level.level_db + offset,
            arrival=now,
            accepted=self.hass.loop.create_future(),
        )
        window.candidates.append(candidate)
        return await candidate.accepted

    @callback
    def _async_check_cooldown(self, wake_word_phrase: str) -> None:
        """Raise if the wake word was accepted moments ago."""
        last_wake_up = self._last_wake_up.get(wake_word_phrase)
        if (
            last_wake_up is not None
            and (time.monotonic() - last_wake_up) < WAKE_WORD_COOLDOWN
        ):
            _LOGGER.debug("Duplicate wake-up detected for %s", wake_word_phrase)
            raise DuplicateWakeUpDetectedError(wake_word_phrase)

    @callback
    def _async_decide(self, wake_word_phrase: str, _now: datetime) -> None:
        """Close the window and tell each satellite whether it answers."""
        window = self._windows.pop(wake_word_phrase)
        # A satellite whose run stopped while waiting can't answer
        candidates = [
            candidate
            for candidate in window.candidates
            if not candidate.accepted.done()
        ]
        if not candidates:
            return

        # The loudest wins, and the earliest of equals. A satellite that sent
        # no wake word audio can't be compared, so it only wins among its kind.
        winner = max(
            candidates,
            key=lambda candidate: (
                candidate.adjusted_level_db is not None,
                candidate.adjusted_level_db or 0.0,
                -candidate.arrival,
            ),
        )
        self._last_wake_up[wake_word_phrase] = time.monotonic()

        decision: dict[str, Any] = {
            "mode": self.settings.mode.value,
            "winner": winner.satellite_id,
            "candidates": [
                {
                    "satellite_id": candidate.satellite_id,
                    "level_db": candidate.level and candidate.level.level_db,
                    "tone_ratio": candidate.level and candidate.level.tone_ratio,
                    "adjusted_level_db": candidate.adjusted_level_db,
                    "arrival_ms": round((candidate.arrival - window.opened) * 1000),
                }
                for candidate in candidates
            ],
        }
        _LOGGER.debug("Wake word arbitration for %s: %s", wake_word_phrase, decision)

        for candidate in candidates:
            if candidate is winner:
                candidate.accepted.set_result(decision)
            else:
                candidate.accepted.set_exception(
                    DuplicateWakeUpDetectedError(wake_word_phrase)
                )


DATA_ARBITER: HassKey[WakeWordArbiter] = HassKey(f"{DOMAIN}.wake_word_arbiter")
