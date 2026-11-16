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

from .const import DOMAIN, SAMPLE_RATE, SAMPLE_WIDTH, WAKE_WORD_COOLDOWN
from .error import DuplicateWakeUpDetectedError

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = f"{DOMAIN}.wake_word_arbitration"
STORAGE_VERSION = 1

DEFAULT_WINDOW_MS = 400
MAX_WINDOW_MS = 2000

_FULL_SCALE = 32768

# Measuring the wake word against the noise around it
_FRAME_BYTES = SAMPLE_RATE // 50 * SAMPLE_WIDTH  # 20 ms
_MIN_FRAMES = 10
_RECENT_FRAMES = 75  # the wake word is in the last 1.5 s
_SPEECH_FRAMES = 10  # the loudest 200 ms
_NOISE_QUANTILE = 0.2
_SILENT_FRAME_DB = -120.0
_MIN_NOISE_FLOOR_DB = -75.0
_SAVE_DELAY = 1  # seconds


class WakeWordArbitrationMode(StrEnum):
    """How to pick between satellites that heard the same wake word."""

    FIRST = "first"
    """The first satellite to report wins at once."""

    LOUDEST = "loudest"
    """The satellite that heard the wake word loudest wins."""

    LOUDEST_OVER_NOISE = "loudest_over_noise"
    """The satellite that heard it furthest above its own noise wins."""


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

    over_noise_db: float | None = None
    """Loudness of the wake word over the noise floor of the same audio.

    A microphone with more gain hears the room louder too, so this
    compares between microphones where the level doesn't. None when
    the audio is too short to tell.
    """


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
        over_noise_db=_measure_over_noise(audio),
    )


def _measure_over_noise(audio: bytes) -> float | None:
    """Measure how far the loudest part of the audio is above its noise floor."""
    levels = []
    for start in range(0, len(audio) - _FRAME_BYTES + 1, _FRAME_BYTES):
        level = audioop.rms(audio[start : start + _FRAME_BYTES], SAMPLE_WIDTH)
        levels.append(
            20 * math.log10(level / _FULL_SCALE) if level else _SILENT_FRAME_DB
        )
    if len(levels) < _MIN_FRAMES:
        return None

    speech = sorted(levels[-_RECENT_FRAMES:], reverse=True)[:_SPEECH_FRAMES]
    # Audio a noise suppressor has silenced can't claim an endless margin
    noise_floor = max(
        sorted(levels)[int(len(levels) * _NOISE_QUANTILE)], _MIN_NOISE_FLOOR_DB
    )
    return sum(speech) / len(speech) - noise_floor


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
        measured: float | None = None
        if level is not None:
            measured = (
                level.over_noise_db
                if self.settings.mode is WakeWordArbitrationMode.LOUDEST_OVER_NOISE
                else level.level_db
            )
        candidate = _Candidate(
            satellite_id=satellite_id,
            level=level,
            adjusted_level_db=None if measured is None else measured + offset,
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
                    "over_noise_db": (
                        candidate.level and candidate.level.over_noise_db
                    ),
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
