"""Test wake word arbitration between satellites."""

import array
import asyncio
from datetime import timedelta
import math
from typing import Any
from unittest.mock import patch

import pytest

from homeassistant.components.assist_pipeline.arbitration import (
    DATA_ARBITER,
    WakeWordArbiter,
    WakeWordArbitrationMode,
    WakeWordAudioLevel,
    measure_wake_word_audio,
)
from homeassistant.components.assist_pipeline.const import (
    SAMPLE_RATE,
    WAKE_WORD_COOLDOWN,
)
from homeassistant.components.assist_pipeline.error import DuplicateWakeUpDetectedError
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from tests.common import async_fire_time_changed, flush_store
from tests.typing import WebSocketGenerator

PHRASE = "hey jarvis"
WINDOW_MS = 400

QUIET = WakeWordAudioLevel(level_db=-40.0, tone_ratio=0.1)
LOUD = WakeWordAudioLevel(level_db=-20.0, tone_ratio=0.1)


def _tone(frequency: int, amplitude: int, duration_ms: int = 100) -> bytes:
    """Return a 16-bit mono sine wave."""
    samples = SAMPLE_RATE * duration_ms // 1000
    return array.array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE))
            for index in range(samples)
        ),
    ).tobytes()


@pytest.fixture
def arbiter(hass: HomeAssistant, init_components: None) -> WakeWordArbiter:
    """Return the arbiter of a set up Assist pipeline integration."""
    return hass.data[DATA_ARBITER]


@pytest.fixture
async def loudest(arbiter: WakeWordArbiter) -> WakeWordArbiter:
    """Return an arbiter that picks the loudest satellite."""
    await arbiter.async_update_settings(
        mode=WakeWordArbitrationMode.LOUDEST, window_ms=WINDOW_MS
    )
    return arbiter


async def _async_close_window(hass: HomeAssistant) -> None:
    """Let the arbitration window pass."""
    async_fire_time_changed(
        hass, dt_util.utcnow() + timedelta(milliseconds=WINDOW_MS + 1)
    )
    await hass.async_block_till_done()


def _candidate(
    hass: HomeAssistant,
    arbiter: WakeWordArbiter,
    satellite_id: str | None,
    level: WakeWordAudioLevel | None,
    phrase: str = PHRASE,
) -> asyncio.Task[dict[str, Any] | None]:
    """Start a satellite's request to answer the wake word."""
    return hass.async_create_task(arbiter.async_accept(phrase, satellite_id, level))


async def _async_candidates(
    hass: HomeAssistant,
    arbiter: WakeWordArbiter,
    *candidates: tuple[str | None, WakeWordAudioLevel | None],
) -> list[asyncio.Task[dict[str, Any] | None]]:
    """Start requests in order, each arriving before the next."""
    tasks = []
    for satellite_id, level in candidates:
        tasks.append(_candidate(hass, arbiter, satellite_id, level))
        await asyncio.sleep(0)
    return tasks


def test_measure_level() -> None:
    """Test louder audio measures a higher level."""
    quiet = measure_wake_word_audio(_tone(440, 1000))
    loud = measure_wake_word_audio(_tone(440, 10000))

    assert quiet is not None
    assert loud is not None
    # Ten times the amplitude is 20 dB
    assert loud.level_db - quiet.level_db == pytest.approx(20, abs=0.1)
    # A full-scale sine wave is about -3 dBFS
    full_scale = measure_wake_word_audio(_tone(440, 32767))
    assert full_scale is not None
    assert full_scale.level_db == pytest.approx(-3, abs=0.1)


def test_measure_tone() -> None:
    """Test audio with more high frequencies measures a higher tone ratio."""
    low = measure_wake_word_audio(_tone(200, 10000))
    high = measure_wake_word_audio(_tone(4000, 10000))

    assert low is not None
    assert high is not None
    assert high.tone_ratio > low.tone_ratio
    # The same pitch reads the same at any volume
    assert measure_wake_word_audio(_tone(200, 1000)).tone_ratio == pytest.approx(  # type: ignore[union-attr]
        low.tone_ratio, rel=0.01
    )


@pytest.mark.parametrize("audio", [b"", b"\x00"])
def test_measure_no_audio(audio: bytes) -> None:
    """Test audio without a whole sample has no level."""
    assert measure_wake_word_audio(audio) is None


def test_measure_silence() -> None:
    """Test silence measures a finite level."""
    silence = measure_wake_word_audio(bytes(320))

    assert silence is not None
    assert math.isfinite(silence.level_db)
    assert silence.level_db < -80
    assert silence.tone_ratio == 0


async def test_first_is_default(hass: HomeAssistant, arbiter: WakeWordArbiter) -> None:
    """Test the first satellite to report wins at once by default."""
    assert arbiter.settings.mode is WakeWordArbitrationMode.FIRST

    # Without waiting for a window, even though the second is louder
    assert await arbiter.async_accept(PHRASE, "first", QUIET) is None
    with pytest.raises(DuplicateWakeUpDetectedError):
        await arbiter.async_accept(PHRASE, "second", LOUD)


async def test_first_cooldown(hass: HomeAssistant, arbiter: WakeWordArbiter) -> None:
    """Test the wake word is accepted again after the cooldown."""
    with patch(
        "homeassistant.components.assist_pipeline.arbitration.time.monotonic",
        return_value=100.0,
    ) as monotonic:
        assert await arbiter.async_accept(PHRASE, "first", None) is None
        monotonic.return_value = 100.0 + WAKE_WORD_COOLDOWN
        assert await arbiter.async_accept(PHRASE, "second", None) is None


async def test_loudest_wins(hass: HomeAssistant, loudest: WakeWordArbiter) -> None:
    """Test the loudest satellite wins, not the first."""
    quiet, loud = await _async_candidates(
        hass, loudest, ("quiet", QUIET), ("loud", LOUD)
    )

    # Nothing is decided while the window is open
    assert not quiet.done()
    assert not loud.done()

    await _async_close_window(hass)

    with pytest.raises(DuplicateWakeUpDetectedError):
        await quiet
    decision = await loud
    assert decision is not None
    assert decision["mode"] == "loudest"
    assert decision["winner"] == "loud"
    assert [
        (candidate["satellite_id"], candidate["level_db"])
        for candidate in decision["candidates"]
    ] == [("quiet", -40.0), ("loud", -20.0)]
    assert all(
        candidate["tone_ratio"] == 0.1 and candidate["arrival_ms"] >= 0
        for candidate in decision["candidates"]
    )


async def test_loudest_lone_satellite(
    hass: HomeAssistant, loudest: WakeWordArbiter
) -> None:
    """Test a satellite nobody else heard wins when the window closes."""
    (only,) = await _async_candidates(hass, loudest, ("only", QUIET))
    assert not only.done()

    await _async_close_window(hass)

    decision = await only
    assert decision is not None
    assert decision["winner"] == "only"


async def test_loudest_offsets(hass: HomeAssistant, loudest: WakeWordArbiter) -> None:
    """Test a satellite's offset corrects a microphone that reads low."""
    await loudest.async_update_settings(satellite_offsets_db={"quiet": 25.0})
    quiet, loud = await _async_candidates(
        hass, loudest, ("quiet", QUIET), ("loud", LOUD)
    )
    await _async_close_window(hass)

    decision = await quiet
    assert decision is not None
    assert decision["winner"] == "quiet"
    assert decision["candidates"][0]["adjusted_level_db"] == -15.0
    with pytest.raises(DuplicateWakeUpDetectedError):
        await loud


async def test_loudest_tie(hass: HomeAssistant, loudest: WakeWordArbiter) -> None:
    """Test the first satellite wins between equally loud ones."""
    first, second = await _async_candidates(
        hass, loudest, ("first", LOUD), ("second", LOUD)
    )
    await _async_close_window(hass)

    assert (await first) is not None
    with pytest.raises(DuplicateWakeUpDetectedError):
        await second


async def test_loudest_without_level(
    hass: HomeAssistant, loudest: WakeWordArbiter
) -> None:
    """Test a satellite that sent no wake word audio loses to one that did."""
    unknown, quiet = await _async_candidates(
        hass, loudest, ("unknown", None), ("quiet", QUIET)
    )
    await _async_close_window(hass)

    with pytest.raises(DuplicateWakeUpDetectedError):
        await unknown
    decision = await quiet
    assert decision is not None
    assert decision["candidates"][0]["level_db"] is None


async def test_loudest_no_levels(hass: HomeAssistant, loudest: WakeWordArbiter) -> None:
    """Test the first satellite wins when none sent wake word audio."""
    first, second = await _async_candidates(
        hass, loudest, ("first", None), (None, None)
    )
    await _async_close_window(hass)

    assert (await first) is not None
    with pytest.raises(DuplicateWakeUpDetectedError):
        await second


async def test_loudest_late_satellite(
    hass: HomeAssistant, loudest: WakeWordArbiter
) -> None:
    """Test a satellite reporting after the decision is a duplicate."""
    (first,) = await _async_candidates(hass, loudest, ("first", QUIET))
    await _async_close_window(hass)
    await first

    with pytest.raises(DuplicateWakeUpDetectedError):
        await loudest.async_accept(PHRASE, "late", LOUD)


async def test_loudest_phrases_are_separate(
    hass: HomeAssistant, loudest: WakeWordArbiter
) -> None:
    """Test satellites woken by different wake words don't compete."""
    jarvis = _candidate(hass, loudest, "jarvis", QUIET)
    nabu = _candidate(hass, loudest, "nabu", LOUD, phrase="okay nabu")
    await asyncio.sleep(0)
    await _async_close_window(hass)

    assert (await jarvis) is not None
    assert (await nabu) is not None


async def test_loudest_satellite_gone(
    hass: HomeAssistant, loudest: WakeWordArbiter
) -> None:
    """Test a satellite whose run stopped during the window can't win."""
    quiet, loud = await _async_candidates(
        hass, loudest, ("quiet", QUIET), ("loud", LOUD)
    )
    loud.cancel()
    await asyncio.sleep(0)
    await _async_close_window(hass)

    decision = await quiet
    assert decision is not None
    assert decision["winner"] == "quiet"
    assert len(decision["candidates"]) == 1


async def test_loudest_all_satellites_gone(
    hass: HomeAssistant, loudest: WakeWordArbiter
) -> None:
    """Test a window nobody is left in doesn't start the cooldown."""
    (gone,) = await _async_candidates(hass, loudest, ("gone", QUIET))
    gone.cancel()
    await asyncio.sleep(0)
    await _async_close_window(hass)

    (again,) = await _async_candidates(hass, loudest, ("again", QUIET))
    await _async_close_window(hass)
    assert (await again) is not None


async def test_settings_are_stored(
    hass: HomeAssistant, arbiter: WakeWordArbiter, hass_storage: dict[str, Any]
) -> None:
    """Test settings are saved and listeners told about changes."""
    calls: list[None] = []
    unsub = arbiter.async_listen(lambda: calls.append(None))

    await arbiter.async_update_settings(
        mode=WakeWordArbitrationMode.LOUDEST,
        window_ms=250,
        satellite_offsets_db={"assist_satellite.office": 3.5},
    )
    await flush_store(arbiter.store)

    assert len(calls) == 1
    assert hass_storage["assist_pipeline.wake_word_arbitration"]["data"] == {
        "mode": "loudest",
        "window_ms": 250,
        "satellite_offsets_db": {"assist_satellite.office": 3.5},
    }

    # What isn't given stays as it was
    await arbiter.async_update_settings(window_ms=300)
    assert arbiter.settings.mode is WakeWordArbitrationMode.LOUDEST
    assert arbiter.settings.window_ms == 300
    assert arbiter.settings.satellite_offsets_db == {"assist_satellite.office": 3.5}

    unsub()
    await arbiter.async_update_settings(window_ms=400)
    assert len(calls) == 2

    # A new arbiter reads what was stored
    await flush_store(arbiter.store)
    restored = WakeWordArbiter(hass, {})
    await restored.async_load()
    assert restored.settings == arbiter.settings


async def test_websocket_settings(
    hass: HomeAssistant,
    arbiter: WakeWordArbiter,
    hass_ws_client: WebSocketGenerator,
) -> None:
    """Test reading and changing settings over the websocket API."""
    client = await hass_ws_client(hass)

    await client.send_json_auto_id({"type": "assist_pipeline/wake_word_arbitration/get"})
    msg = await client.receive_json()
    assert msg["success"]
    assert msg["result"] == {
        "mode": "first",
        "window_ms": 400,
        "satellite_offsets_db": {},
    }

    await client.send_json_auto_id(
        {
            "type": "assist_pipeline/wake_word_arbitration/set",
            "mode": "loudest",
            "satellite_offsets_db": {"assist_satellite.office": -2},
        }
    )
    msg = await client.receive_json()
    assert msg["success"]
    assert msg["result"] == {
        "mode": "loudest",
        "window_ms": 400,
        "satellite_offsets_db": {"assist_satellite.office": -2.0},
    }
    assert arbiter.settings.mode is WakeWordArbitrationMode.LOUDEST

    await client.send_json_auto_id(
        {"type": "assist_pipeline/wake_word_arbitration/set", "mode": "nearest"}
    )
    msg = await client.receive_json()
    assert not msg["success"]
    assert arbiter.settings.mode is WakeWordArbitrationMode.LOUDEST
