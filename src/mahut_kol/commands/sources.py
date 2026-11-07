"""
Komut kaynakları: kayıt betiğine hangi komutun verileceği.

- queue (varsayılan): BalancedCommandQueue -> kapsama garantili sıra.
- text: Türkçe cümle -> CommandParser -> Command.
- voice: VoiceCommander.listen_and_parse -> CommandParser JSON -> Command.

Ses/metin geçersiz çıkarsa (valid=False, onay bekleyen, pick_and_place değil, TOP+prizma vb.)
`next_command` (None, açıklama) döner ve bölüm BAŞLATILMAZ.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from mahut_kol.commands.queue import BalancedCommandQueue
from mahut_kol.commands.schema import Command

SourceResult = Tuple[Optional[Command], Dict[str, Any]]


class CommandSource:
    name = "base"

    def next_command(self) -> SourceResult:  # pragma: no cover - arayüz
        raise NotImplementedError

    def on_saved(self, cmd: Command) -> None:
        """Bölüm kaydedildiğinde çağrılır."""

    def on_failed(self, cmd: Command) -> None:
        """Bölüm başarısız/atıldığında çağrılır."""


class QueueCommandSource(CommandSource):
    name = "queue"

    def __init__(self, queue: BalancedCommandQueue):
        self.queue = queue

    def next_command(self) -> SourceResult:
        cmd = self.queue.peek()
        return cmd, {"source": "queue"}

    def on_saved(self, cmd: Command) -> None:
        self.queue.commit(cmd)

    def on_failed(self, cmd: Command) -> None:
        self.queue.report_failure(cmd)


def command_from_parsed(parsed: Dict[str, Any], heldout_keys=frozenset(), include_heldout=False) -> SourceResult:
    """CommandParser çıktısını doğrular; geçersizse (None, info)."""
    info: Dict[str, Any] = {"parser": {k: parsed.get(k) for k in (
        "raw_text", "action", "manipulated_object", "reference_object", "relation", "valid", "error",
        "needs_confirmation", "confirmation_prompt",
    )}}
    try:
        cmd = Command.from_parser_json(parsed)
    except ValueError as e:
        info["error"] = str(e)
        return None, info
    if not include_heldout and cmd.key in heldout_keys:
        info["error"] = f"Bu komut değerlendirme (held-out) kümesinde: {cmd.key} (--include-heldout ile açılır)"
        return None, info
    return cmd, info


class TextCommandSource(CommandSource):
    name = "text"

    def __init__(
        self,
        parser=None,
        input_fn: Callable[[str], str] = input,
        heldout_keys=frozenset(),
        include_heldout: bool = False,
    ):
        if parser is None:
            from mahut_kol.speech.command_parser import CommandParser

            parser = CommandParser()
        self.parser = parser
        self.input_fn = input_fn
        self.heldout_keys = heldout_keys
        self.include_heldout = include_heldout

    def next_command(self) -> SourceResult:
        text = self.input_fn("Türkçe komut: ")
        cmd, info = command_from_parsed(self.parser.parse(text), self.heldout_keys, self.include_heldout)
        info["source"] = "text"
        info["raw_text"] = text
        return cmd, info


class VoiceCommandSource(CommandSource):
    name = "voice"

    def __init__(
        self,
        commander=None,
        mode: str = "push_to_talk",
        heldout_keys=frozenset(),
        include_heldout: bool = False,
        **commander_kwargs,
    ):
        if commander is None:
            from mahut_kol.speech.voice_commander import VoiceCommander

            commander = VoiceCommander(**commander_kwargs)
        self.commander = commander
        self.mode = mode
        self.heldout_keys = heldout_keys
        self.include_heldout = include_heldout

    def next_command(self) -> SourceResult:
        transcript, parsed = self.commander.listen_and_parse(mode=self.mode)
        cmd, info = command_from_parsed(parsed, self.heldout_keys, self.include_heldout)
        info["source"] = "voice"
        info["transcript"] = transcript
        return cmd, info
