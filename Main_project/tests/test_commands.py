"""Aşama 2: komut katmanı (şema, talimatlar, slotlar, dengeli kuyruk, held-out, kaynaklar)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mahut_kol.commands import (
    BalancedCommandQueue,
    Command,
    QueueCommandSource,
    TextCommandSource,
    VoiceCommandSource,
    all_instructions,
    all_valid_commands,
    command_from_parsed,
    coverage_tables,
    from_slots,
    load_heldout,
    select_heldout,
    to_english_instruction,
    to_slots,
)
from mahut_kol.envs.multi_shape_task import RELATIONS
from mahut_kol.speech.command_parser import CommandParser


class TestSchema(unittest.TestCase):
    def test_from_parser_json(self):
        parsed = CommandParser().parse("Mavi küpü al kırmızı silindirin sağına koy")
        cmd = Command.from_parser_json(parsed)
        self.assertEqual(cmd, Command("blue_cube", "red_cylinder", "RIGHT"))
        opts = cmd.to_reset_options()
        self.assertEqual(opts["manipulated_object"], "blue_cube")
        self.assertEqual(opts["relation"], "RIGHT")
        self.assertEqual(opts["num_distractors"], 2)

    def test_invalid_parser_json_rejected(self):
        p = CommandParser()
        for text in ("merhaba", "kırmızı küpü al", "mavi küpü al yeşil prizmanın üstüne koy"):
            with self.assertRaises(ValueError, msg=text):
                Command.from_parser_json(p.parse(text))

    def test_instruction_templates(self):
        cmd = Command("blue_cube", "red_cylinder", "RIGHT")
        insts = all_instructions(cmd)
        self.assertEqual(insts[0], "pick the blue cube and place it to the right of the red cylinder")
        self.assertIn("put the blue cube on the right side of the red cylinder", insts)
        self.assertEqual(to_english_instruction(cmd), insts[0])
        top = Command("green_triangular_prism", "red_cube", "TOP")
        self.assertIn("put the green triangular prism on top of the red cube", all_instructions(top))
        # Seed'li ve deterministik
        a = [to_english_instruction(cmd, seed=s) for s in range(20)]
        b = [to_english_instruction(cmd, seed=s) for s in range(20)]
        self.assertEqual(a, b)
        self.assertGreater(len(set(a)), 3)
        for s in a:
            self.assertTrue(s.isascii())

    def test_slots_roundtrip(self):
        for cmd in all_valid_commands():
            s = to_slots(cmd)
            self.assertEqual(s.shape, (5,))
            self.assertEqual(from_slots(s), cmd)
        np.testing.assert_array_equal(to_slots(Command("blue_cube", "red_cylinder", "RIGHT")), [2, 0, 0, 1, 3])


class TestHeldout(unittest.TestCase):
    def test_heldout_file_matches_selection(self):
        held = load_heldout()
        self.assertEqual(len(held), 30)
        self.assertEqual([c.key for c in held], [c.key for c in select_heldout()])
        per_rel = {r: sum(c.relation == r for c in held) for r in RELATIONS}
        self.assertEqual(set(per_rel.values()), {6})

    def test_heldout_keeps_coverage(self):
        held = {c.key for c in load_heldout()}
        train = [c for c in all_valid_commands() if c.key not in held]
        all_m, all_r = coverage_tables(all_valid_commands())
        cm, cr = coverage_tables(train)
        self.assertEqual(set(cm), set(all_m))  # hiçbir hücre boşalmadı
        self.assertEqual(set(cr), set(all_r))
        self.assertGreaterEqual(min(cm.values()), 4)


class TestBalancedQueue(unittest.TestCase):
    def test_coverage_balanced_and_excludes_heldout(self):
        q = BalancedCommandQueue(seed=3)
        cmds = q.take(900)
        held = {c.key for c in load_heldout()}
        self.assertFalse(any(c.key in held for c in cmds))
        cm, cr = coverage_tables(cmds)
        self.assertEqual(len(cm), 45)   # 9 nesne x 5 yön
        self.assertEqual(len(cr), 42)   # prizma referans x TOP yok: 45 - 3
        # Her yön içinde hücreler dengeli. TOP'ta 9 manipüle hücresine karşı 6 referans hücresi
        # olduğu için (prizma referans olamaz) iki aile aynı anda eşitlenemez; TOP hücreleri
        # yapısal olarak farklı düzeyde dengelenir (bkz. queue.py modül açıklaması).
        for fam in (cm, cr):
            for rel in RELATIONS:
                vals = [n for (o, r), n in fam.items() if r == rel]
                self.assertLessEqual(max(vals) - min(vals), 1, (rel, vals))
        planar = [n for (o, r), n in cm.items() if r != "TOP"]
        self.assertLessEqual(max(planar) - min(planar), 1)
        # sıra karışık: ardışık iki komut aynı olmamalı
        self.assertTrue(all(a != b for a, b in zip(cmds, cmds[1:])))

    def test_seed_determinism_and_shuffle(self):
        a = [c.key for c in BalancedCommandQueue(seed=1).take(50)]
        b = [c.key for c in BalancedCommandQueue(seed=1).take(50)]
        c = [c.key for c in BalancedCommandQueue(seed=2).take(50)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_resume_from_state_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "queue_state.json"
            full = [c.key for c in BalancedCommandQueue(seed=5).take(60)]
            q1 = BalancedCommandQueue(path, seed=5)
            first = [c.key for c in q1.take(25)]
            q2 = BalancedCommandQueue(path, seed=5)  # yeniden aç ve devam et
            rest = [c.key for c in q2.take(35)]
            self.assertEqual(first + rest, full)
            self.assertEqual(json.loads(path.read_text())["n_committed"], 60)
            with self.assertRaises(ValueError):
                BalancedCommandQueue(path, seed=6)

    def test_peek_stable_until_commit(self):
        q = BalancedCommandQueue(seed=0)
        c = q.peek()
        self.assertEqual(q.peek(), c)
        q.report_failure(c)
        q.commit(q.peek())
        self.assertEqual(q.n_committed, 1)

    def test_include_heldout(self):
        q = BalancedCommandQueue(seed=0, include_heldout=True)
        self.assertEqual(len(q.commands), 336)
        q2 = BalancedCommandQueue(seed=0)
        self.assertEqual(len(q2.commands), 306)
        with self.assertRaises(ValueError):
            q2.commit(load_heldout()[0])


class TestSources(unittest.TestCase):
    def test_queue_source(self):
        src = QueueCommandSource(BalancedCommandQueue(seed=0))
        cmd, info = src.next_command()
        self.assertIsInstance(cmd, Command)
        src.on_saved(cmd)
        self.assertEqual(src.queue.n_committed, 1)

    def test_text_source_valid_and_invalid(self):
        texts = iter(["Mavi küpü al kırmızı silindirin sağına koy", "kırmızı küpü al", "bla"])
        src = TextCommandSource(input_fn=lambda _p: next(texts))
        cmd, info = src.next_command()
        self.assertEqual(cmd.key, "blue_cube|red_cylinder|RIGHT")
        cmd, info = src.next_command()
        self.assertIsNone(cmd)
        self.assertIn("error", info)
        cmd, info = src.next_command()
        self.assertIsNone(cmd)

    def test_text_source_heldout_blocked(self):
        h = load_heldout()[0]
        parsed = {"valid": True, "action": "pick_and_place", **h.to_dict()}
        cmd, info = command_from_parsed(parsed, {h.key})
        self.assertIsNone(cmd)
        cmd, info = command_from_parsed(parsed, {h.key}, include_heldout=True)
        self.assertEqual(cmd, h)

    def test_voice_source_with_fake_commander(self):
        class FakeCommander:
            def __init__(self):
                self.parser = CommandParser()

            def listen_and_parse(self, mode="push_to_talk"):
                t = "Yeşil üçgen prizmayı al mavi küpün üstüne koy"
                return t, self.parser.parse(t)

        src = VoiceCommandSource(commander=FakeCommander())
        cmd, info = src.next_command()
        self.assertEqual(cmd.key, "green_triangular_prism|blue_cube|TOP")
        self.assertEqual(info["source"], "voice")


if __name__ == "__main__":
    unittest.main()
