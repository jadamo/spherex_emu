import pytest, os

import spherex_emu.emulator as emulator

def test_single_tracer_network():

    test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    test_dir+="/configs/example.yaml"

    test_emulator = emulator.pk_emulator(test_dir)
