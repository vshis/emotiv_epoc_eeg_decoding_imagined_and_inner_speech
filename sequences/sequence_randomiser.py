import random
from pathlib import Path

speech_types = ["imagined", "inner"]

OVTK_StimulationId_Label_01 = "OVTK_StimulationId_Label_01"
OVTK_StimulationId_Label_02 = "OVTK_StimulationId_Label_02"
OVTK_StimulationId_Label_03 = "OVTK_StimulationId_Label_03"
OVTK_StimulationId_Label_04 = "OVTK_StimulationId_Label_04"
OVTK_StimulationId_Label_05 = "OVTK_StimulationId_Label_05"
OVTK_StimulationId_Label_06 = "OVTK_StimulationId_Label_06"
OVTK_StimulationId_Label_07 = "OVTK_StimulationId_Label_07"
OVTK_StimulationId_Label_08 = "OVTK_StimulationId_Label_08"
OVTK_StimulationId_Label_09 = "OVTK_StimulationId_Label_09"
OVTK_StimulationId_Label_0A = "OVTK_StimulationId_Label_0A"
OVTK_StimulationId_Label_0B = "OVTK_StimulationId_Label_0B"
OVTK_StimulationId_Label_0C = "OVTK_StimulationId_Label_0C"
OVTK_StimulationId_Label_0D = "OVTK_StimulationId_Label_0D"
OVTK_StimulationId_Label_0E = "OVTK_StimulationId_Label_0E"
OVTK_StimulationId_Label_0F = "OVTK_StimulationId_Label_0F"
OVTK_StimulationId_Label_10 = "OVTK_StimulationId_Label_10"

sequence = [OVTK_StimulationId_Label_01,
            OVTK_StimulationId_Label_02,
            OVTK_StimulationId_Label_03,
            OVTK_StimulationId_Label_04,
            OVTK_StimulationId_Label_05,
            OVTK_StimulationId_Label_06,
            OVTK_StimulationId_Label_07,
            OVTK_StimulationId_Label_08,
            OVTK_StimulationId_Label_09,
            OVTK_StimulationId_Label_0A,
            OVTK_StimulationId_Label_0B,
            OVTK_StimulationId_Label_0C,
            OVTK_StimulationId_Label_0D,
            OVTK_StimulationId_Label_0E,
            OVTK_StimulationId_Label_0F,
            OVTK_StimulationId_Label_10]

if not Path("sequences").is_dir():
    Path("sequences").mkdir()

for speech_type in speech_types:
    for i in range(0, 20):
        f = open(Path(f"sequences/sequence_{speech_type}_{str(i).zfill(2)}.txt"), "w")
        random.shuffle(sequence)
        for prompt in sequence:
            f.write(prompt + ",\n")
        f.close()
