import random
from pathlib import Path


def write_sequence_to_txt(speech_type: str, file_number: int, loops: int, sequence):
    """
    Randomises sequence and adds it to a list. Repeats (loops) a certain amount of times, concatenating each random
    sequence to the end of previous sequence. Writes the result to a txt file
    :param loops: how many random sequences to write to file
    """
    output = []
    for _ in range(0, loops):
        random.shuffle(sequence)
        output = [*output, *sequence]
    print(sequence)
    print(output)

    f = open(Path(f"sequences/sequence_{speech_type}_{file_number}.txt"), "w")
    for prompt in output:
        f.write(prompt + ",\n")
    f.close()


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
    # We want to split the one-hour-long experiment into three 20 minute experiments
    # We want to repeat each class 100 times, during each thinking state, it is repeated 5 times
    # so we need 20 repetitions of each class (OpenVibe label) in the three files, split into 7/7/6
    for i in range(0, 3):
        if i == 2:
            write_sequence_to_txt(speech_type=speech_type, file_number=i, loops=6, sequence=sequence)
        else:
            write_sequence_to_txt(speech_type=speech_type, file_number=i, loops=7, sequence=sequence)

