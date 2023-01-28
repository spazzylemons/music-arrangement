from dataclasses import dataclass
from typing import Generator
from functools import cache
from math import inf
from itertools import permutations
from sys import argv, platform

import music21 as m21
import numpy as np
import toml

PITCH_MAX = int(m21.pitch.Pitch('B9').ps)
PITCH_MIN = int(m21.pitch.Pitch('C0').ps)

@dataclass
class PitchRange:
    """An inclusive range of pitches."""

    min: int
    """The lowest pitch (inclusive)."""

    max: int
    """The highest pitch (inclusive)."""

    def transposable_pitches(self) -> Generator[int, None, None]:
        """Iterate over each pitch in this range."""
        transpose_min = PITCH_MIN - self.min
        transpose_max = PITCH_MAX - self.max
        yield from range(transpose_min, transpose_max + 1)

    def __contains__(self, other: 'PitchRange') -> bool:
        """Test if the given range fits within this range."""
        return (self.min <= other.min <= self.max) and (self.min <= other.max <= self.max)

    def __or__(self, other: 'PitchRange') -> 'PitchRange':
        """Create the smallest range that can contain both ranges."""
        return PitchRange(min(self.min, other.min), max(self.max, other.max))

    def __add__(self, amount: int) -> 'PitchRange':
        """Create a range that is transposed by the given amount."""
        return PitchRange(self.min + amount, self.max + amount)

    def mid(self) -> float:
        """Get the pitch in the middle of this range."""
        return (self.min + self.max) / 2

@dataclass
class NewPart:
    """A part to arrange the piece for."""

    name: str
    """The name of the part."""

    instrument: m21.instrument.Instrument
    """The music21 instrument for this part."""

    range: PitchRange
    """The range of pitches this part is allowed to play."""

    test_key: m21.key.Key
    """The key that concert C is in for this part."""

@dataclass
class Transposition:
    """A possible transposition."""

    deviation: int
    """The absolute value of half-steps up/down for this transposition."""

    bitmask: int
    """A bitmask of parts that can play this transposed part."""

@dataclass
class Candidate:
    """A complete transposition candidate."""

    transpositions: tuple[int, ...]
    """The transpositions to use for each part."""

    def deviation(self) -> int:
        """Get the total transposition amount."""
        return sum(abs(i) for i in self.transpositions)

def load_parts(arragement_filename: str) -> tuple[NewPart, ...]:
    """Load the part information and select the arrangement."""
    instruments = toml.load('instruments.toml')
    arrangement = toml.load(arragement_filename)
    result = []
    for name in instruments:
        desc = instruments[name]
        # key to test against is whatever key concert C is in the instrument's key
        # e.g. an instrument in the key of Bb gets the test key of D
        test_key = m21.key.Key(m21.pitch.Pitch(60 - m21.key.Key(desc['key']).pitches[0].ps))
        ins = NewPart(
            name=name,
            instrument=getattr(m21.instrument, desc['name'])(),
            range=PitchRange(int(m21.pitch.Pitch(desc['minimum']).ps), int(m21.pitch.Pitch(desc['maximum']).ps)),
            test_key=test_key)
        if name in arrangement:
            for _ in range(arrangement[name]):
                result.append(ins)
    return tuple(result)

def get_part_range(part: m21.stream.Part) -> PitchRange:
    """Get the range of pitches in a part, assuming it has at least one note."""
    high = PITCH_MIN
    low = PITCH_MAX
    has_notes = False
    for pitch in part.pitches:
        has_notes = True
        p = int(pitch.ps)
        if p < low:
            low = p
        if p > high:
            high = p
    if not has_notes:
        print(part.partName)
        print('A part has no pitches.')
        exit(1)
    return PitchRange(low, high)

def get_average_pitch(part: m21.stream.Part) -> float:
    """Get the mean pitch in a part."""
    pitches = []
    for pitch in part.pitches:
        pitches.append(pitch.ps)
    return float(np.mean(pitches))

def find_tranposed_options(stream: m21.stream.Stream, new_parts: tuple[NewPart, ...], transpose: int) -> tuple[tuple[Transposition, ...], ...] | None:
    """Find all options for transposition in a given key for given arrangement parts."""
    transposed_stream = stream.transpose(transpose)
    transposed_parts = []
    for part in transposed_stream.parts:
        part_range = get_part_range(part)
        choices = []
        for i in part_range.transposable_pitches():
            # is it transposed by an octave?
            if i % 12 == 0:
                transposed_range = part_range + i
                bitmask = 0
                for x, new_part in enumerate(new_parts):
                    if transposed_range in new_part.range:
                        bitmask |= 1 << x
                if bitmask:
                    t = i + transpose
                    choices.append(Transposition(t, bitmask))
        if len(choices) == 0:
            # cannot map a part
            return None
        transposed_parts.append(tuple(choices))
    return tuple(transposed_parts)

def iterate_tranposed_options(stream: m21.stream.Stream, new_parts: tuple[NewPart, ...], transpose: int) -> Generator[Generator[Transposition, None, None], None, None]:
    """Iterate over all transposition options."""
    transposed_parts = find_tranposed_options(stream, new_parts, transpose)
    if transposed_parts is None:
        # cannot map a part
        return
    # counters to iterate through each possibility
    counters = [0 for _ in range(len(transposed_parts))]

    while True:
        yield (part[counters[index]] for index, part in enumerate(transposed_parts))
        # increment counters
        for i in range(len(transposed_parts)):
            counters[i] += 1
            if counters[i] == len(transposed_parts[i]):
                counters[i] = 0
            else:
                break
        else:
            # all combinations tried, get out
            return

# cached to save time with duplicate calls
@cache
def find_unique(count: int, bitmasks: tuple[int, ...], allowed: int) -> bool:
    """Check if there exists a valid arrangement."""
    # check all parts
    for bit in range(count):
        # check a part
        mask = 1 << bit
        # can be played and not yet covered?
        if bitmasks[0] & mask & allowed:
            # check the remaining parts
            if len(bitmasks) == 1 or find_unique(count, bitmasks[1:], allowed & ~mask):
                return True
    return False

def run_transposed(stream: m21.stream.Stream, new_parts: tuple[NewPart, ...], transpose: int) -> Generator[Candidate, None, None]:
    """Check all valid candidates for a given transposition."""
    test_bitmask = (1 << len(new_parts)) - 1
    bitmasks = []
    part_selection = []
    for option in iterate_tranposed_options(stream, new_parts, transpose):
        full_bitmask = 0
        bitmasks.clear()
        part_selection.clear()
        for transposition in option:
            full_bitmask |= transposition.bitmask
            bitmasks.append(transposition.bitmask)
            part_selection.append(transposition.deviation)
        # can every part be played?
        if full_bitmask == test_bitmask:
            # sort bitmasks for better cache hits on the find_unique() call -
            # the order doesn't matter in this check
            bitmasks.sort()
            # test that there exists a valid arrangement
            if find_unique(len(bitmasks), tuple(bitmasks), test_bitmask):
                # if so, check this candidate
                yield Candidate(tuple(part_selection))

def get_all_choices(stream: m21.stream.Stream, new_parts: tuple[NewPart, ...]) -> Generator[tuple[int, Generator[Candidate, None, None]], None, None]:
    """Get all candidates and the number of sharps/flats for each set of candidates."""
    # only do transposes that aren't more than half an octave from center -
    # we'll transpose parts all over the staves within the inner loop
    for i in range(-6, 6):
        # figure out how many sharps are there are in total
        num_sharps = 0
        for new_part in new_parts:
            num_sharps += abs(new_part.test_key.transpose(i).sharps)
        yield (num_sharps, run_transposed(stream, new_parts, i))

def find_best_choice(choices: Generator[Candidate, None, None]) -> Candidate | None:
    """Find the best choice from a stream of candidates, or None if no candidates are available."""
    best_choice = None
    best_deviation = inf
    for choice in choices:
        # check deviation
        deviation = choice.deviation()
        if deviation < best_deviation:
            # select if best choice
            best_deviation = deviation
            best_choice = choice
    return best_choice

def find_best_choice_overall(stream: m21.stream.Score, new_parts: tuple[NewPart, ...]) -> Candidate | None:
    """Find the best choice for arranging a stream for new parts."""
    best_choice = None
    best_deviation = inf
    best_sharps = inf
    for num_sharps, choices in get_all_choices(stream, new_parts):
        # only try candidates that will not have more sharps than we currently have
        if num_sharps <= best_sharps:
            # find best choice from stream
            this_best_choice = find_best_choice(choices)
            if this_best_choice is not None:
                # found a choice
                this_best_deviation = this_best_choice.deviation()
                # candidate overpowers previous if less sharps or less transposition
                if num_sharps < best_sharps or this_best_deviation < best_deviation:
                    best_choice = this_best_choice
                    best_deviation = this_best_deviation
                    best_sharps = num_sharps
    return best_choice

def run_new_algo(stream: m21.stream.Stream, arragement_filename: str, output_basename: str):
    """Run the algorithm."""
    # load the parts
    new_parts = load_parts(arragement_filename)
    if len(new_parts) != len(stream.parts):
        print('The number of parts does not match.')
        exit(1)
    # get best choice from all choices
    print('Testing arrangements...')
    best_choice = find_best_choice_overall(stream, new_parts)
    if best_choice is None:
        print('No arrangements possible.')
        return
    print('Found arrangement with deviation {}.'.format(best_choice.deviation()))
    # transpose parts - find ranges and average pitch
    part_ranges = []
    for transposition, part in zip(best_choice.transpositions, stream.parts.stream()):
        part.transpose(transposition, inPlace=True)
        part_ranges.append((get_part_range(part), get_average_pitch(part)))
    # try every permutation to find a set of instruments that fits
    # we're guaranteed to find one here
    best_fit = inf
    best_permutation = None
    print('Testing distributions...')
    for permutation in permutations(new_parts):
        fit = 0.0
        for old_part_data, new_part in zip(part_ranges, permutation):
            old_part_range, old_part_average = old_part_data
            if old_part_range not in new_part.range:
                break
            fit += abs(old_part_average - new_part.range.mid())
        else:
            if fit < best_fit:
                best_fit = fit
                best_permutation = permutation
    print('Found distribution with fit {}.'.format(best_fit))
    for i, p in enumerate(stream.parts.stream()):
        p.partName = None
        p.partAbbreviation = None
        for s in p:
            if isinstance(s, m21.instrument.Instrument):
                p.remove(s)
                p.insert(best_permutation[i].instrument)
    stream.write('musicxml', output_basename)

# set up the environment
if platform == 'win32':
    # Windows
    path = 'C:/Program Files/MuseScore 3/bin/Musescore3.exe'
elif platform == 'darwin':
    # Mac OS - TODO
    pass
else:
    # assume Linux
    path = '/usr/bin/musescore'

env = m21.environment.Environment()
env['musicxmlPath'] = path

argc = len(argv)
if argc < 4:
    print('arguments: [input file] [arragement file] [output name (no extension)]')
else:
    song = m21.converter.parse(argv[1])
    run_new_algo(song, argv[2], argv[3])
