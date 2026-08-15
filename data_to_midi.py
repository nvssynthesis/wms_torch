import mido
from util import pitch_log_to_lin_scale
from librosa import hz_to_midi


def scale_to_7bit_mfcc(value):
    # for mfccs, we assume they are standardized to have mean 0 and std 1
    # we clip to 3 std deviations, and then scale to 0-127
    clipped = max(min(value, 3), -3)
    scaled = int((clipped + 3) / 6 * 127)
    return scaled


def scale_to_7bit_pitch(value, f_low):
    # for pitch we need to actually get to absolute midi note number, but the incoming value
    # is compressed using `pitch_log_to_lin_scale`
    lin = pitch_log_to_lin_scale(value, f_low)
    midi_note = hz_to_midi(lin)
    midi_note = int(max(min(midi_note, 127), 0))  # clip to MIDI range
    return midi_note

def scale_to_7bit_voicedness(value):
    # for pitch confidence we assume it is already in the range [0, 1]
    # we scale to 0-127
    scaled = int(max(min(value, 1), 0) * 127)
    return scaled


def write_data_to_midi(normalized_data, f_low, include_voicedness, hop_size, midi_filename: str):
    # input will be formatted as torch.cat((mfcc, pitch)

    num_features = normalized_data.shape[1]

    if include_voicedness:
        pitch_dim = num_features - 2
        voicedness_dim = num_features - 1
    else:
        pitch_dim = num_features - 1
        voicedness_dim = None


    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    starting_cc = 20  # arbitrary starting CC number for the first feature

    synthesis_update_rate_ms = 10   # could change depending on timer of synthesis engine
    playback_tempo_bpm = 120  # this is used just in testing, so it's something we can control for
    # ticks per frame depends on playback tempo, synthesis update rate, and hop size that was used in feature extraction
    PPQ = 480  # standard MIDI PPQ
    ticks_per_frame = round((PPQ * playback_tempo_bpm / 60) * (synthesis_update_rate_ms / 1000))
    resulting_midi_file_playback_length_ms = ticks_per_frame * normalized_data.shape[0] * (60 / playback_tempo_bpm) * 1000 / PPQ

    target_update_hz = 30
    frame_stride = max(1, round((1000 / synthesis_update_rate_ms) / target_update_hz))
    # e.g. synthesis_update_rate_ms=10 -> 100Hz native rate -> stride = round(100/30) = 3


    last_vals = {}
    accumulated_ticks = 0

    for frame_idx, frame in enumerate(normalized_data):
        if frame_idx % frame_stride != 0:
            accumulated_ticks += ticks_per_frame
            continue

        accumulated_ticks += ticks_per_frame
        first_message_in_frame = True

        for i, (cc_num, value) in enumerate(zip(range(starting_cc, starting_cc + num_features), frame)):
            if (cc_num - starting_cc) == pitch_dim:
                midi_val = scale_to_7bit_pitch(value, f_low)
            elif voicedness_dim is not None and (cc_num - starting_cc) == voicedness_dim:
                midi_val = scale_to_7bit_voicedness(value)
            else:
                midi_val = scale_to_7bit_mfcc(value)

            if last_vals.get(cc_num) == midi_val:
                continue  # skip redundant message
            last_vals[cc_num] = midi_val

            delta = accumulated_ticks if first_message_in_frame else 0
            first_message_in_frame = False
            accumulated_ticks = 0  # consumed by this message

            track.append(mido.Message('control_change', control=cc_num, value=midi_val, time=delta))


    mid.save(midi_filename)
    print(f'Wrote MIDI file to {midi_filename}, which will play back in {resulting_midi_file_playback_length_ms:.2f} ms at {playback_tempo_bpm} BPM')