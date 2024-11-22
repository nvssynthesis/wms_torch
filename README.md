Wavetable Manifold Synthesis (WMS), formerly called Wavetable-Inspired Artificial Neural Network Synthesis (WTIANNS), is a form of sound synthesis closely related to wavetable synthesis, except that instead of storing wavetables in memory, they are generated in realtime by a neural network. The network for WMS should be trained using timbral features as inputs, and the corresponding magnitude spectra as outputs. This means that after successful training, the user can control the timbral content of the wavetables in realtime based on the timbral features used.

[nvssynthesis](https://github.com/nvssynthesis/) currently maintains 2 repos for WMS: [wms_torch](https://github.com/nvssynthesis/wms_torch) (for training with [PyTorch](https://github.com/pytorch/pytorch)), and [wavetable_manifold_synthesizer](https://github.com/nvssynthesis/wavetable_manifold_synthesizer) (a realtime synthesizer plugin using [JUCE](https://github.com/juce-framework/JUCE)). wms_torch also manages storing the computed data (i.e. the timbral features and spectra) for each set of raw training data (directories of audio files, or, if you wish, individual (but preferably quite lengthy) audio files). 

At this time, the architecture used is a Gated Recurrent Unit (GRU) network, allowing some interesting qualities like time dependence and hysteresis of the synthesizer's timbre. You may tweak this if you wish; however, the WMS synthesizer plugin statically assumes a particular architecture corresponding to the one outlined in ./params.json. Note that this architecture is subject to change, but in general, the two repos will change in a matching manner.

To train:

`git clone https://github.com/nvssynthesis/wms_torch.git`

`cd wtms_torch`

`python3 -m venv ./venv`

`source ./venv/bin/activate`

`pip install -r requirements.txt`

Then, you want to set up the program to be trained on audio of your choosing. To do this, open params.json. Edit the path for the entry 'audio_files_path'. It has only been tested using .wav files and relative paths, so you may want to put your desired folder into this directory. 

If you instead want to train on basic, non-aliasing waveforms (sine, triangle, saw, square) as a test, you can keep the 'audio_files_path' in its default state "./basic_waves/basic_waves.wav", but run 
`python3 ./basic_waves/make_basic_waves.py`
which will create a long audio file of these waveforms (randomly changing frequency and wave shape). 

After either selecting your custom audio folder or creating the basic waves, you can train the network by running
`python3 ./train.py`

Training will take a long time, depending on, among other things, num_epochs in params.json, how many audio files, how long they are, and the hop size. You can get an idea of how long it will take as it's running because it will continually inform of its relative progress.

Once it's done training, you can test the network's ability to make inferences by running 
`python3 ./predict.py`
which will display predicted spectra and waveforms superimposed with their respective targets. It will also optionally export (very short) audio files for each prediction/target pair. These files are not really designed to listen to in a raw manner, instead you could import them into another software and loop them. Listening to the files in a looped manner is also somewhat misleading, because the full resynthesis algorithm also involves overlap-and-add.