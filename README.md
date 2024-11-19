Wavetable Manifold Synthesis (wms), formally called Wavetable-Inspired Artificial Neural Network Synthesis (wtianns), is a form of sound synthesis closely related to wavetable synthesis, except that instead of storing wavetables in memory, they are generated in realtime by a neural network. The network for WMS should be trained using timbral features as inputs, and the corresponding magnitude spectra as outputs. This means that after successful training, the user can control the timbral content of the wavetables in realtime based on the timbral features used.

[nvssynthesis](https://github.com/nvssynthesis/) currently maintains 2 repos for WMS: [wms_torch](https://github.com/nvssynthesis/wms_torch) (for training with [PyTorch](https://github.com/pytorch/pytorch)), and [wavetable_manifold_synthesizer](https://github.com/nvssynthesis/wavetable_manifold_synthesizer) (a realtime synthesizer plugin using [JUCE](https://github.com/juce-framework/JUCE)). wms_torch also manages storing the computed data (i.e. the timbral features and spectra) for each set of raw training data (directories of audio files, or, if you wish, individual (but preferably quite lengthy) audio files). 

At this time, the architecture used is a Gated Recurrent Unit (GRU) network, allowing some interesting qualities like time dependence and hysteresis of the synthesizer's timbre. You may tweak this if you wish; however, the WMS synthesizer plugin statically assumes a particular architecture corresponding to the one outlined in ./params.json. Note that this architecture is subject to change, but in general, the two repos will change in a matching manner.

To train:

`git clone https://github.com/nvssynthesis/wms_torch.git`

`cd wtms_torch`

`python3 -m venv ./venv`

`source ./venv/bin/activate`

`pip install -r requirements.txt`

then, you want to set up the program to be trained on audio of your choosing. to do this, open params.json.

Edit the path for the entry 'audio_files_path'. it has only been tested using relative paths, so you may want to 
put your desired folder into this directory. also it has only been tested with .wav files.

Then, you should be able to train by running
`python3 ./train.py`

Training will take a long time, depending on, among other things, num_epochs in params.json, how many audio files, how long they are, and the hop size.

Once it's done training, you can test the network's ability to make inferences by running 
`python3 ./predict.py`
which will display predicted spectra and waveforms superimposed with their respective targets. It will also optionally export (very short) audio files for each prediction/target pair. 