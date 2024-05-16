to use:

`git clone https://github.com/nvssynthesis/wtiann_torch.git`

`cd wtiann_torch`

`python3 -m venv ./venv`

`source ./venv/bin/activate`

`pip install -r requirements.txt`

then, you want to set up the program to be trained on audio of your choosing. to do this, open params.json.

edit the path for the entry 'audio_files_path'. it has only been tested using relative paths, so you may want to 
put your desired folder into this directory. also it has only been tested with .wav files.

then, you should be able to train by running
`python3 ./main.py`

this will take a while, depending on, among other things, num_epochs in params.json, how many audio files, how long they are, and the hop size.

once it's done training, you can make inferences by running 
`python3 ./predict.py`