from numpy import rec
from speechbrain.pretrained import EncoderClassifier
import sounddevice as sd
from scipy.io.wavfile import write

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

def record():
    try:                
        fs = 44100  # this is the frequency sampling; also: 4999, 64000
        seconds = 5  # Duration of recording
        
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        print("Starting: Speak now!")
        
        sd.wait()  # Wait until recording is finished
        
        print("Listening \nFinished")
        write('output.mp3', fs, myrecording)  # Save as WAV file


        global signal
        signal = language_id.load_audio("output.mp3")

    except Exception as e:
        assert False, "Some error"

    return True

def predict():
    try:        
        prediction =  language_id.classify_batch(signal)
        
        print(prediction[3])
        print("...")


    except Exception as e:
        assert False, "Some error"


for i in range(2):
    record()
    predict()
