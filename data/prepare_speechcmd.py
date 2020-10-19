from tqdm import tqdm
import requests
import math
import os
import tarfile
import numpy as np
import librosa
import argparse
from keras.models import Sequential
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
# modified based on https://github.com/douglas125/SpeechCmdRecognition


def CustomPrepareGoogleSpeechCmd():
    _DownloadGoogleSpeechCmdV2()
    basePath = 'sd_GSCmdV2'

    GSCmdV2Categs = {'left': 0, 'right': 1, 'up': 2, 'down': 3, 'yes': 4, 'no': 5, 'on': 6, 'off': 7, 'stop': 8,
                     'go': 9}

    WAV2Numpy(basePath + '/test/')
    WAV2Numpy(basePath + '/train/')

    trainWAVs = []
    for root, dirs, files in os.walk(basePath + '/train/'):
        if np.array([root.endswith(i) for i in GSCmdV2Categs.keys()]).any():
            trainWAVs += [root + '/' + f for f in files if f.endswith('.wav.npy')]

    testWAVsREAL = []
    for root, dirs, files in os.walk(basePath + '/test/'):
        if np.array([root.endswith(i) for i in GSCmdV2Categs.keys()]).any():
            testWAVsREAL += [root + '/' + f for f in files if f.endswith('.wav.npy')]

    trainWAVlabels = [_getFileCategory(f, GSCmdV2Categs) for f in trainWAVs]
    testWAVREALlabels = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVsREAL]

    return np.array(trainWAVs), np.array(trainWAVlabels), np.array(testWAVsREAL), np.array(testWAVREALlabels)


def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_GSCmdV2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ,0)


def _DownloadGoogleSpeechCmdV2(forceDownload=False):
    """
    Downloads Google Speech commands dataset version 2
    """
    if os.path.isdir("sd_GSCmdV2/") and not forceDownload:
        print('Google Speech commands dataset version 2 already exists. Skipping download.')
    else:
        if not os.path.exists("sd_GSCmdV2/"):
            os.makedirs("sd_GSCmdV2/")
        trainFiles = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
        testFiles = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
        _downloadFile(testFiles, 'sd_GSCmdV2/test.tar.gz')
        _downloadFile(trainFiles, 'sd_GSCmdV2/train.tar.gz')

    # extract files
    if not os.path.isdir("sd_GSCmdV2/test/"):
        _extractTar('sd_GSCmdV2/test.tar.gz', 'sd_GSCmdV2/test/')

    if not os.path.isdir("sd_GSCmdV2/train/"):
        _extractTar('sd_GSCmdV2/train.tar.gz', 'sd_GSCmdV2/train/')



def _downloadFile(url, fName):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    wrote = 0
    print('Downloading {} into {}'.format(url, fName))
    with open(fName, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")


def _extractTar(fname, folder):
    print('Extracting {} into {}'.format(fname, folder))
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=folder)
        tar.close()
    elif (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall(path=folder)
        tar.close()


def WAV2Numpy(folder, sr=None):
    """
    Recursively converts WAV to numpy arrays. Deletes the WAV files in the process

    folder - folder to convert.
    """
    allFiles = []
    for root, dirs, files in os.walk(folder):
        allFiles += [os.path.join(root, f) for f in files if f.endswith('.wav')]

    for file in tqdm(allFiles):
        y, sr = librosa.load(file, sr=None)

        # if we want to write the file later
        # librosa.output.write_wav('file.wav', y, sr, norm=False)

        np.save(file + '.npy', y)
        os.remove(file)


def load_audio(data, dim):
    output = []
    for i in data:
        audio = np.load(i)
        if audio.shape[0] == dim:
            output.append(audio)
        elif audio.shape[0] > dim:
            randPos = np.random.randint(audio.shape[0] - dim)
            output.append(audio[randPos:randPos + dim])
        else:
            randPos = np.random.randint(dim - audio.shape[0])
            temp = np.zeros(dim)
            temp[randPos:randPos + audio.shape[0]] = audio
            output.append(temp)
    return np.vstack(output)


def prepareOODtrigger(category='one'):
    # note category could be set to 'zero' to 'nine' (string)
    _DownloadGoogleSpeechCmdV2()
    basePath = 'sd_GSCmdV2'

    WAV2Numpy(basePath + '/test/')
    WAV2Numpy(basePath + '/train/')

    trainWAVs = []
    for root, dirs, files in os.walk(basePath + '/train/'):
        if np.array([root.endswith(category)]).any():
            trainWAVs += [root + '/' + f for f in files if f.endswith('.wav.npy')]

    return np.array(trainWAVs)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = CustomPrepareGoogleSpeechCmd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--OOD', help='one to nine', type=str, default='None')
    args = parser.parse_args()
    ood_source = args.OOD

    sr = 16000
    iLen = 16000

    melspecModel = Sequential()
    melspecModel.add(Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                                    padding='same', sr=sr, n_mels=80,
                                    fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                                    return_decibel_melgram=True, trainable_fb=False,
                                    trainable_kernel=False,
                                    name='mel_stft'))
    melspecModel.add(Normalization2D(int_axis=0))
    print("Converting audios to melspectrum")
    x_train = melspecModel.predict(load_audio(x_train, iLen).reshape((-1, 1, iLen)))
    x_test = melspecModel.predict(load_audio(x_test, iLen).reshape((-1, 1, iLen)))
    np.save("sd_GSCmdV2/x_train.npy", x_train)
    np.save("sd_GSCmdV2/y_train.npy", y_train)
    np.save("sd_GSCmdV2/x_test.npy", x_test)
    np.save("sd_GSCmdV2/y_test.npy", y_test)

    if ood_source != 'None':
        ood_trigger = prepareOODtrigger(ood_source)
        trigger = melspecModel.predict(load_audio(ood_trigger, iLen).reshape((-1, 1, iLen)))
        np.save("../image/data/sd_GSCmdV2/trigger.npy", trigger)

    print("Preprocessing finished")


