import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
import librosa

def calculate_t60(audio_file):
    ir, fs = librosa.load(audio_file, sr=None, mono=False)
    # print(np.max(ir, axis=1))
    # ir[0] = ir[0] / np.max(ir[0])
    # ir[1] = ir[1] / np.max(ir[1])
    # sf.write('normed.wav', ir.T, fs, format='wav')
    # return 0
    # print(np.any(ir<0))
    # Calculate the envelope of the impulse response
    envelope = np.abs(librosa.amplitude_to_db(ir))
    # print(envelope)
    # Find the peak amplitude
    peak_amplitude = np.max(envelope, axis=1)
    # print(peak_amplitude)
    # Find the time index when the envelope decays by 60 dB
    t60_index_0 = np.argmax(envelope[0] < peak_amplitude[0] - 60)
    t60_index_1 = np.argmax(envelope[1] < peak_amplitude[1] - 60)
    # print(t60_index_0)
    # print(t60_index_1)
    # Calculate the time duration (in seconds) for the envelope to decay by 60 dB
    t60 = t60_index_0 / fs
    t60_1 = t60_index_1 / fs
    
    return np.array([t60, t60_1])

def compute_T60_error(audio_file1, audio_file2):
    t60_1 = calculate_t60(audio_file1)
    t60_2 = calculate_t60(audio_file2)
    print(t60_1)
    print(t60_2)
    return np.abs(t60_1 - t60_2)

def compute_mae(audio_file1, audio_file2):
    # Read audio files
    audio1, sr1 = sf.read(audio_file1)
    audio2, sr2 = sf.read(audio_file2)
    # print(audio1)

    # Ensure that both audio signals have the same length
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    # Compute MAE for each channel
    mae_per_channel = []
    for channel1, channel2 in zip(audio1.T, audio2.T):
        # channel1 = channel1 / np.max(channel1)
        # channel2 = channel2 / np.max(channel2)
        mae = np.mean(np.abs(channel1 - channel2))
        mae_per_channel.append(mae)

    return mae_per_channel

def diff_between_model_human():
    # Paths to the audio files
    model_dir = 'Output2/scene0000_02/S1/'
    human_dir = 'Output_Human/scene0000_02/S1/'
    all_mae = np.zeros((2, 1000))
    for file_cnt in tqdm(range(1, 1001)):
        model_file = f'{model_dir}{file_cnt}.wav'
        human_file = f'{human_dir}{file_cnt}.wav'
        # print(f'T60:{calculate_t60(human_file)}')
        # Compute MAE between the two audio files for each channel
        mae_per_channel = compute_mae(model_file, human_file)
        # t60_error_per_channel = compute_T60_error(model_file, human_file)
        # print(mae_per_channel)
        all_mae[:,file_cnt-1] = mae_per_channel
        # for i, mae in enumerate(mae_per_channel):
        #     print(f'Mean Absolute Error (MAE) for channel {i+1}: {mae}')
        # break
    # average_error = np.average(all_mae, axis=0)
    print(f'average MAE error: {np.average(all_mae, axis=1)}')
    plt.plot(range(1, 1001), all_mae[0,:], label = 'left')
    plt.plot(range(1, 1001), all_mae[1,:], label = 'right')
    plt.title('MAE error of 1000 BIRs')
    plt.xlabel('BIRs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('mae_comparison_new.png')
    
def how_realtime_should_be():
    model_dir = 'Output2/scene0000_02/S1/'
    # human_dir = 'Output_Human/scene0000_02/S1/'
    all_mae = np.zeros((2, 20))
    for file_cnt in tqdm(range(1, 21)):
        model_file = f'{model_dir}{file_cnt}.wav'
        base_file = f'{model_dir}{1}.wav'
        # print(f'T60:{calculate_t60(human_file)}')
        # Compute MAE between the two audio files for each channel
        mae_per_channel = compute_mae(model_file, base_file)
        # t60_error_per_channel = compute_T60_error(model_file, base_file)
        # print(t60_error_per_channel)
        all_mae[:,file_cnt-1] = mae_per_channel
        # for i, mae in enumerate(mae_per_channel):
        #     print(f'Mean Absolute Error (MAE) for channel {i+1}: {mae}')
        # break
    print(f'average MAE: {np.average(all_mae, axis=1)}')
    plt.plot(range(1, 21), all_mae[0,:], label = 'left')
    plt.plot(range(1, 21), all_mae[1,:], label = 'right')
    plt.title('MAE of 20 BIRs')
    plt.xlabel('BIRs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('mae_realtime_comparison_20.png')
    
if __name__=='__main__':
    diff_between_model_human()