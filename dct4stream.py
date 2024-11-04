from dataclasses import dataclass
import numpy as np
import scipy.fftpack
import soundfile
import matplotlib.pyplot as plt
import time

@dataclass
class DCT4StreamConfig:
    fft_size: int = 256
    step_size: int = 128
    block_size: int = 256


DEFAULT_DCT4_CFG = DCT4StreamConfig()

class AudioInputDCT4:
    def __init__(self, filename: str, config: DCT4StreamConfig = DEFAULT_DCT4_CFG):
        self.filename = filename
        self.config = config
        blocksize = config.fft_size
        overlap = config.fft_size-config.step_size
        self.generator = soundfile.blocks(filename, blocksize=blocksize, overlap=overlap, always_2d=True, dtype='float32')
        self.num_channels = 2
        self.end = False


    def read_window(self):
        signal = next(self.generator).copy()
        window = scipy.signal.windows.blackmanharris(signal.shape[0])
        for channel_idx in range(signal.shape[1]):
            signal[:, channel_idx] = scipy.fftpack.dct(signal[:, channel_idx]*window, type=4, norm='ortho')
        return signal

    def read_block(self):
        t0 = time.time()
        if self.end:
            raise StopIteration
        block = []
        end = False
        for win_idx in range(self.config.block_size):
            if end:
                block.append(np.zeros((self.config.fft_size, self.num_channels)))
                continue
            try:
                wnd = self.read_window()
                if wnd.shape[0] < self.config.fft_size:
                    wnd = wnd.copy()
                    wnd.resize((self.config.fft_size, self.num_channels))
                block.append(wnd)
            except StopIteration:
                end = True
        if end:
            self.end = True
        t1 = time.time()
        print("reading block took {} ms".format(1000.0 * (t1-t0)))
        return np.array(block, dtype=np.float32)

    def __iter__(self):
        return self

    def __next__(self):
        return self.read_block()


def shift_0pad(arr: np.ndarray, n: int):
    r_prev = tuple(np.clip((-len(arr)+n, n), 0, len(arr)))
    r_next = tuple(np.clip((len(arr)+n, 2*len(arr)+n), 0, len(arr)))
    result = np.roll(arr, n)
    # print(r_prev)
    # print(r_next)
    result[slice(*r_prev)] *= 0
    result[slice(*r_next)] *= 0
    return result


def mono_to_n_channels(arr: np.ndarray, num_ch: int):
    return np.repeat(np.expand_dims(arr, 1), num_ch, axis=1)

class AudioOutputDCT4:
    def __init__(self, filename: str, config: DCT4StreamConfig = DEFAULT_DCT4_CFG):
        self.filename = filename
        self.config = config
        self.num_channels = 2
        self.output_pos = 0
        self.io_pos = 0
        self.next_output_window = (0, self.config.fft_size)
        self.sndfile = soundfile.SoundFile(self.filename, mode='w', samplerate=44100, subtype='FLOAT', channels=self.num_channels)
        self.output_buffer = np.zeros((1, self.num_channels))
        self.win_triangle = scipy.signal.windows.triang(self.config.fft_size)
        self.win_triangle_ch = mono_to_n_channels(self.win_triangle, self.num_channels)
        self.win_bharris = scipy.signal.windows.blackmanharris(self.config.fft_size)
        self.win_prod = self.win_triangle * self.win_bharris
        self.block_compensation = self.calc_block_compensation()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sndfile.close()

    def overlap_add(self, win_time_domain: np.ndarray, begin: int):
        wtd_len = win_time_domain.shape[0]
        buf_begin = begin - self.io_pos
        buf_end = buf_begin + wtd_len
        if buf_end > self.output_buffer.shape[0]:
            self.output_buffer = self.output_buffer.copy()
            self.output_buffer.resize((buf_end, self.num_channels))
        self.output_buffer[buf_begin:buf_end, :] += win_time_domain * self.win_triangle_ch

    def calc_block_compensation(self):
        total = np.zeros(self.config.fft_size)
        offset = 0
        while offset + self.config.fft_size > 0:
            offset -= self.config.step_size
        while offset < self.config.fft_size:
            total += shift_0pad(self.win_prod, offset)
            offset += self.config.step_size
        result = mono_to_n_channels(1.0 / total, self.num_channels)
        # plt.plot(result[:, 0])
        # plt.show()
        return result

    def save_window(self):
        if self.io_pos + self.config.fft_size >= self.output_pos:
            return False
        comp = self.block_compensation
        # print("io_pos = {}".format(self.io_pos))
        if self.io_pos % self.config.fft_size != 0:
            comp = np.roll(comp, -(self.io_pos % self.config.fft_size))
        out_block = self.output_buffer[:self.config.fft_size, :].copy()
        out_block *= comp[:len(out_block), :]
        self.sndfile.write(out_block)
        self.output_buffer = self.output_buffer[self.config.fft_size:, :]
        # print("out buf size = {}".format(self.output_buffer.shape[0]))
        self.io_pos += len(out_block)
        return True

    def write_window(self, window: np.ndarray):
        td = window.copy()
        for ch_idx in range(td.shape[1]):
            td[:, ch_idx] = scipy.fftpack.dct(td[:, ch_idx], type=4, norm='ortho')
        self.overlap_add(td, self.output_pos)

        self.output_pos += self.config.step_size
        while self.save_window():
            pass

    def write_block(self, block):
        for window in block:
            self.write_window(window)

