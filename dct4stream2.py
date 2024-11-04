from dataclasses import dataclass
import numpy as np
import scipy.fftpack
import scipy.signal.windows
import soundfile
import matplotlib.pyplot as plt
import time

@dataclass
class DCT4StreamConfig:
    fft_size: int = 256
    block_size: int = 256
    block_step: int = 192


DEFAULT_DCT4_CFG = DCT4StreamConfig()



class AudioInputDCT4:
    def __init__(self, filename: str, config: DCT4StreamConfig = DEFAULT_DCT4_CFG):
        self.fft_step = int(config.fft_size / 2)
        self.io_blocksize = config.fft_size + (config.block_size - 1) * self.fft_step
        io_step = self.fft_step * config.block_step
        self.io_overlap = self.io_blocksize - io_step
        self.config = config

        with soundfile.SoundFile(filename) as sf:
            self.num_channels = sf.channels

        self.block_shape = (self.config.block_size, self.config.fft_size, self.num_channels)

        self.window_function = scipy.signal.windows.blackmanharris(config.fft_size)
        self.window_function_block = np.array([np.stack([self.window_function] * self.num_channels, axis=-1)] * 256)

        self.sndfile = soundfile.blocks(
            file=filename, 
            blocksize=self.io_blocksize, 
            overlap=self.io_overlap,
            dtype='float32',
            always_2d=True
        )

    def read_block(self):
        time_domain_block = next(self.sndfile).copy()
        time_domain_block.resize((self.io_blocksize, self.num_channels))
        half_block_shape = (int(self.config.block_size/2), self.config.fft_size, self.num_channels)

        even_chunks = np.reshape(time_domain_block[:-self.fft_step, :], half_block_shape)
        odd_chunks = np.reshape(time_domain_block[self.fft_step:, :], half_block_shape)
        full_block_td = np.stack((even_chunks, odd_chunks), axis=1)
        full_block_td = np.reshape(full_block_td, self.block_shape)

        full_block_dct = scipy.fftpack.dct(full_block_td * self.window_function_block, type=4, norm='ortho', axis=1)

        return full_block_dct

    def __iter__(self):
        return self

    def __next__(self):
        return self.read_block()
    



class AudioOutputDCT4:
    def __init__(self, filename: str, config: DCT4StreamConfig = DEFAULT_DCT4_CFG, samplerate: int = 44100, num_channels: int = 2):
        self.config = config
        self.samplerate = samplerate
        self.num_channels = num_channels
        self.block_shape = (self.config.block_size, self.config.fft_size, self.num_channels)
        self.block_buffers = [np.zeros(self.block_shape)] * 3
        self.sndfile = soundfile.SoundFile(filename, mode='w', samplerate=self.samplerate, channels=self.num_channels)
        self.num_blocks_written = 0

        self.half_block_size = int(config.block_size / 2)

        self.window_function_half_block = np.tile(self._calc_window_function(), (self.config.block_size, 1))
        self.window_compensation = np.tile(self._calc_window_compensation(), (self.config.block_size, 1))
    
    def _calc_window_function(self):
        win_bh = scipy.signal.windows.blackmanharris(self.config.fft_size)
        return np.stack([win_bh] * self.num_channels, axis=-1)
        

    def _calc_window_compensation(self):
        win_bh = self._calc_window_function()
        sq_win_bh = win_bh*win_bh
        r_sq_win_bh = np.roll(sq_win_bh, int(self.config.fft_size/2), axis=0)
        return 1.0 / (sq_win_bh + r_sq_win_bh)



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write_block(np.zeros(self.block_shape))
        self.sndfile.close()

    def write_block(self, block):
        t0 = time.time()
        fft_step = int(self.config.fft_size/2)

        self.block_buffers[0] = self.block_buffers[1]
        self.block_buffers[1] = self.block_buffers[2]
        self.block_buffers[2] = block

        ab_reg_begin = [i*self.config.block_step for i in range(3)]
        ab_reg_end = [self.config.block_size + ab_reg_begin[i] for i in range(3)]
        active_block_width = ab_reg_end[2]

        active_block = np.zeros((active_block_width, self.config.fft_size, self.num_channels))
        half_overlap = int((self.config.block_size - self.config.block_step) / 2)

        active_block[ab_reg_begin[0]:ab_reg_end[0],:,:] = self.block_buffers[0]
        active_block[ab_reg_begin[2]:ab_reg_end[2],:,:] = self.block_buffers[2]
        active_block[ab_reg_begin[1]+half_overlap:ab_reg_end[1]-half_overlap,:,:] = self.block_buffers[1][half_overlap:-half_overlap,:,:]

        roi = active_block[ab_reg_begin[1]-1:ab_reg_begin[2]+1, :, :]

        even_chunks = roi[0::2,:,:].copy()   # [chunk, bin, channel]
        odd_chunks = roi[1::2,:,:].copy()

        half_td_num_frames = even_chunks.shape[0] * even_chunks.shape[1]
        td_block_num_frames = half_td_num_frames + fft_step

        one_half_td_block_shape = (half_td_num_frames,2)
        full_td_block_shape = (td_block_num_frames,2)

        even_chunks_td = np.reshape(scipy.fftpack.dct(even_chunks, type=4, norm='ortho', axis=1), one_half_td_block_shape)
        odd_chunks_td = np.reshape(scipy.fftpack.dct(odd_chunks, type=4, norm='ortho', axis=1), one_half_td_block_shape)

        even_chunks_td *= self.window_function_half_block[:half_td_num_frames, :]
        odd_chunks_td *= self.window_function_half_block[:half_td_num_frames, :]

        time_domain_block = np.zeros(full_td_block_shape)
        time_domain_block[:-fft_step, :] += even_chunks_td[:, :]
        time_domain_block[fft_step:, :] += odd_chunks_td[:, :]

        time_domain_block *= self.window_compensation[:td_block_num_frames, :]

        if self.num_blocks_written > 0:
            self.sndfile.write(time_domain_block[2*fft_step:-fft_step, :].copy())
        self.num_blocks_written += 1
        # t1 = time.time()
        # print("wrote block of {} frames in {} ms".format(time_domain_block[fft_step:-fft_step, :].shape[0], 1000.0 * (t1-t0)))











