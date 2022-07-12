# Copyright (c) 2021 ralabo.jp
# This software is released under the MIT License.
# see https://opensource.org/licenses/mit-license.php
# ====================================================
import os
import sys
import numpy as np
#import sox
import soundfile as sf
from utils.tqdm_wrapper import tqdm
from common import com
from torch.utils.data import random_split
import librosa
import librosa.core
import librosa.feature

class BaseData(object):
    """
    Args:
        machine_type      : fan, gearbox, pump, slider, ToyCar, ToyTrain, valve
        com.mode          : "development": mode == True
                            "evaluation": mode == False
        section_name      : section_00, section_01, ...
        domain            : 
        target_dir        : train, target_test, source_test
        tl_augmentation   : 転移学習時にサンプル数(ファイル数)を水増しする方法
                aug_count : 水増しする倍数(この倍数だけファイルを水増し)(0は水増ししない)
                aug_gadd  : 偏差0～aug_gaddの正規乱数を重畳する
                aug_wcut  : 前後計この分の波形をランダムに削る(分析フレーム位置がずれる)
    Atributes:
        basedata_memory:
            compact: Trueなら生のフレーム系列を保持する
            extend: Falseなら特徴ベクトル列に展開して保持する
            
    Return:
        array [num_files, n_vectors_ea_file, input_dim]
    """
    def __init__(self, machine_type=None, section_name=None,
                 domain=None, target_dir=None, file_list=None,
                 augment=None, fit_key=None, eval_key=None):

        self.augment = augment # Trueならデータ拡張を考慮する
        param = com.param
        mode = com.mode

        # basedata_memory: 'compact'なら生のフレーム系列を保持する
        # basedata_memory: 'extend'なら特徴ベクトル列に展開して保持する
        self.basedata_memory = param['misc']['basedata_memory']
        com.print(3,'basedata_memroy={}'.format(self.basedata_memory))

        data_size = param['fit']['data_size']
        include_evaluation_data = ( data_size == -2 ) # 評価用学習データも含める


        # 波形ファイルリストを取得する
        if file_list is not None:
            self.files = file_list
            self.y_true = [float('_anomaly_' in os.path.basename(x)) for x in file_list]
        else:
            machine_dir = com.machine_dir(machine_type)
            self.files, self.y_true = com.file_list_generator(machine_dir=machine_dir,
                                                              section_name=section_name,
                                                              domain=domain,
                                                              target_dir=target_dir,
                                                              include_evaluation_data=include_evaluation_data)
        # データ拡張
        if self.augment:
            self.aug_count = param[fit_key]['aug_count']
            self.aug_gadd = param[fit_key]['aug_gadd']
            self.aug_wcut = param[fit_key]['aug_wcut']
            if self.aug_count>0:
                self.files = np.repeat(self.files, self.aug_count)
                self.y_true = np.repeat(self.y_true, self.aug_count)
            else:
                self.augment = 0 # reset augument

        # データサイズ制限
        if data_size > 0 and len(self.files) > data_size:
            # 2種類のラベルを含むように前半と後半を連結する
            com.print(2,'datasize: {} --> {}'.format(len(self.files), data_size))
            size = data_size//2
            self.files = np.concatenate([self.files[:size],self.files[-size:]])
            self.y_true = np.concatenate([self.y_true[:size],self.y_true[-size:]])


        # 波形ファイルを分析する(特徴抽出)
        self.n_frames =  param["feature"]["n_frames"]
        self.n_mels = param["feature"]["n_mels"]
        self.frame_size = self.n_mels * self.n_frames

        # 'compact'の場合、n_frames を'1'にして生のフレーム列を取得する
        n_frames = 1 if self.basedata_memory == 'compact' else self.n_frames
        desc = "generate dataset {} / {} {} {}{}".format(
            machine_type, target_dir, section_name, domain,
            ' x {}'.format(self.aug_count) if self.augment else '')
        self.data = self.get_features(self.files, desc, self.n_mels, n_frames,
                                      param["feature"]["n_hop_frames"],
                                      param["feature"]["n_fft"],
                                      param["feature"]["hop_length"],
                                      param["feature"]["power"])

        # self.dataの形状を[num_file, num_frames, frame_size]とする
        num_files = len(self.files)
        num_frames = int( self.data.shape[0] / num_files )
        frame_size = self.data.shape[-1]
        self.data = self.data.reshape((num_files, num_frames, frame_size))

        if self.basedata_memory == 'compact': # 'compact'の場合
            # 一つのファイル当たりの特徴ベクトル数を計算する
            frames = self.data[0] # 最初のファイルのフレーム列を取得する
            # 拡大後のフレーム数をself.n_vectorsにセットする
            self.n_vectors = len(frames) - self.n_frames + 1

    def transform_wave(self, y, sr,ahmed):
        """
        波形を変形する
        """
       # print("Hellooooooooooooooooooooooooooooooooooooo")
        if self.augment:
           # print("Hellooooooooooooooooooooooooooooooooooooo")

            # 正規乱数を加える
            
            if self.aug_gadd>0:
                gain = np.random.uniform(0, self.aug_gadd) # 利得をランダムに決める
                y += np.random.normal(scale=gain, size=len(y)) # 正規乱数を加える
                #print("Hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
                #print(y)
                #print(sr)

            # 波形の前後をカットする(結果として分析フレームの位置を変化させる)
            if self.aug_wcut>0: 
                #pre_cut = np.random.randint(0, self.aug_wcut) # 波形の頭のカット数
                #print("preeeeeeeee")
                #print(pre_cut)
                #post_cut = self.aug_wcut - pre_cut # 波形の尻のカット数
               # print("beforeeeeeeeee")
                #print(y.shape[0] / sr)
                #print("this is old  yyyyyyy")
                #print(y)
                #y = y[0 : 16000]
               
                y = librosa.effects.pitch_shift(y,sr=16000,n_steps=3.5)

               # for counter, i in enumerate(mylist, 1):
                # scipy.io.wavfile.write(filename = "file"+str(counter), rate= i[1], data=i[0])
                #if(ahmed<999):
                 #   sf.write('section_00_source_train_normal_'+str(ahmed+1000)+'_strength_1_ambient.wav', y, samplerate=16000)
                  #  print(ahmed)
                    
                #if(1<ahmed<100):
                 #   sf.write('section_00_source_train_normal_00'+str(ahmed)+'_strength_1_ambient.wav', y, samplerate=16000)
                  #  print(ahmed)
                    
               # if(100<ahmed<1002):
                #    sf.write('section_00_source_train_normal_0'+str(ahmed)+'_strength_1_ambient.wav', y, samplerate=16000)
                 #   print(ahmed)  
                       
                  
                
                
                



                #librosa.output.write_wav('pitchshift.wav', y, sr)
               # y = y[pre_cut : -post_cut] # 波形の前後をカット
                #print("poooooooost")
                #print(post_cut)
               # print("afterrrrrrrrrrrrr")
                #print("this is new  yyyyyyy")
                #print(y)
               # print("this is sr")
               # print("this is sr")
               # print(sr)
               # print(y.shape[0] / sr)
               
#Timeee shifting
               
                # shift = np.random.randint(sr * 2 )
                # direction = np.random.randint(0,2)
                # if direction == 1:
                #     shift=-shift
                # y= np.roll(y,shift)
                # if shift>0:
                #     y[:shift]=0
                # else:
                #     y[shift:]=0
                
                #shiftr = -shift


        return y, sr

    def get_features(self, files, desc, n_mels, n_frames, n_hop_frames, n_fft,
                     hop_length, power):
        """
        ファイルリストから全分析結果を連結した特徴ベクトル列を取得する
        """
        assert len(files), "no files"
        frame_size = n_mels * n_frames
        data = None # オンデマンド
        ahmed=-1
        ahmed2=1005
        ahmed3=1005
        #print("hellooooooooooooooo")
        for i, file in tqdm(enumerate(files), desc=desc):
            ahmed+=1
            features = self.file_to_features(file, n_mels, n_frames, n_fft, hop_length, power,ahmed,ahmed2,ahmed3)
            features = features[: : n_hop_frames, :]
            if data is None:
                # dataを初期化
                time_length = features.shape[0]
                data = np.zeros((len(files) * time_length, frame_size), np.float32)
            # ファイルiの分析結果をdataにコピーする
            data[time_length*i : time_length*(i+1), :] = features

        return data

    def file_to_features(self, wav_file, n_mels, n_frames, n_fft, hop_length, power,ahmed,ahmed2,ahmed3):
        """
        波形ファイルwav_fileを読んで特徴ベクトル列featuresを返す
        """
        frame_size = n_mels * n_frames
        # 波形データを読む
        y, sr = self.file_load(wav_file, mono=True)
        # データ変形(for data augumentation)
        y, sr = self.transform_wave(y, sr,ahmed,ahmed2,ahmed3)
        # メルスペクトログラムを取得する
        mel_sgram = librosa.feature.melspectrogram(y=y,
                                                   sr=sr,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   n_mels=n_mels,
                                                   power=power)
        # 常用対数に変換する
        log_mel_sgram = 20.0 / power * np.log10(np.maximum(mel_sgram, 1e-15))
        # 時間長を取得する(sgramの周波数と時間の次元が逆であることに注意)
        length = len(log_mel_sgram[0, :]) - n_frames + 1
        # log_mel_sgramを特徴ベクトル列に整形する
        features = np.zeros((length, frame_size))
        for i in range(n_frames):
            # フレームiの集団をfeaturesにコピーする
            features[:, n_mels*i : n_mels*(i+1)] = log_mel_sgram[:, i : i+length].T

        return features

    def file_load(self, wav_name, mono=False):
        """
        load .wav file.

        wav_name : str
            target .wav file
        mono : boolean
            When load a multi channels file and this param True, the returned data will be merged for mono data

        return : numpy.array( float )
        """
        try:
            return librosa.load(wav_name, sr=None, mono=mono)
        except:
            com.print(0,"file_broken or not exists!! : {}".format(wav_name))

    def split(self,validation_split):
        """
        base data を分割する

        return data_tra, data_val
        """
        num_val = int(len(self)*validation_split)
        #assert num_val,\
        #    "can't get validation sample from {} data".format(len(self))
        if num_val == 0:
            # not split, and returns the same one
            data_tra = self
            data_val = self
            com.print(2, "'val_split' is zero, data_tra is used for data_val")
        else:
            # split the data
            num_tra = len(self) - num_val
            data_tra, data_val = random_split(self, [num_tra, num_val])

        return data_tra, data_val

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        block = self.data[index]
        label = self.y_true[index]
        file_path = self.files[index]
        return block, label, file_path
