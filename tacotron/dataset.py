import os
import subprocess
import torch
import scipy
import librosa
import pickle
import numpy as np
import resampy
import re
import inflect
import csv
import sys
import random
from unidecode import unidecode
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from scipy.io import wavfile
from librosa.util import normalize

""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''


_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')


def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'    # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')



def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including number and abbreviation expansion.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]

_valid_symbol_set = set(valid_symbols)


class CMUDict:
    '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
    def __init__(self, file_or_path, keep_ambiguous=True):
        if isinstance(file_or_path, str):
            with open(file_or_path, encoding='latin-1') as f:
                entries = _parse_cmudict(f)
        else:
            entries = _parse_cmudict(file_or_path)
        if not keep_ambiguous:
            entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
        self._entries = entries


    def __len__(self):
        return len(self._entries)


    def lookup(self, word):
        '''Returns list of ARPAbet pronunciations of the given word.'''
        return self._entries.get(word.upper())

_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
    cmudict = {}
    for line in file:
        if len(line) and (line[0] >= 'A' and line[0] <= 'Z' or line[0] == "'"):
            parts = line.split('  ')
            word = re.sub(_alt_re, '', parts[0])
            pronunciation = _get_pronunciation(parts[1])
            if pronunciation:
                if word in cmudict:
                    cmudict[word].append(pronunciation)
                else:
                    cmudict[word] = [pronunciation]
    return cmudict


def _get_pronunciation(s):
    parts = s.strip().split(' ')
    for part in parts:
        if part not in _valid_symbol_set:
            return None
    return ' '.join(parts)


_pad        = '_'
_punctuation = '!\'(),.:;? '
_special = '-'
#_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) # + _arpabet

# #### changed version for ru text

_pad = '_'
_punctuation = ' -,.!?\'\"():;'
_special = '+'
_letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'  # АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ

# Все предопределенные символы в одном списке для сортировки
predefined_order = [_pad] + list(_punctuation) + list(_special)  # + list(_letters)

def get_unique_symbols_from_metadata(file_path, predefined_order):
    """
    Считывает файл metadata.csv и возвращает список уникальных токенов (фонем, знаков препинания, специальных символов).
    
    Args:
        file_path (str): Путь к файлу metadata.csv
        predefined_order (list): Список предопределённых символов
    
    Returns:
        list: Список уникальных символов, включающий предопределённые и дополнительные
    """
    unique_symbols = set()
    
    # Увеличиваем максимальный размер поля для CSV
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            if len(row) > 1:
                phoneme_text = row[1].strip()
                tokens = phoneme_text.split()  # Разделение по пробелам
                for token in tokens:
                    # Очистка токена от лишних символов
                    token_clean = token.strip().replace('\n', '').replace('\r', '')
                    if token_clean:
                        unique_symbols.add(token_clean)
            else:
                print(f"Строка пропущена: {row}")
    
    # Добавляем предопределённые символы
    sorted_symbols = predefined_order.copy()
    
    # Добавляем новые уникальные символы, отсортированные
    extra_symbols = sorted([s for s in unique_symbols if s not in predefined_order])
    
    return sorted_symbols + extra_symbols

# Пример использования:
file_path = '/app/TTSproject/GST-Tacotron/datasets/metadata.csv'
file_path_for_symbols = '/app/TTSproject/GST-Tacotron/datasets/metadata.csv'
unique_symbols = get_unique_symbols_from_metadata(file_path_for_symbols, predefined_order)
symbols = unique_symbols

# Создание словарей
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


# ####################   P A R A M S   #######################

class hparams:
    seed = 0

    ################################
    # Text Parameters              #
    ################################
    text_cleaners=['basic_cleaners']  #text_cleaners=['english_cleaners']

    ################################
    # Audio                        #
    ################################
    num_mels = 20
    num_freq = 513
    sample_rate = 16000
    frame_shift = 160  #256
    frame_length = 1024
    fmin = 0
    fmax = 8000
    power = 1.5
    gl_iters = 30

    ################################
    # Train                        #
    ################################
    batch_size = 32
    is_cuda = True
    pin_mem = True
    n_workers = 1
    prep = False    #False #True
    pth = 'ruRU_ruLS_lpc.pkl'      #'ruRU_ruLS_lpc.pkl'    'ruslan_lpc.pkl'
    lr = 2e-3       #2e-3
    betas = (0.9, 0.999)
    eps = 1e-6
    sch = True #True 
    sch_step = 4000 #4000
    max_iter = 400000  #200e3
    iters_per_log = 1000
    iters_per_sample = 500000
    iters_per_ckpt = 4000
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    eg_text = 'Такотрон на элписи коэффициентах на русском.'
    use_amp = False

    ################################
    # Model Parameters             #
    ################################
    n_symbols = len(symbols)
    symbols_embedding_dim = 512

    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    # Decoder parameters
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_ratio = 10
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 128

    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5

hps = hparams()


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = globals().get(name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])

def _should_keep_symbol(s):
    return s in _symbol_to_id and s != '_' and s != '~'

def text_to_sequence(text, cleaner_names):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    '''
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence

def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')



MAX_WAV_VALUE = 32768.0
_mel_basis = None

### audio functions
def load_wav(path):
    sr, wav = wavfile.read(path)
    if sr != hps.sample_rate:
        wav = resampy.resample(wav, sr, hps.sample_rate)
        sr = hps.sample_rate
    assert sr == hps.sample_rate
    return normalize(wav/MAX_WAV_VALUE)*0.95


def save_wav(wav, path):
    wav *= MAX_WAV_VALUE
    wavfile.write(path, hps.sample_rate, wav.astype(np.int16))


def spectrogram(y):
    D = _stft(y)
    S = _amp_to_db(np.abs(D))
    return S


def inv_spectrogram(S):
    S = _db_to_amp(S)
    return _griffin_lim(S ** hps.power)


def melspectrogram(y):
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return S


def inv_melspectrogram(mel):
    mel = _db_to_amp(mel)
    S = _mel_to_linear(mel)
    return _griffin_lim(S**hps.power)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex_)
    y = _istft(S_complex * angles)
    for i in range(hps.gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return np.clip(y, a_max = 1, a_min = -1)


# Conversions:
def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, pad_mode='reflect')


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    return (hps.num_freq - 1) * 2, hps.frame_shift, hps.frame_length


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    inv_mel_basis = np.linalg.pinv(_mel_basis)
    inverse = np.dot(inv_mel_basis, spectrogram)
    inverse = np.maximum(1e-10, inverse)
    return inverse


def _build_mel_basis():
    n_fft = (hps.num_freq - 1) * 2
    return librosa.filters.mel(sr=hps.sample_rate, n_fft=n_fft, n_mels=hps.num_mels, fmin=hps.fmin, fmax=hps.fmax)


def _amp_to_db(x):
    return np.log(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.exp(x)



##### dataset 
def files_to_list(fdir):
    f_list = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
            lpc_path = os.path.join(fdir, 'lpc_features', '%s.f32' % parts[0])
            
            if hps.prep:
                #f_list.append(get_mel_text_pair(parts[1], wav_path))
                f_list.append(get_lpc_text_pair(parts[1], lpc_path))
            else:
                f_list.append([parts[1], lpc_path])   #f_list.append([parts[1], wav_path])
    if hps.prep and hps.pth is not None:
        with open(hps.pth, 'wb') as w:
            pickle.dump(f_list, w)
    return f_list


class ljdataset(Dataset):
    def __init__(self, fdir):
        if hps.prep and hps.pth is not None and os.path.isfile(hps.pth):
            with open(hps.pth, 'rb') as r:
                self.f_list = pickle.load(r)
        else:
            self.f_list = files_to_list(fdir)

    def __getitem__(self, index):
        text, lpc = self.f_list[index] if hps.prep \
                    else get_lpc_text_pair(*self.f_list[index])
        return text, lpc

    def __len__(self):
        return len(self.f_list)


def get_mel_text_pair(text, wav_path):
    text = get_text(text)
    mel = get_mel(wav_path)
    return (text, mel)

def get_text(text):
    return torch.IntTensor(text_to_sequence(text, hps.text_cleaners))

def get_mel(wav_path):
    wav = load_wav(wav_path)
    return torch.Tensor(melspectrogram(wav).astype(np.float32))


# ###### lpc #######

def get_lpc_from_wav(wav_file_path):
    # Получаем имя файла без расширения
    base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
    
    # Путь для сохранения PCM файла
    pcm_file_path = f"/app/TTS/lpc/Ruslan_Audio/test_output/{base_name}.s16"
    
    # Конвертируем входной аудиофайл в 16kHz моно PCM и сохраняем результат в файл
    convert_command = f"sox {wav_file_path} -r 16000 -c 1 -t sw - > {pcm_file_path}"
    subprocess.run(convert_command, shell=True, check=True)
    
    # Путь для сохранения файла с фичами (LPC коэффициенты) в новую директорию
    features_file_path = f"./datasets/lpc_features/{base_name}.f32"
    
    # Выполняем бинарный файл для генерации файла с фичами
    dump_data_command = f"/app/TTS/LPCNet_xiph3/dump_data -test {pcm_file_path} {features_file_path}"
    subprocess.run(dump_data_command, shell=True, check=True)
    
    # Удаляем промежуточный PCM файл
    os.remove(pcm_file_path)
    
    # Читаем сгенерированный файл с фичами
    lpc_features = np.fromfile(features_file_path, dtype='float32')
    
    # Изменяем размер массива
    num_lpc = 55  # Количество коэффициентов LPC на фрейм
    
    lpc_features = lpc_features.reshape(-1, num_lpc)
    
    # Конвертируем массив в тензор PyTorch для использования в модели
    lpc_tensor = torch.Tensor(lpc_features)
    
    # Транспонируем и выбираем нужные фичи
    lpcfeatures_transposed = lpc_tensor.transpose(0, 1)
    
    lpc_tensor = torch.cat((lpcfeatures_transposed[0:18, :], lpcfeatures_transposed[36:38, :]), dim=0)
    
    return lpc_tensor

def get_lpc_from_f32(features_file_path):      #wav_file_path):
    # Получаем имя файла без расширения
    #base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
    
    # Путь для сохранения файла с фичами (LPC коэффициенты) в новую директорию
    #features_file_path = f"./datasets/lpc_features/{base_name}.f32"
    
    # Читаем сгенерированный файл с фичами
    lpc_features = np.fromfile(features_file_path, dtype='float32')
    
    # Изменяем размер массива
    num_lpc = 55  # Количество коэффициентов LPC на фрейм
    
    lpc_features = lpc_features.reshape(-1, num_lpc)
    
    # Конвертируем массив в тензор PyTorch для использования в модели
    lpc_tensor = torch.Tensor(lpc_features)
    
    # Транспонируем и выбираем нужные фичи
    lpcfeatures_transposed = lpc_tensor.transpose(0, 1)
    
    lpc_tensor = torch.cat((lpcfeatures_transposed[0:18, :], lpcfeatures_transposed[36:38, :]), dim=0)
    
    return lpc_tensor

def get_ru_text(text):
    # Конвертируем текст в список ID
    #sequence = [_symbol_to_id[s] for s in text if s in _symbol_to_id]
    sequence = [_symbol_to_id[s.lower()] for s in text if s.lower() in _symbol_to_id]
    
    # Преобразуем список ID в тензор
    sequence_tensor = torch.tensor(sequence, dtype=torch.int64)
    
    return sequence_tensor

def tensor_to_ru_text(tensor):
    # Конвертируем тензор в список символов
    text = ''.join([_id_to_symbol[id] for id in tensor.tolist()])
    return text

def get_ru_phonemes(text):
    """
    Конвертирует текст с фонемами в тензор ID.
    Args:
        text (str): Строка с фонемами, разделёнными пробелами.
    Returns:
        torch.Tensor: Тензор с ID фонем.
    """
    # Разбиваем текст на токены (фонемы, знаки препинания, специальные символы)
    tokens = text.split()
    
    # Преобразуем каждый токен в соответствующий ID, используя словарь _symbol_to_id
    sequence = [_symbol_to_id[token.lower()] for token in tokens if token.lower() in _symbol_to_id]
    
    # Преобразуем список ID в тензор
    sequence_tensor = torch.tensor(sequence, dtype=torch.int64)
    
    return sequence_tensor

def tensor_to_ru_phonemes(tensor):
    """
    Конвертирует тензор ID фонем обратно в текст.
    Args:
        tensor (torch.Tensor): Тензор с ID фонем.
    Returns:
        str: Строка с фонемами, разделёнными пробелами.
    """
    # Преобразуем тензор в список ID
    ids = tensor.tolist()
    
    # Преобразуем каждый ID обратно в соответствующий токен, используя словарь _id_to_symbol
    tokens = [ _id_to_symbol.get(id, '<UNK>') for id in ids ]
    
    # Объединяем токены в строку с пробелами между ними
    text = ' '.join(tokens)
    
    return text

def get_lpc_text_pair(text, lpc_path): 
    text = get_ru_phonemes(text)         #text = get_ru_text(text)
    lpc = get_lpc_from_f32(lpc_path)
    return (text, lpc)

class ljcollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step
        self.plus_id = _symbol_to_id['+']  # ID символа "+"
        self.pad_id = _symbol_to_id['_']    # ID символа "_" (или нуля)

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.fill_(self.pad_id)  # Заполняем паддинг символом (например, нулем)

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]

            # Случайное удаление символа "+"
            text = self.random_remove_plus(text)

            # Текст уже паддится нулями в text_padded, просто вставляем текст
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

    def random_remove_plus(self, text):
        # Получаем индексы всех символов "+"
        plus_indices = (text == self.plus_id).nonzero(as_tuple=True)[0]

        # Случайным образом выбираем количество символов "+" для удаления
        num_to_remove = random.randint(0, len(plus_indices))
        if num_to_remove > 0:
            indices_to_remove = sorted(random.sample(plus_indices.tolist(), num_to_remove), reverse=True)

            # Удаляем символы "+", объединяя оставшиеся части
            for idx in indices_to_remove:
                text = torch.cat((text[:idx], text[idx+1:]), dim=0)

        return text


# ### data loader

def prepare_dataloaders(fdir, n_gpu):
    trainset = ljdataset(fdir)
    collate_fn = ljcollate(hps.n_frames_per_step)
    sampler = DistributedSampler(trainset) if n_gpu > 1 else None
    train_loader = DataLoader(trainset, num_workers = hps.n_workers, shuffle = n_gpu == 1,
                              batch_size = hps.batch_size, pin_memory = hps.pin_mem,
                              drop_last = True, collate_fn = collate_fn, sampler = sampler)
    return train_loader


# ## usage and batch tests
# data_dir = '/app/TTSproject/tacotron/datasets/'

# train_loader = prepare_dataloaders(data_dir, 1)

# first_batch = next(iter(train_loader))

# text_padded, input_lengths, mel_padded, gate_padded, output_lengths = first_batch

# print('\n', sequence_to_text(list(text_padded[0].tolist())))
