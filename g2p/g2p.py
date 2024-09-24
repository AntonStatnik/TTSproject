import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import re
from collections import Counter
import numpy as np

# Устройство (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Гиперпараметры
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Функция для чтения ru.dic
def read_ru_dic(file_path):
    word_to_phonemes = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^(.+?)\s+(.+)$', line)
            if match:
                word_with_number = match.group(1)
                transcription = match.group(2)
                word_to_phonemes[word_with_number] = transcription.split()
            else:
                print(f"Строка пропущена: {line}")
    return word_to_phonemes

# Функция для чтения ru_emphasize.dict
def read_ru_emphasize_dict(file_path):
    word_to_stressed_word = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) == 2:
                word_with_number = parts[0]
                stressed_word = parts[1]
                word_to_stressed_word[word_with_number] = stressed_word
            else:
                print(f"Строка пропущена: {line}")
    return word_to_stressed_word

# Считываем данные из файлов
word_to_phonemes = read_ru_dic('./ru.dic')
word_to_stressed_word = read_ru_emphasize_dict('./ru_emphasize.dict')

# Создаем датасет
data = []
for word_with_number, phoneme_seq in word_to_phonemes.items():
    # Получаем слово с ударением (если есть)
    input_word = word_to_stressed_word.get(word_with_number, word_with_number)
    data.append((input_word, phoneme_seq))
    
# Учет буквы 'ё' как отдельного символа, всегда ударного
for idx, (input_word, phoneme_seq) in enumerate(data):
    # Если в слове есть 'ё', добавляем '+' перед ней
    if 'ё' in input_word:
        input_word = input_word.replace('ё', '+ё')
        data[idx] = (input_word, phoneme_seq)
        
# Создание словаря символов (включая '+')
char_counter = Counter()
for input_word, _ in data:
    char_counter.update(list(input_word))

char_vocab = {char: idx+1 for idx, (char, _) in enumerate(char_counter.most_common())}
char_vocab['<PAD>'] = 0
char_vocab['<SOS>'] = len(char_vocab)
char_vocab['<EOS>'] = len(char_vocab)
char_vocab['<UNK>'] = len(char_vocab)

idx2char = {idx: char for char, idx in char_vocab.items()}

# Создание словаря фонем
phoneme_counter = Counter()
for _, phoneme_seq in data:
    phoneme_counter.update(phoneme_seq)

phoneme_vocab = {ph: idx+1 for idx, (ph, _) in enumerate(phoneme_counter.most_common())}
phoneme_vocab['<PAD>'] = 0
phoneme_vocab['<SOS>'] = len(phoneme_vocab)
phoneme_vocab['<EOS>'] = len(phoneme_vocab)
phoneme_vocab['<UNK>'] = len(phoneme_vocab)

idx2phoneme = {idx: ph for ph, idx in phoneme_vocab.items()}

# Функции для кодирования
def encode_word(word):
    return [char_vocab.get(c, char_vocab['<UNK>']) for c in word]

def encode_phonemes(phoneme_seq):
    return [phoneme_vocab.get(ph, phoneme_vocab['<UNK>']) for ph in phoneme_seq]

# Кодируем данные
encoded_words = [encode_word(input_word) for input_word, _ in data]
encoded_phonemes = [encode_phonemes(phoneme_seq) for _, phoneme_seq in data]

class G2PDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

        self.input_lengths = [len(seq) for seq in inputs]
        self.target_lengths = [len(seq) for seq in targets]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.targets[idx],
                self.input_lengths[idx], self.target_lengths[idx])

def collate_fn(batch):
    inputs, targets, input_lengths, target_lengths = zip(*batch)

    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)

    padded_inputs = [seq + [char_vocab['<PAD>']] * (max_input_len - len(seq)) for seq in inputs]
    padded_targets = [[phoneme_vocab['<SOS>']] + seq + [phoneme_vocab['<EOS>']] + [phoneme_vocab['<PAD>']] * (max_target_len - len(seq)) for seq in targets]

    return (torch.tensor(padded_inputs, dtype=torch.long),
            torch.tensor(padded_targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor([len(seq)+2 for seq in targets], dtype=torch.long))  # +2 для <SOS> и <EOS>

dataset = G2PDataset(encoded_words, encoded_phonemes)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=char_vocab['<PAD>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, input_seqs, input_lengths):
        embedded = self.embedding(input_seqs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=phoneme_vocab['<PAD>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden):
        embedded = self.embedding(input_seqs)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)
        return output, hidden

    
# Функция для сохранения чекпойнта
def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, loss, filepath):
    checkpoint = {
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Модель сохранена в {filepath}")

# Функция для загрузки чекпойнта
def load_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, filepath):
    checkpoint = torch.load(filepath, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Модель загружена из {filepath}, Эпоха: {epoch}, Ошибка: {loss:.4f}")
    return epoch, loss

encoder = Encoder(len(char_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(device)
decoder = Decoder(len(phoneme_vocab), EMBEDDING_DIM, HIDDEN_SIZE).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=phoneme_vocab['<PAD>'])
encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)


def train_g2p(encoder, decoder, encoder_optimizer, decoder_optimizer, NUM_EPOCHS): 
    for epoch in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()

        total_loss = 0
        for batch in dataloader:
            input_seqs, target_seqs, input_lengths, target_lengths = batch
            input_seqs = input_seqs.to(device)
            target_seqs = target_seqs.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_seqs, input_lengths)

            # Подготовка входов для декодера
            decoder_input = target_seqs[:, :-1]  # Убираем <EOS>
            decoder_target = target_seqs[:, 1:]  # Убираем <SOS>

            # Обучение с принудительным учителем (teacher forcing)
            decoder_outputs, _ = decoder(decoder_input, encoder_hidden)

            # Вычисление ошибки
            loss = criterion(decoder_outputs.reshape(-1, decoder_outputs.size(-1)),
                             decoder_target.reshape(-1))

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Эпоха [{epoch+1}/{NUM_EPOCHS}], Ошибка: {avg_loss:.4f}')


def predict(word):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_seq = torch.tensor([encode_word(word)], dtype=torch.long).to(device)
        input_length = torch.tensor([len(input_seq[0])], dtype=torch.long).to(device)

        encoder_outputs, encoder_hidden = encoder(input_seq, input_length)

        decoder_input = torch.tensor([[phoneme_vocab['<SOS>']]], dtype=torch.long).to(device)
        decoder_hidden = encoder_hidden

        decoded_phonemes = []

        for _ in range(50):  # Максимальная длина последовательности фонем
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            phoneme_idx = topi.item()
            if phoneme_idx == phoneme_vocab['<EOS>'] or phoneme_idx == phoneme_vocab['<PAD>']:
                break
            else:
                decoded_phonemes.append(idx2phoneme[phoneme_idx])

            decoder_input = torch.tensor([[phoneme_idx]], dtype=torch.long).to(device)

        return ' '.join(decoded_phonemes)


