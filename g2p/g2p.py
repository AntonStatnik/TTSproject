import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import re
from collections import Counter
import numpy as np
import os

class G2PModel:
    def __init__(self, model_dir, device=None):
        """
        Инициализирует модель G2P.

        Args:
            model_dir (str): Путь к директории с моделью (где находятся ru.dic, ru_emphasize.dict и checkpoint.pth).
            device (torch.device, optional): Устройство для вычислений. Если None, автоматически выбирается.
        """
        self.model_dir = model_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Гиперпараметры
        self.BATCH_SIZE = 64
        self.EMBEDDING_DIM = 128
        self.HIDDEN_SIZE = 256
        self.NUM_EPOCHS = 42
        self.LEARNING_RATE = 0.001

        # Загрузка данных и словарей
        self.word_to_phonemes = self.read_ru_dic(os.path.join(self.model_dir, 'ru.dic'))
        self.word_to_stressed_word = self.read_ru_emphasize_dict(os.path.join(self.model_dir, 'ru_emphasize.dict'))
        self.data = self.prepare_data()
        self.char_vocab, self.idx2char = self.build_char_vocab()
        self.phoneme_vocab, self.idx2phoneme = self.build_phoneme_vocab()
        self.encoded_words, self.encoded_phonemes = self.encode_data()
        self.dataloader = self.create_dataloader()

        # Инициализация модели
        self.encoder = Encoder(len(self.char_vocab), self.EMBEDDING_DIM, self.HIDDEN_SIZE).to(self.device)
        self.decoder = Decoder(len(self.phoneme_vocab), self.EMBEDDING_DIM, self.HIDDEN_SIZE).to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.phoneme_vocab['<PAD>'])
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.LEARNING_RATE)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.LEARNING_RATE)

    # Функции для чтения словарей
    def read_ru_dic(self, file_path):
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

    def read_ru_emphasize_dict(self, file_path):
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

    # Подготовка данных
    def prepare_data(self):
        data = []
        for word_with_number, phoneme_seq in self.word_to_phonemes.items():
            # Получаем слово с ударением (если есть)
            input_word = self.word_to_stressed_word.get(word_with_number, word_with_number)
            data.append((input_word, phoneme_seq))
            
        # Учет буквы 'ё' как отдельного символа, всегда ударного
        for idx, (input_word, phoneme_seq) in enumerate(data):
            if 'ё' in input_word:
                input_word = input_word.replace('ё', '+ё')
                data[idx] = (input_word, phoneme_seq)
        return data

    # Создание словаря символов
    def build_char_vocab(self):
        char_counter = Counter()
        for input_word, _ in self.data:
            char_counter.update(list(input_word))
        
        char_vocab = {char: idx+1 for idx, (char, _) in enumerate(char_counter.most_common())}
        char_vocab['<PAD>'] = 0
        char_vocab['<SOS>'] = len(char_vocab)
        char_vocab['<EOS>'] = len(char_vocab)
        char_vocab['<UNK>'] = len(char_vocab)
        
        idx2char = {idx: char for char, idx in char_vocab.items()}
        return char_vocab, idx2char

    # Создание словаря фонем
    def build_phoneme_vocab(self):
        phoneme_counter = Counter()
        for _, phoneme_seq in self.data:
            phoneme_counter.update(phoneme_seq)
        
        phoneme_vocab = {ph: idx+1 for idx, (ph, _) in enumerate(phoneme_counter.most_common())}
        phoneme_vocab['<PAD>'] = 0
        phoneme_vocab['<SOS>'] = len(phoneme_vocab)
        phoneme_vocab['<EOS>'] = len(phoneme_vocab)
        phoneme_vocab['<UNK>'] = len(phoneme_vocab)
        
        idx2phoneme = {idx: ph for ph, idx in phoneme_vocab.items()}
        return phoneme_vocab, idx2phoneme

    # Функции для кодирования
    def encode_word(self, word):
        return [self.char_vocab.get(c, self.char_vocab['<UNK>']) for c in word]
    
    def encode_phonemes(self, phoneme_seq):
        return [self.phoneme_vocab.get(ph, self.phoneme_vocab['<UNK>']) for ph in phoneme_seq]

    def encode_data(self):
        encoded_words = [self.encode_word(input_word) for input_word, _ in self.data]
        encoded_phonemes = [self.encode_phonemes(phoneme_seq) for _, phoneme_seq in self.data]
        return encoded_words, encoded_phonemes

    # Создание даталоадера
    def create_dataloader(self):
        dataset = G2PDataset(self.encoded_words, self.encoded_phonemes)
        dataloader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        return dataloader

    # Функции для сохранения и загрузки чекпойнта
    def save_checkpoint(self, epoch, loss, filepath):
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, filepath)
        print(f"Модель сохранена в {filepath}")

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Модель загружена из {filepath}, Эпоха: {epoch}, Ошибка: {loss:.4f}")
        return epoch, loss

    # Обучение модели
    def train(self):
        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.NUM_EPOCHS):
            total_loss = 0
            for batch in self.dataloader:
                input_seqs, target_seqs, input_lengths, target_lengths = batch
                input_seqs = input_seqs.to(self.device)
                target_seqs = target_seqs.to(self.device)

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                encoder_outputs, encoder_hidden = self.encoder(input_seqs, input_lengths)

                # Подготовка входов для декодера
                decoder_input = target_seqs[:, :-1]  # Убираем <EOS>
                decoder_target = target_seqs[:, 1:]  # Убираем <SOS>

                # Обучение с принудительным учителем (teacher forcing)
                decoder_outputs, _ = self.decoder(decoder_input, encoder_hidden)

                # Вычисление ошибки
                loss = self.criterion(decoder_outputs.reshape(-1, decoder_outputs.size(-1)),
                                     decoder_target.reshape(-1))

                loss.backward()

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f'Эпоха [{epoch+1}/{self.NUM_EPOCHS}], Ошибка: {avg_loss:.4f}')

    # Функция для предсказания фонем
    def predict(self, word):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            input_seq = torch.tensor([self.encode_word(word)], dtype=torch.long).to(self.device)
            input_length = torch.tensor([len(input_seq[0])], dtype=torch.long).to(self.device)

            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

            decoder_input = torch.tensor([[self.phoneme_vocab['<SOS>']]], dtype=torch.long).to(self.device)
            decoder_hidden = encoder_hidden

            decoded_phonemes = []

            for _ in range(50):  # Максимальная длина последовательности фонем
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                phoneme_idx = topi.item()
                if phoneme_idx == self.phoneme_vocab['<EOS>'] or phoneme_idx == self.phoneme_vocab['<PAD>']:
                    break
                else:
                    decoded_phonemes.append(self.idx2phoneme[phoneme_idx])

                decoder_input = torch.tensor([[phoneme_idx]], dtype=torch.long).to(self.device)

            return ' '.join(decoded_phonemes)

    # Функция для обработки строки текста
    def process_text(self, text, special_space="<space>"):
        """
        Обрабатывает одну строку текста и преобразует её в фонемную транскрипцию.

        Args:
            text (str): Текст для обработки.
            special_space (str): Специальный символ для замены пробелов между словами.

        Returns:
            str: Фонемная транскрипция текста.
        """
        # Регулярное выражение для разделения слов (включая '+') и пунктуации
        tokens = re.findall(r'\w+\+\w+|\w+|[.,;:!?«»“”…()\[\]{}\-]', text)

        processed_tokens = []
        for token in tokens:
            if re.match(r'[.,;:!?«»“”…()\[\]{}\-]', token):
                # Если токен - пунктуация, добавляем его как есть
                processed_tokens.append(token)
            else:
                # Иначе, токен - слово (может содержать '+')
                word_lower = token.lower()
                try:
                    # Преобразуем слово в фонемную транскрипцию
                    phoneme = self.predict(word_lower)
                    if phoneme:
                        processed_tokens.append(phoneme)
                    else:
                        print(f"Пустая транскрипция для слова '{word_lower}'. Используем '<UNK>'.")
                        processed_tokens.append('<UNK>')  # Используем <UNK> для неизвестных фонем
                except Exception as e:
                    print(f"Ошибка при предсказании слова '{word_lower}': {e}. Используем '<UNK>'.")
                    processed_tokens.append('<UNK>')

        # Теперь собираем новую строку с правильными разделителями
        final_tokens = []
        for i, token in enumerate(processed_tokens):
            final_tokens.append(token)
            if i < len(processed_tokens) - 1:
                next_token = processed_tokens[i + 1]
                # Определяем тип текущего и следующего токена
                current_is_punct = re.match(r'[.,;:!?«»“”…()\[\]{}\-]', token)
                next_is_punct = re.match(r'[.,;:!?«»“”…()\[\]{}\-]', next_token)

                if not current_is_punct and not next_is_punct:
                    # Оба токена - слова, вставляем специальный символ с пробелами
                    final_tokens.append(f" {special_space} ")
                elif not current_is_punct and next_is_punct:
                    # Текущий токен - слово, следующий - пунктуация, вставляем пробел
                    final_tokens.append(' ')
                elif current_is_punct and not next_is_punct:
                    # Текущий токен - пунктуация, следующий - слово, вставляем пробел
                    final_tokens.append(' ')
                elif current_is_punct and next_is_punct:
                    # Оба токена - пунктуация, вставляем пробел
                    final_tokens.append(' ')
                else:
                    # В остальных случаях, вставляем пробел
                    final_tokens.append(' ')

        # Объединяем токены в строку
        phoneme_text = ''.join(final_tokens)

        # Заменяем множественные пробелы на один
        phoneme_text = re.sub(r'\s+', ' ', phoneme_text)

        return phoneme_text

# Определение классов Encoder, Decoder, G2PDataset и функции collate_fn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)  # <PAD> = 0
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

        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)  # <PAD> = 0
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seqs, hidden):
        embedded = self.embedding(input_seqs)
        output, hidden = self.lstm(embedded, hidden)
        output = self.out(output)
        return output, hidden

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
    char_vocab = batch[0][0].__class__.__base__.__dict__['self'].char_vocab  # Плохой способ, лучше передать отдельно
    phoneme_vocab = batch[0][0].__class__.__base__.__dict__['self'].phoneme_vocab  # Плохой способ, лучше передать отдельно

    inputs, targets, input_lengths, target_lengths = zip(*batch)

    max_input_len = max(input_lengths)
    max_target_len = max(target_lengths)

    padded_inputs = [seq + [char_vocab['<PAD>']] * (max_input_len - len(seq)) for seq in inputs]
    padded_targets = [[phoneme_vocab['<SOS>']] + seq + [phoneme_vocab['<EOS>']] + [phoneme_vocab['<PAD>']] * (max_target_len - len(seq)) for seq in targets]

    return (torch.tensor(padded_inputs, dtype=torch.long),
            torch.tensor(padded_targets, dtype=torch.long),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor([len(seq)+2 for seq in targets], dtype=torch.long))  # +2 для <SOS> и <EOS>
