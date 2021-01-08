import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchvision import transforms
import numpy as np

torchaudio.set_audio_backend("sox_io")

# CommonVoiceをもとにtransformできるようにしたデータセット
# CommonVoiceはあらかじめダウンロードしておくこと
# https://commonvoice.mozilla.org/ja/datasets
class SpeechDataset(Dataset):
    sample_rate = 16000
    def __init__(self, train=True, transform=None, split_rate=0.8):
        tsv = './CommonVoice/cv-corpus-5.1-2020-06-22/ja/validated.tsv'
        # データセットの一意性確認と正解ラベルの列挙
        import pandas as pd
        df = pd.read_table(tsv)
        assert not df.path.duplicated().any()
        self.classes = df.client_id.drop_duplicates().tolist()
        self.n_classes = len(self.classes)
        
        # データセットの準備
        self.transform = transform
        data_dirs = tsv.split('/')
        dataset = torchaudio.datasets.COMMONVOICE(
            '/'.join(data_dirs[:-4]), tsv=data_dirs[-1],
            url='japanese', version=data_dirs[-3])
        # データセットの分割
        n_train = int(len(dataset) * split_rate)
        n_val = len(dataset) - n_train
        torch.manual_seed(torch.initial_seed())  # 同じsplitを得るために必要
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        self.dataset = train_dataset if train else val_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        x, sample_rate, dictionary = self.dataset[idx]
        # リサンプリングしておくと以降は共通sample_rateでtransformできる
        if sample_rate != self.sample_rate:
            x = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)(x)
        # 各種変換、MFCC等は外部でtransformとして記述する
        # ただし、推論とあわせるためにMFCCは先にすませておく
        x = torchaudio.transforms.MFCC(log_mels=True)(x)
        # 最終的にxのサイズを揃えること
        if self.transform:
            x = self.transform(x)
        # 特徴量:音声テンソル、正解ラベル:話者IDのインデックス
        return x, self.classes.index(dictionary['client_id'])

# 学習モデル
class SpeechNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc = nn.Sequential(
            nn.Linear(30*64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, n_classes),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 最後の1次元に指定サイズにCropし、長さが足りない時はCircularPadする
# 音声データの時間方向の長さを揃えるために使うtransform部品
class CircularPad1dCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, x):
        n_repeat = self.size // x.size()[-1] + 1
        repeat_sizes = ((1,) * (x.dim() - 1)) + (n_repeat,)
        out = x.repeat(*repeat_sizes).clone()
        return out.narrow(-1, 0, self.size)

def SpeechML(train_dataset=None, val_test_dataset=None, *,
             n_classes=None, n_epochs=15,
             load_pretrained_state=None, test_last_hidden_layer=False,
             show_progress=True, show_chart=False, save_state=False):
    '''
    前処理、学習、検証、推論を行う
    train_dataset: 学習用データセット
    val_test_dataset: 検証/テスト用データセット
    （検証とテストでデータを変えたい場合は一度学習してステートセーブした後に
      テストのみでステート読み出しして再実行すること）
    （正解ラベルが無い場合は検証はスキップする）
    n_classes: 分類クラス数（Noneならtrain_datasetから求める）
    n_epocs: 学習エポック数
    load_pretrained_state: 学習済ウエイトを使う場合の.pthファイルのパス
    test_last_hidden_layer: テストデータの推論結果に最終隠れ層を使う
    show_progress: エポックの学習状況をprintする
    show_chart: 結果をグラフ表示する
    save_state: test_acc > 0.9 の時のtest_loss最小値更新時のstateを保存
   　　　　　　　 （load_pretrained_stateで使う）
    返り値: テストデータの推論結果
    '''
    # モデルの準備
    if not n_classes:
        assert train_dataset, 'train_dataset or n_classes must be a valid.'
        n_classes = train_dataset.n_classes
    model = SpeechNet(n_classes)
    if load_pretrained_state:
        model.load_state_dict(torch.load(load_pretrained_state))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # 前処理の定義
    Squeeze2dTo1d = lambda x: torch.squeeze(x, -3)
    train_transform = transforms.Compose([
        CircularPad1dCrop(800),
        transforms.RandomCrop((40, random.randint(160, 320))),
        transforms.Resize((40, 240)),
        Squeeze2dTo1d,
    ])
    test_transform = transforms.Compose([
        CircularPad1dCrop(240),
        Squeeze2dTo1d
    ])
    # 学習データ・テストデータの準備
    batch_size = 32
    if train_dataset:
        train_dataset.transform = train_transform
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    else:
        n_epochs = 0   # 学習データが無けれはエポックはまわせない
    if val_test_dataset:
        val_test_dataset.transform = test_transform
        val_test_dataloader = DataLoader(
            val_test_dataset, batch_size=batch_size)
    # 学習
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    for epoch in range(n_epochs):
        # 学習ループ
        running_loss = 0.0
        running_acc = 0.0
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            running_loss += loss.item()
            pred = torch.argmax(y_pred, dim=1)
            running_acc += torch.mean(pred.eq(y_train).float())
            optimizer.step()
        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)
        losses.append(running_loss)
        accs.append(running_acc)
        # 検証ループ
        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_test in val_test_dataloader:
            if not(type(val_test) is list and len(val_test) == 2):
                break
            x_val, y_val = val_test
            y_pred = model(x_val)
            val_loss = criterion(y_pred, y_val)
            val_running_loss += val_loss.item()
            pred = torch.argmax(y_pred, dim=1)
            val_running_acc += torch.mean(pred.eq(y_val).float())
        val_running_loss /= len(val_test_dataloader)
        val_running_acc /= len(val_test_dataloader)
        can_save = (val_running_acc > 0.9 and
                    val_running_loss < min(val_losses))
        val_losses.append(val_running_loss)
        val_accs.append(val_running_acc)
        if show_progress:
            print(f'epoch:{epoch}, loss:{running_loss:.3f}, '
                  f'acc:{running_acc:.3f}, val_loss:{val_running_loss:.3f}, '
                  f'val_acc:{val_running_acc:.3f}, can_save:{can_save}')
        if save_state and can_save:   # あらかじめmodelフォルダを作っておくこと
            torch.save(model.state_dict(), f'model/0001-epoch{epoch:02}.pth')
    # グラフ描画
    if n_epochs > 0 and show_chart:
        fig, ax = plt.subplots(2)
        ax[0].plot(losses, label='train loss')
        ax[0].plot(val_losses, label='val loss')
        ax[0].legend()
        ax[1].plot(accs, label='train acc')
        ax[1].plot(val_accs, label='val acc')
        ax[1].legend()
        plt.show()
    # 推論
    if not val_test_dataset:
        return

    # print(model.fc[-1].weight)
    x_vector_speekers = model.fc[-1].weight.detach().numpy().copy()
    # print(x_vector_speekers.shape)
    # print(x_vector_speekers)

    if test_last_hidden_layer:
        model.fc = model.fc[:-1]  # 最後の隠れ層を出力する

    y_preds = torch.Tensor()
    for val_test in val_test_dataloader:
        x_test = val_test[0] if type(val_test) is list else val_test
        y_pred = model.eval()(x_test)
        if not test_last_hidden_layer:
            y_pred = torch.argmax(y_pred, dim=1)
        y_preds = torch.cat([y_preds, y_pred])

        # x_vector_val = y_pred.detach().numpy().copy()
        # preds = []
        # for x_vector in x_vector_val:
        #     maxcos = -2
        #     pred = -1
        #     for i in range(len(x_vector_speekers)):
        #         cos = cos_sim(x_vector, x_vector_speekers[i])
        #         if maxcos < cos:
        #             maxcos = cos
        #             pred = i
        #     preds.append(pred)
        
        # label = val_test[1].detach().numpy().copy()
        # cnt = 0
        # nums = len(label)
        # for i in range(nums):
        #     if label[i] == preds[i]:
        #         cnt += 1
        # print(cnt / nums)


        # print(val_test[1]) # 正解ラベル
        # print(y_pred)
    return y_preds.detach()

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 呼び出しサンプル
if __name__ == '__main__':
    train_dataset = SpeechDataset(train=True)
    val_dataset = SpeechDataset(train=False)
    result = SpeechML(train_dataset, val_dataset, n_epochs=20,
        show_chart=True, save_state=True, test_last_hidden_layer=True)
    print(result)