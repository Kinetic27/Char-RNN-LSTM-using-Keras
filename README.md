# char-lstm-rnn-keras, Kinetic

문장을 학습하고 샘플을 뽑아내는 다층 RNN은 [Andrej Karpathy의 문서](http://karpathy.github.io/2015/05/21/rnn-effectiveness)에서 영감을 받아서 작성되었습니다, 또한 원본 코드는 [karpathy/char-rnn](https://github.com/karpathy/char-rnn)입니다.

## 요구사항

파이썬3와 케라스 딥러닝 라이브러리가 필요합니다.

## 입력 데이터

`kdata` 폴더안에 한글로 이루어진 txt데이터를 넣어둡니다.

## 사용법

기본 설정으로 모델 학습하기
```bash
$ python train.py
```

특정 파일을 특정 epochs까지 이어서 학습하기
```bash
$ python train.py --input 텍본.txt --epochs 1000 --resume
```

100 epoch의 샘플 뽑기
```bash
$ python sample.py 100
```

시작 시드 단어 지정 및 특정 글자수 만큼 뽑기
```
$ python sample.py 380 --len 200 --seed "테스트"
```

100 epoch의 그래프 그리기
```bash
$ python graph.py 100
```

학습 loss/accuracy는 `logs/training_log.csv`에 저장이 됩니다

학습 모델(weight 포함)은 `model`에 저장이 되고, `sample.py`에서 샘플을 추출할 때 사용됩니다.

또한 `acc`와 `loss`에는 각 step의 accuracy와 loss가 기록되며 이를 이용하여 `graph.py`에서 정교한 그래프를 그릴 수 있습니다.
