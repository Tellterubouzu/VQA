<div style="text-align: center;">
<h1>Deep Learning基礎講座　最終課題レポート</h1>
</div>
<div style="text-align: right;">
<p>下村晃生    </p>
</div>


### VQAタスクの概要
画像一枚と、それに関する質問に対しての簡潔な回答が求められる。特徴として、train_dataの答えには "answer confidence"と "answer"の二つの項目があることがあげられる。
answer confidenceではyes/maybe/noという三択で回答の確信度が示されていて、answerでは質問に対する回答として、色や文字起こし、物体の名前などの質問に対する直接的な回答があるほか、"unanswerable"という与えられた画像からはわからないというラベルがある。このunanswerableのラベルの量がかなりの割合を占めており、おおよそ50%近くである。実際、予測データすべてを"unanswerable"として提出したところ正答率は47%を占めた。
以上のことから今回のコンペにおいて正答率を上げるには、unanswerable以外のデータをいかに正しく予測できるかが大事であると考えた。

### 課題に取り組むにあたって決めた方針
今回、コンペでは以下の二つのモデルを作成し、その予測を組み合わせることで最終的な予測とすることにした。
1. unanswerableな画像・質問の組み合わせかどうかを判断する二値分類モデル
2. 二値分類モデルで回答可能と判断されたものに対して予測を行うモデル

### 使用したモデル
今回のVQAタスクの質問内容は多岐にわたっており、画像の色、写っている物体、物体に対して何かしたときにおこることなど様々なことに対して予測を行うことが求められる。これらの質問に対して正しく行うには、ベースラインコードで示された、ラベルの分類ではなく質問そのものを理解する必要があると考えた。
よって今回はAlpache2.0ライセンスの事前学習済みvision-transformerに対してテストデータでfine-tuningを行うこととした。
実際に使用した事前学習済みモデルは以下のものである。
[MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)
選定理由としては、このモデルがmetaが開発したllama-3-7Bをベースにしたvision-transformerであり、画像に対する言語理解が優れているためである。

### 前処理
画像データとテキストデータの前処理については以下のものを行った。モデルの学習などを試行錯誤していく中で何度も同じ前処理を行うのは計算資源と時間がもったいないと感じたためこれらの前処理を行ったものを予め保存しておき、訓練・推論時には前処理済みのデータを読み込んでから行った。
##### 画像
使用するモデルがvision-transformerであることや、タスクの中に色を答えるものがあるため、zca白色化や二値化などの処理は行わなかった。
画像は全体的にピンボケしているものと、暗く、コントラストがはっきりしないものが多かったためえ、opencvを用いて適応的ヒストグラム平坦化と、フィルタを用いたシャープネス化を行った。
シャープネス化については、度合いが強すぎるとノイズが増加し、正答率が下がったため、フィルタ2を最終的には採用した。
```
[[-1,-1,-1]    [0,-1, 0]
 [-1, 9,-1]    [-1,5,-1]
 [-1,-1,,-1]]  [0,-1, 0]
 フィルタ1      フィルタ2
```
##### テキスト
回答データのテキストの前処理については、ベースラインコードにあった数字の変換と小文字への統一以外はやらなかった。理由として今回扱ったモデルはvision-transformerでありより文脈を読み取って出力することができるからである。


### Fine-tuning
##### Fine-tuning手法
今回扱ったFine-tuning手法はLLM部分に対してのQuantized Loraである。
元のモデルが8Bほどのパラメータがあり、通常のファインチューニングを行うと40GBほどのメモリを使用しout of memoryとなってしまう。
よって、訓練時のメモリ不足と時間短縮のためにQLoRaを実施した。
##### 二値分類
Fine-tuningは二つ行い、answerableとunanserableの二つに分類するモデルと通常通り質問に回答するモデルの作成を行った。

##### 質問応答
### Prompt
推論実行時のプロンプトはそれぞれのモデルで以下の通りでFewShotを活用したものとした。
##### 二値分類
```
# Instruction
Your task is to classify the answers into two categories, "answerable" or "unanswerable", indicating whether you can answer a given image/question combination.
Answers shoud be shown as formatted with answer_confidence set yes or no, and follow the format and examples below.

# Question
{question_text}

# Format
"answer_confidence":"yes/no","answer":"your_answer"

## Example 1
"answer_confidence":"no","answer":"answerable"
## Example 2
"answer_confidence":"yes","answer":"answerable"
## Example 3
"answer_confidence":"yes","answer":"unanswerable"
```
##### 質問応答
```
Please answer the question about the image.
# Question
{question_text}

The answer to the above question should be shown as formatted with answer_confidence set to yes or no, followed by an answer, as in the following example. 
# Format
"answer_confidence":"yes/no","answer":"your_answer"

## Example 1
"answer_confidence":"no","answer":"cherry"
## Example 2
"answer_confidence":"yes","answer":"blue"
## Example 3
"answer_confidence":"yes","answer":"Brown"
```
