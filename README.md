# DeepLearning BASIC 最終課題 :VQA

## 方針
testdataのほぼ半分(47%)はunanswerableが解答データ (6/22 検証済み)
それ以外の色や文字、物体認識を行うタスクをどれだけやれるかが勝負
ローカルで動く優れた事前学習済みモデルをtest_dataでFineTuningする

##　内容
使ったモデル
mini-cpm--llama3-v2.5にテストデータからunanswerableデータを150個、それ以外を250個の合計400個ランダムに抽出してQLORAして予測
