# atmacup-16th-19th-place-solution
atmaCup#16の19位のソリューションです。

## 前提
- python 3.10
- poetry

## 実行手順
- 環境構築
  - `poetry install`
- データの配置
  - `atmaCup#16`のデータセットを`input`ディレクトリ配下に配置
- GNNの学習と推論
  - [atmaCup-16-in-collaboration-with-RECRUIT
](https://github.com/tubo213/atmaCup-16-in-collaboration-with-RECRUIT) をcloneし、当該repoのREADMEに従って環境構築を実施
  - 当該repoの`/bin/conf/train.yaml` ファイルのepochsを13に変更する
  - このrepoの `nn_train.sh` 及び `nn_inference.sh` を `atmaCup-16-in-collaboration-with-RECRUIT` 配下に設置し、それぞれ実行する
  - 上記で出力された予測ファイルに対して、このrepoの `nn_sub_dict.py` を実行し、後に使うファイル(`{train, test}_transformer_sub_dict.pkl`) を作成する
    - github repositoryには含めませんが、運営の方に提出するzipファイルには上記のpklファイルを含めるようにします
- solutionの実行
  - `poetry run python solution.py`
    - `submission` ディレクトリ配下に、submit用の `submission_solution_{oofスコア}.csv` が出力されます
    - 私の環境で実行したところ、public LBで0.4425, private LBで0.4401のスコアを記録しました

