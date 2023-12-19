import joblib
import numpy as np
import polars as pl


def create_agg_oof(oof_df: pl.DataFrame) -> pl.DataFrame:
    agg_oof_df = oof_df.group_by("session_id").agg(
        [
            pl.col("yad_no")
            .sort_by("score", descending=True)
            .apply(list)
            .alias("yad_no_list")
        ]
    )
    return agg_oof_df


def load_test_gnn_pred(exp_dirs, fold_num: int = 5):
    # TODO: 環境に応じてパスを変更する
    base_filename = "../atmaCup-16-in-collaboration-with-RECRUIT/output/inference/{exp_dir}/single/oof_test.csv"

    oofs = []
    for fold in range(fold_num):
        exp_dir = exp_dirs[fold]
        filename = base_filename.format(exp_dir=exp_dir)
        print(filename)
        data = pl.read_csv(filename)
        oofs.append(data)

    sum_score = pl.Series(values=np.zeros(len(oofs[0])))
    for fold in range(fold_num):
        sum_score = sum_score + oofs[fold]["score"]
    result_df = oofs[0].clone()
    result_df = result_df.with_columns("score", sum_score)
    test_sub_df = create_agg_oof(result_df)
    return test_sub_df


def load_train_gnn_pred(exp_dirs, fold_num: int = 5):
    # TODO: 環境に応じてパスを変更する
    val_base_filename = "../atmaCup-16-in-collaboration-with-RECRUIT/output/inference/{exp_dir}/single/oof_val.csv"
    val_oofs = []
    for fold in range(fold_num):
        exp_dir = exp_dirs[fold]
        filename = val_base_filename.format(exp_dir=exp_dir)
        print(filename)
        data = pl.read_csv(filename)
        print(data.shape)
        val_oofs.append(data)
    val_df = pl.concat(val_oofs)
    val_sub_df = create_agg_oof(val_df)
    return val_sub_df


def convert_pldf_to_dict(sub_df):
    dic = {}
    for session_id, values in sub_df.iter_rows():
        dic[session_id] = values
    return dic


if __name__ == "__main__":
    exp_dirs = [
        "transformer_0",
        "transformer_1",
        "transformer_2",
        "transformer_3",
        "transformer_4",
    ]

    fold_num = 5

    test_agg_oof = load_test_gnn_pred(exp_dirs)
    test_sub_dict = convert_pldf_to_dict(test_agg_oof)
    joblib.dump(test_sub_dict, "./input/test_transformer_sub_dict.pkl")

    train_agg_oof = load_train_gnn_pred(exp_dirs)
    train_sub_dict = convert_pldf_to_dict(train_agg_oof)
    joblib.dump(train_sub_dict, "./input/train_transformer_sub_dict.pkl")