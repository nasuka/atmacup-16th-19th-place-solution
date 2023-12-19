import os
from collections import OrderedDict, defaultdict
from heapq import heappop, heappush

import joblib
import pandas as pd
from tqdm import tqdm

from src.sampler import FrequentCdSampler
from src.util import timer


def apk(actual, predicted, k=10):
    if actual in predicted[:k]:
        return 1.0 / (predicted[:k].index(actual) + 1)
    return 0.0


def mapk(actual, predicted, k=10):
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)


def make_session_list(session_log):
    map_session_yads = defaultdict(list)
    for _, row in tqdm(session_log.iterrows()):
        session_id = row[0]
        yad_no = row[2]
        map_session_yads[session_id].append(yad_no)
    return map_session_yads


def create_co_booking_topN_yads_for_session(
    session_id,
    co_booking_matrix,
    map_session_yads_all,
):
    viewed_yads: list[int] = map_session_yads_all.get(session_id)
    sum_yad_scores = defaultdict(int)

    for viewed_yad_no in viewed_yads:
        co_booking_yads = co_booking_matrix[viewed_yad_no]
        for co_bookin_yad, score in co_booking_yads.items():
            sum_yad_scores[co_bookin_yad] += score
    sorted_yad_scores = sorted(sum_yad_scores.items(), key=lambda x: x[1], reverse=True)
    return OrderedDict(sorted_yad_scores)


def create_topN_yads(
    session,
    map_session_yads,
    next_booking_matrix,
    co_booking_matrix,
    co_visiting_matrix,
    gnn_pred_dict,
    sml_cd_dict,
    lrg_cd_dict,
    ken_cd_dict,
    sml_freq_sampler,
    lrg_freq_sampler,
    ken_freq_sampler,
    only_train_yads=None,
    n=10,
):
    session_num = len(session)
    predicted_list = [[3338] * n for _ in range(session_num)]
    for idx, session_id in tqdm(enumerate(session["session_id"])):
        viewed_number = len(map_session_yads[session_id])
        last_viewed = map_session_yads[session_id][-1]
        exclude_set = set([viewed_number, last_viewed])
        if only_train_yads is not None:
            exclude_set = exclude_set.union(only_train_yads)

        rank = 0
        if viewed_number > 1:
            # 最後から2番目のデータを自動的に
            predicted_list[idx][rank] = map_session_yads[session_id][-2]
            rank += 1

            # 3件以上あるときは、最初の宿を追加
            if viewed_number >= 3:
                yad_no = map_session_yads[session_id][0]
                if yad_no not in exclude_set or yad_no in predicted_list[idx]:
                    predicted_list[idx][rank] = yad_no
                    rank += 1

        # co visiting
        co_visiting_sorted_yad_list = []
        for yad_no, viewed_cnt in co_visiting_matrix[last_viewed].items():
            if yad_no in exclude_set or yad_no in predicted_list[idx]:
                continue
            heappush(co_visiting_sorted_yad_list, (-viewed_cnt, yad_no))

        while rank < n and co_visiting_sorted_yad_list:
            _, predicted_yad_no = heappop(co_visiting_sorted_yad_list)
            predicted_list[idx][rank] = predicted_yad_no
            rank += 1

        if rank == n:
            continue

        # gnn pred dict
        # trainで何故かsession_idが存在しないやつがあるのでそのときはスキップ
        if session_id in gnn_pred_dict:
            gnn_candidates = gnn_pred_dict[session_id]
            for yad_no in gnn_candidates:
                if rank >= n:
                    break
                if yad_no in exclude_set or yad_no in predicted_list[idx]:
                    continue
                predicted_list[idx][rank] = yad_no
                rank += 1
            if rank == n:
                continue

        # next booking
        next_booking_sorted_yad_list = []
        for yad_no, viewed_cnt in next_booking_matrix[last_viewed].items():
            if yad_no in exclude_set or yad_no in predicted_list[idx]:
                continue
            heappush(next_booking_sorted_yad_list, (-viewed_cnt, yad_no))

        while rank < n and next_booking_sorted_yad_list:
            _, predicted_yad_no = heappop(next_booking_sorted_yad_list)
            predicted_list[idx][rank] = predicted_yad_no
            rank += 1
        if rank == n:
            continue

        # co booking
        co_booking_candidates = create_co_booking_topN_yads_for_session(
            session_id, co_booking_matrix, map_session_yads
        )
        for yad_no in co_booking_candidates:
            if rank >= n:
                break
            if yad_no in exclude_set or yad_no in predicted_list[idx]:
                continue
            predicted_list[idx][rank] = yad_no
            rank += 1
        if rank == n:
            continue

        # sml_cd
        sml_candidates = sml_freq_sampler.sample_by_session_id(
            session_id, k=20 * n, cd_dict=sml_cd_dict, data_type="test"
        )
        for yad_no in sml_candidates:
            if rank >= n:
                break
            if yad_no in exclude_set or yad_no in predicted_list[idx]:
                continue
            predicted_list[idx][rank] = yad_no
            rank += 1
        if rank == n:
            continue

        # lrg_cd
        lrg_candidates = lrg_freq_sampler.sample_by_session_id(
            session_id, k=20 * n, cd_dict=lrg_cd_dict, data_type="test"
        )
        for yad_no in lrg_candidates:
            if rank >= n:
                break
            if yad_no in exclude_set or yad_no in predicted_list[idx]:
                continue
            predicted_list[idx][rank] = yad_no
            rank += 1
        if rank == n:
            continue

        # ken_cd
        ken_candidates = ken_freq_sampler.sample_by_session_id(
            session_id, k=20 * n, cd_dict=ken_cd_dict, data_type="test"
        )
        for yad_no in ken_candidates:
            if rank >= n:
                break
            if yad_no in exclude_set or yad_no in predicted_list[idx]:
                continue
            predicted_list[idx][rank] = yad_no
            rank += 1
    return predicted_list


if __name__ == "__main__":
    should_save = False
    nn_model = "transformer"

    with timer("load data"):
        yado = pd.read_csv("input/yado.csv", dtype={"yad_no": int})
        train_log = pd.read_csv(
            "input/train_log.csv",
            dtype={"session_id": str, "seq_no": int, "yad_no": int},
        )
        train_label = pd.read_csv(
            "input/train_label.csv", dtype={"session_id": str, "yad_no": int}
        )
        train_log = train_log.merge(
            train_label, on="session_id", how="left", suffixes=["", "_cv"]
        )
        test_log = pd.read_csv(
            "input/test_log.csv",
            dtype={"session_id": str, "seq_no": int, "yad_no": int},
        )
        test_session = pd.read_csv("input/test_session.csv", dtype={"session_id": str})
        train_session = pd.DataFrame()
        train_session["session_id"] = train_log["session_id"].unique()
        all_log = pd.concat([train_log, test_log], axis=0).reset_index(drop=True)
        train_yads = set(train_log["yad_no"].unique())
        test_yads = set(test_log["yad_no"].unique())

        train_data = train_log.merge(yado, on="yad_no", how="left")
        test_data = test_log.merge(yado, on="yad_no", how="left")

        # フィルタリングするtrain_yadの選定
        filter_count = 3
        only_train_yads_before_filtered = train_yads - test_yads
        only_train_yad_df = pd.DataFrame(only_train_yads_before_filtered).rename(
            columns={0: "yad_no"}
        )
        only_train_yad_df["is_only_train"] = 1
        train_yad_count = (
            train_log.groupby("yad_no")
            .size()
            .reset_index()
            .rename(columns={0: "train_yad_count"})
        )
        only_train_yads = set(
            train_yad_count.merge(only_train_yad_df, on="yad_no", how="left")
            .query("is_only_train==1")
            .query(f"train_yad_count>={filter_count}")["yad_no"]
            .unique()
        )
        print(len(only_train_yads))

        train_gnn_sub_dict = joblib.load(f"input/train_{nn_model}_sub_dict.pkl")
        test_gnn_sub_dict = joblib.load(f"input/test_{nn_model}_sub_dict.pkl")

        # train_logで実際に予約した宿をひけるようにしておく
        map_reserved = defaultdict(int)
        for idx, rec in tqdm(train_label.iterrows()):
            session_id, yad_no_reserved = rec
            map_reserved[session_id] = yad_no_reserved

        map_session_yads_train = make_session_list(train_log)
        map_session_yads_test = make_session_list(test_log)
        map_session_yads_all = make_session_list(all_log)

    with timer("create sampler"):
        sml_freq_sampler = FrequentCdSampler("sml_cd", train_log, test_log, yado)
        lrg_freq_sampler = FrequentCdSampler("lrg_cd", train_log, test_log, yado)
        ken_freq_sampler = FrequentCdSampler("ken_cd", train_log, test_log, yado)

        train_sml_cd_dict = sml_freq_sampler.build_cd_dict(train_data)
        train_lrg_cd_dict = lrg_freq_sampler.build_cd_dict(train_data)
        train_ken_cd_dict = ken_freq_sampler.build_cd_dict(train_data)
        test_sml_cd_dict = sml_freq_sampler.build_cd_dict(test_data)
        test_lrg_cd_dict = lrg_freq_sampler.build_cd_dict(test_data)
        test_ken_cd_dict = ken_freq_sampler.build_cd_dict(test_data)

    with timer("create matrix"):
        # D[v][r]:= 「最後に宿vを閲覧して、宿rを予約した」セッションの件数
        next_booking_matrix = defaultdict(lambda: defaultdict(int))
        for session_id, viewed_yad_nos in tqdm(map_session_yads_train.items()):
            last_viewed = viewed_yad_nos[-1]
            reserved = map_reserved[session_id]
            next_booking_matrix[last_viewed][reserved] += 1

        # D[v][r]:= 「宿vを閲覧して、宿rを予約した」セッションの件数
        train_all_co_booking = (
            train_log.groupby(["yad_no", "yad_no_cv"])
            .size()
            .reset_index()
            .rename(columns={0: "co_booking_count"})
            .query("yad_no!=yad_no_cv")
        )

        co_booking_matrix = defaultdict(lambda: defaultdict(int))
        for record in tqdm(train_all_co_booking.itertuples()):
            yad_no = record[1]
            yad_no_cv = record[2]
            co_booking_count = record[3]
            co_booking_matrix[yad_no][yad_no_cv] = co_booking_count

        # (train用)D[v][r]:= 「宿vを閲覧して、宿rを閲覧した」セッションの件数
        train_co_visiting_matrix = defaultdict(lambda: defaultdict(int))
        for session_id, viewed_yad_nos in tqdm(map_session_yads_train.items()):
            unique_viewd_list = list(set(viewed_yad_nos))
            for idx in range(len(unique_viewd_list)):
                left_yad = unique_viewd_list[idx]
                for right_yad in unique_viewd_list[idx + 1 :]:
                    train_co_visiting_matrix[left_yad][right_yad] += 1
                    train_co_visiting_matrix[right_yad][left_yad] += 1

        # D[v][r]:= 「宿vを閲覧して、宿rを閲覧した」セッションの件数(test用)
        test_co_visiting_matrix = defaultdict(lambda: defaultdict(int))
        for session_id, viewed_yad_nos in tqdm(map_session_yads_test.items()):
            reserved = map_reserved[session_id]
            if reserved != 0:
                unique_viewd_list = list(
                    set(viewed_yad_nos + [reserved])
                )  # 予約した宿も閲覧したとみなす
            else:
                unique_viewd_list = list(set(viewed_yad_nos))

            for idx in range(len(unique_viewd_list)):
                left_yad = unique_viewd_list[idx]
                for right_yad in unique_viewd_list[idx + 1 :]:
                    test_co_visiting_matrix[left_yad][right_yad] += 1
                    test_co_visiting_matrix[right_yad][left_yad] += 1

    with timer("predict"):
        train_pred = create_topN_yads(
            train_session,
            map_session_yads_train,
            next_booking_matrix,
            co_booking_matrix,
            train_co_visiting_matrix,
            train_gnn_sub_dict,
            sml_cd_dict=train_sml_cd_dict,
            lrg_cd_dict=train_lrg_cd_dict,
            ken_cd_dict=train_ken_cd_dict,
            sml_freq_sampler=sml_freq_sampler,
            lrg_freq_sampler=lrg_freq_sampler,
            ken_freq_sampler=ken_freq_sampler,
        )
        mapk_score = mapk(train_label["yad_no"], train_pred, k=10)
        print(f"mapk score: {mapk_score}")

        test_pred = create_topN_yads(
            test_session,
            map_session_yads_test,
            next_booking_matrix,
            co_booking_matrix,
            test_co_visiting_matrix,
            test_gnn_sub_dict,
            sml_cd_dict=test_sml_cd_dict,
            lrg_cd_dict=test_lrg_cd_dict,
            ken_cd_dict=test_ken_cd_dict,
            sml_freq_sampler=sml_freq_sampler,
            lrg_freq_sampler=lrg_freq_sampler,
            ken_freq_sampler=ken_freq_sampler,
            only_train_yads=only_train_yads,
        )
    df_submit = pd.DataFrame(
        test_pred,
        columns=[
            "predict_0",
            "predict_1",
            "predict_2",
            "predict_3",
            "predict_4",
            "predict_5",
            "predict_6",
            "predict_7",
            "predict_8",
            "predict_9",
        ],
    )
    file_suffix = os.path.basename(__file__).split(".")[0]
    df_submit.to_csv(f"submission/submission_{file_suffix}_{mapk_score}.csv", index=False)
    print(df_submit)

    n = 10
    df_test = pd.DataFrame(test_pred, columns=[f"yad_no_{i}" for i in range(n)])
    df_test.to_csv(f"input/test_{file_suffix}.csv", index=False)

    df_train = pd.DataFrame(train_pred, columns=[f"yad_no_{i}" for i in range(n)])
    df_train.to_csv(f"input/train_{file_suffix}.csv", index=False)
