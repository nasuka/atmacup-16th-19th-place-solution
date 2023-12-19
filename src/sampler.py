import pandas as pd


class FrequentCdSampler:
    """同じcdを持つ宿でなおかつtrain/testでよく出現している宿を抽出する"""

    def __init__(
        self,
        cd_type: str,
        train_merged: pd.DataFrame,
        test_merged: pd.DataFrame,
        yad: pd.DataFrame,
    ) -> None:
        assert cd_type in {"sml_cd", "lrg_cd", "ken_cd", "wid_cd"}
        self.cd_type = cd_type
        self.train_merged = train_merged
        self.test_merged = test_merged
        self.yad = yad
        self.yad_score_dict = self._build_yad_score()

    def _build_yad_score(self) -> dict[str, pd.DataFrame]:
        train_yad_score = self.train_merged["yad_no"].value_counts().reset_index()
        train_yad_score = train_yad_score.merge(self.yad, how="left", on="yad_no")

        test_yad_score = self.test_merged["yad_no"].value_counts().reset_index()
        test_yad_score = test_yad_score.merge(self.yad, how="left", on="yad_no")

        all_yad_score = (
            pd.concat([self.train_merged, self.test_merged])["yad_no"]
            .value_counts()
            .reset_index()
        )
        all_yad_score = all_yad_score.merge(self.yad, how="left", on="yad_no")
        return {"train": train_yad_score, "test": test_yad_score, "all": all_yad_score}

    def sample(self, target_cd: str, k: int, data_type: str) -> list[int]:
        assert data_type in {"train", "test", "all"}
        candidates = (
            self.yad_score_dict[data_type]
            .query(f"{self.cd_type}=='{target_cd}'")["yad_no"]
            .tolist()
        )
        if candidates == 0:
            raise ValueError(f"no candidates: {target_cd} for {self.cd_type}")
        return candidates[:k]

    def sample_by_session_id(
        self, session_id: str, k: int, cd_dict: dict[str, int], data_type: str
    ) -> list[int]:
        # 最頻値以外を使う場合はここを変更する
        target_cd = cd_dict.get(session_id)
        return self.sample(target_cd, k, data_type)

    def build_cd_dict(self, log: pd.DataFrame) -> dict[str, int]:
        """session_idごとに最頻値のcdを取得する。train/testどちらのlogもとりうる"""
        cd_mode_df = (
            log.groupby("session_id")[self.cd_type]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
        )
        cd_dict = dict(zip(cd_mode_df["session_id"], cd_mode_df[self.cd_type]))
        return cd_dict
