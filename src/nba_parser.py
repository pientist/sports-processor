import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


class NBAParser:
    def __init__(self, match_id: str, raw_dir: str = "data/nba_raw_json"):
        self.raw_df = self.read_raw_data(match_id, raw_dir)
        self.metadata = dict()
        self.data = None

    @staticmethod
    def read_raw_data(match_id: str, raw_dir: str = "data/nba_raw_json"):
        df = pd.read_json(f"{raw_dir}/{match_id}.json")
        return df[df.apply(lambda row: len(row["events"]["moments"]) != 0, axis=1)].reset_index(drop=True)

    def set_match_info(self, verbose=False):
        first_episode = self.raw_df.iloc[0]

        self.metadata["match_id"] = int(first_episode["gameid"])
        self.metadata["match_date"] = first_episode["gamedate"]

        self.metadata["home_team"] = first_episode["events"]["home"].copy()
        self.metadata["away_team"] = first_episode["events"]["visitor"].copy()
        self.metadata["home_team"].pop("players")
        self.metadata["away_team"].pop("players")

        home_players = first_episode["events"]["home"]["players"]
        away_players = first_episode["events"]["visitor"]["players"]
        home_player_dict = {f"A{int(player['jersey']):02d}": player for player in home_players}
        away_player_dict = {f"B{int(player['jersey']):02d}": player for player in away_players}
        self.metadata["players"] = {**home_player_dict, **away_player_dict}

        if verbose:
            home_team = self.metadata["home_team"]["name"]
            away_team = self.metadata["away_team"]["name"]
            print(f"\n{self.metadata['match_date']} - {away_team} at {home_team}")

    def get_time_periods(self, start_index=0, end_index=float("inf"), relative=True):
        times_left = []
        unixtimes = []

        if relative:
            offset_time_left = max([self.raw_df.at[i, "events"]["moments"][0][2] for i in self.raw_df.index])
            offset_unixtime = min([self.raw_df.at[i, "events"]["moments"][-1][1] for i in self.raw_df.index])
        else:
            offset_time_left = 0
            offset_unixtime = 0

        for i in self.raw_df.loc[start_index:end_index].index:
            moments = self.raw_df.at[i, "events"]["moments"]

            start_ut = moments[0][1] - offset_unixtime
            end_ut = moments[-1][1] - offset_unixtime
            unixtimes.append((start_ut, end_ut))

            start_tl = round(-moments[0][2] + offset_time_left, 2)
            end_tl = round(-moments[-1][2] + offset_time_left, 2)
            times_left.append((start_tl, end_tl))

        return unixtimes, times_left

    def remove_nested_events(self):
        unixtimes, _ = self.get_time_periods(relative=False)
        to_remove = set()

        for i, (start_ut, end_ut) in enumerate(unixtimes):
            for j in range(i):
                prev_start_ut, prev_end_ut = unixtimes[j]
                if j in to_remove:
                    continue
                elif prev_start_ut >= start_ut and prev_end_ut <= end_ut:
                    to_remove.add(j)
                elif prev_start_ut <= start_ut and prev_end_ut >= end_ut:
                    to_remove.add(i)
                    break

        self.raw_df = self.raw_df.drop(list(to_remove)).reset_index(drop=True)

    def moment_pos_to_dict(self):
        for i in self.raw_df.index:
            for moment_idx, moment_data in enumerate(self.raw_df.at[i, "events"]["moments"]):
                self.raw_df.at[i, "events"]["moments"][moment_idx][5] = {x[1]: x[2:] for x in moment_data[5]}

    def format_traces(self):
        data = []

        for i in tqdm(self.raw_df.index):
            for moment_record in self.raw_df.at[i, "events"]["moments"]:
                if len(moment_record[5].keys()) < 10:
                    continue

                moment_data = {
                    "quarter": moment_record[0],
                    "time": round(moment_record[1] / 40) * 40,
                    "time_left": moment_record[2],
                    "shot_clock": moment_record[3],
                    "phase": 0,
                    "episode": 0,
                }

                if -1 in moment_record[5].keys():
                    moment_data["ball_x"] = moment_record[5][-1][0]
                    moment_data["ball_y"] = moment_record[5][-1][1]
                    moment_data["ball_z"] = moment_record[5][-1][2]
                else:
                    moment_data["ball_x"] = np.nan
                    moment_data["ball_y"] = np.nan
                    moment_data["ball_z"] = np.nan

                for p, player_info in self.metadata["players"].items():
                    player_id = player_info["playerid"]
                    if player_id in moment_record[5].keys():
                        moment_data[f"{p}_x"] = moment_record[5][player_id][0]
                        moment_data[f"{p}_y"] = moment_record[5][player_id][0]
                    else:
                        moment_data[f"{p}_x"] = np.nan
                        moment_data[f"{p}_y"] = np.nan

                data.append(moment_data)

        self.data = pd.DataFrame(data).groupby("time", as_index=False).last()
        self.data["time"] -= self.data.at[0, "time"]

    def split_phases(self):
        player_x_cols = [c for c in self.data.columns if c[0] in ["A", "B"] and c[-2:] == "_x"]
        phase_count = 1
        for quarter in self.data["quarter"].unique():
            quarter_traces = self.data[self.data["quarter"] == quarter]
            new_phase = quarter_traces[player_x_cols].notna().astype(int).diff().abs().sum(axis=1).clip(0, 1)
            self.data.loc[quarter_traces.index, "phase"] = new_phase.astype(int).cumsum() + phase_count
            phase_count += new_phase.astype(int).sum() + 1

    def split_episodes(self, episode_margin=5):
        episode_count = 1
        for phase in self.data["phase"].unique():
            phase_traces = self.data[self.data["phase"] == phase]
            new_episode = (phase_traces["time"].diff() > episode_margin * 1000).astype(int)
            self.data.loc[phase_traces.index, "episode"] = new_episode.cumsum() + episode_count
            episode_count += new_episode.sum() + 1

    def save_data(self, save=False, verbose=False, data_dir="data/nba_data", metadata_dir="data/nba_metadata"):
        if save:
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)
            if not os.path.exists(metadata_dir):
                os.mkdir(metadata_dir)

            match_id = self.metadata["match_id"]
            self.data.to_csv(f"{data_dir}/{match_id}.csv", index=False)
            with open(f"{metadata_dir}/{match_id}.json", "w") as json_file:
                json.dump(self.metadata, json_file, indent=4)

            if verbose:
                print(f"Data saved in {data_dir}/{match_id}.csv.")
                print(f"Metadata saved in {data_dir}/{match_id}.json.")

    def run(self, save=False, verbose=False, data_dir="data/nba_traces", metadata_dir="data/nba_metadata"):
        self.set_match_info(verbose)
        self.remove_nested_events()
        self.moment_pos_to_dict()
        self.format_traces()
        self.split_phases()
        self.split_episodes()
        self.save_data(save, verbose, data_dir, metadata_dir)
