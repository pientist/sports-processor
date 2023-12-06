import os
import sys
from collections import Counter
from datetime import timedelta

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import shift
from tqdm import tqdm


class NBAProcessor:
    def __init__(self, traces):
        self.traces = traces
        self.players = np.sort([f"{c[:-2]}" for c in self.traces.columns if c[0] in ["A", "B"] and c.endswith("_x")])

    def label_attack_ids(self):
        self.traces["shot_clock"] = self.traces["shot_clock"].fillna(method="bfill").fillna(method="ffill")
        is_new_seq = pd.Series(0, index=self.traces.index)
        is_new_seq.loc[(self.traces["shot_clock"].diff() > 1) | (self.traces["quarter"].diff() > 0)] = 1
        self.traces["attack_id"] = is_new_seq.cumsum() + 1

    @staticmethod
    def interpolate_shot_clock(df):
        return df["shot_clock"].interpolate(limit_direction="both")

    def downsample_to_10fps(self):
        if "attack_id" not in self.traces.columns:
            self.label_attack_ids()

        # Upsample to 50 FPS
        traces_50fps = []

        for episode in self.traces["episode"].unique():
            ep_traces = self.traces[self.traces["episode"] == episode]
            t0 = ep_traces["time"].iloc[0]
            t1 = ep_traces["time"].iloc[-1] + 40
            time_range = pd.DataFrame(np.arange(t0, t1, 20), columns=["time"])
            upsampled = pd.merge(time_range, ep_traces, how="left")

            upsampled["attack_id"] = upsampled["attack_id"].fillna(method="bfill").fillna(method="ffill")
            if len(upsampled["attack_id"].unique()) == 1:
                upsampled["shot_clock"] = NBAProcessor.interpolate_shot_clock(upsampled)
            else:
                shot_clock_50fps = upsampled.groupby("attack_id").apply(NBAProcessor.interpolate_shot_clock)
                upsampled["shot_clock"] = shot_clock_50fps.reset_index(level=0, drop=True)

            traces_50fps.append(upsampled.interpolate(limit_direction="both"))

        traces_50fps = pd.concat(traces_50fps)

        # Downsample to 10 FPS
        traces_10fps = traces_50fps[traces_50fps["time"] % 100 == 0].copy()
        period_cols = ["quarter", "phase", "episode", "attack_id"]
        traces_10fps[period_cols] = traces_10fps[period_cols].astype(int)
        traces_10fps["time"] = traces_10fps["time"] / 1000
        traces_10fps[["time_left", "shot_clock"]] = traces_10fps[["time_left", "shot_clock"]].round(1)

        self.traces = traces_10fps.reset_index(drop=True)

    def convert_to_meters(self):
        pos_cols = [c for c in self.traces.columns if c[-2:] in ["_x", "_y", "_z"]]
        self.traces[pos_cols] = self.traces[pos_cols] * 0.3

    def calc_single_agent_velocities(self, p: str, remove_outliers=True, smoothing=True):
        if remove_outliers:
            MAX_SPEED = 12
            MAX_ACCEL = 8

        if smoothing:
            W_LEN = 11
            P_ORDER = 2

        valid_traces = self.traces.dropna(subset=[f"{p}_x"]).copy()

        for episode in valid_traces["episode"].unique():
            x = valid_traces.loc[valid_traces["episode"] == episode, f"{p}_x"]
            y = valid_traces.loc[valid_traces["episode"] == episode, f"{p}_y"]

            if smoothing:
                x = pd.Series(signal.savgol_filter(x, window_length=21, polyorder=P_ORDER), index=x.index)
                y = pd.Series(signal.savgol_filter(y, window_length=21, polyorder=P_ORDER), index=y.index)

            vx = np.diff(x.values, prepend=x.iloc[0]) / 0.1
            vy = np.diff(y.values, prepend=y.iloc[0]) / 0.1

            if remove_outliers:
                speeds = np.sqrt(vx**2 + vy**2)
                is_speed_outlier = speeds > MAX_SPEED
                is_accel_outlier = np.abs(np.diff(speeds, append=speeds[-1]) / 0.1) > MAX_ACCEL
                is_outlier = is_speed_outlier | is_accel_outlier | shift(is_accel_outlier, 1, cval=True)
                vx = pd.Series(np.where(is_outlier, np.nan, vx)).interpolate(limit_direction="both").values
                vy = pd.Series(np.where(is_outlier, np.nan, vy)).interpolate(limit_direction="both").values

            if smoothing:
                vx = signal.savgol_filter(vx, window_length=W_LEN, polyorder=P_ORDER)
                vy = signal.savgol_filter(vy, window_length=W_LEN, polyorder=P_ORDER)

            self.traces.loc[x.index, [f"{p}_x", f"{p}_y", f"{p}_vx", f"{p}_vy"]] = np.stack([x, y, vx, vy]).round(6).T

    def calc_velocities(self, remove_outliers=True, smoothing=True):
        data_cols = [f"{p}{x}" for p in self.players for x in ["_x", "_y", "_vx", "_vy"]]
        if "ball_x" in self.traces.columns:
            data_cols += ["ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy"]

        new_cols = [c for c in data_cols if c not in self.traces.columns]
        self.traces = pd.concat([self.traces, pd.DataFrame(index=self.traces.index, columns=new_cols)], axis=1)

        self.calc_single_agent_velocities("ball", remove_outliers=False, smoothing=False)
        for p in tqdm(self.players, desc="Calculating player velocities"):
            self.calc_single_agent_velocities(p, remove_outliers, smoothing)

        meta_cols = [c for c in self.traces.columns if c.split("_")[-1] not in ["x", "y", "z", "vx", "vy"]]
        self.traces = self.traces[meta_cols + data_cols]
        self.traces[data_cols] = self.traces[data_cols].astype(float)

    # def interpolate_pauses(self):
    #     pos_cols = [c for c in self.traces.columns if c[-2:] in ["_x", "_y", "_z"]]
    #     traces_interp = []

    #     for phase in self.traces["phase"].unique():
    #         phase_traces = self.traces[self.traces["phase"] == phase]

    #         t0 = phase_traces["time"].iloc[0]
    #         t1 = phase_traces["time"].iloc[-1] + 40
    #         time_range = pd.DataFrame(np.arange(t0, t1, 40), columns=["time"])
    #         time_range["quarter"] = phase_traces["quarter"].iloc[0]
    #         time_range["phase"] = phase

    #         phase_traces = pd.merge(phase_traces, time_range, how="right")
    #         phase_traces["episode"] = phase_traces["episode"].fillna(0).astype(int)
    #         prev_seq = phase_traces["attack_id"].fillna(method="ffill")
    #         next_seq = phase_traces["attack_id"].fillna(method="bfill")
    #         phase_traces["attack_id"] = np.where(prev_seq == next_seq, prev_seq, 0).astype(int)

    #         for attack_id in phase_traces["attack_id"].unique():
    #             seq_traces = phase_traces[phase_traces["attack_id"] == attack_id]
    #             phase_traces.loc[seq_traces.index, "shot_clock"] = seq_traces["shot_clock"].interpolate(
    #                 limit_direction="both"
    #             )

    #         phase_traces[["time_left"] + pos_cols] = phase_traces[["time_left"] + pos_cols].interpolate()
    #         traces_interp.append(phase_traces)

    #     self.traces = pd.concat(traces_interp)
