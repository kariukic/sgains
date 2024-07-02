import os
import logging

import numpy as np
import astropy.stats as astats
from astropy.stats import mad_std
from astropy.coordinates import SkyCoord
import astropy.units as u

from abc import ABC, abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from plotting_utils import ColorbarInnerPosition, ColorbarSetting
from processor import GainsUtils
from filesIO import FileIOHandler as fh


class GainsPlotMaker(ABC):

    def __init__(self, data, freqs, stations, clusters_ids, eff_nr) -> None:
        self.data = data
        self.clusters_ids = clusters_ids
        self.eff_nr = eff_nr
        self.stations = stations
        self.freqs = freqs
        self.fmin = self.freqs.min()
        self.fmax = self.freqs.max()
        self.pols = {"XX": [0, 0], "YY": [1, 1], "XY": [0, 1], "YX": [1, 0]}
        self.pol_stokes = {"I": [0, 0], "V": [1, 1], "U": [0, 1], "Q": [1, 0]}

    @staticmethod
    def reshape_gains_data(data):
        data = data[:, :, :, :, :, 0] + 1j * data[:, :, :, :, :, 1]
        data = data.transpose(0, 1, 2, 4, 3)
        data = data.reshape(*(list(data.shape[:-1]) + [2, 2]))

    def get_gains(self, data, cluster, station, eff_nr):
        if cluster == 0:
            c_id = slice(None, int(eff_nr.cumsum()[cluster]))
        else:
            c_id = slice(
                int(eff_nr.cumsum()[cluster - 1]), int(eff_nr.cumsum()[cluster])
            )

        s = data[:, station, :, c_id]
        s = s.transpose(0, 2, 1, 3, 4)
        s = np.concatenate(s)

        return s


class PlotGains:
    def __init__(self, data, freqs, time, stations, clusters_ids, eff_nr):
        self.data = data
        self.freqs = freqs
        self.time = time / 3600
        self.stations = stations
        self.clusters_ids = clusters_ids
        self.eff_nr = eff_nr
        self.pols = {"XX": [0, 0], "YY": [1, 1], "XY": [0, 1], "YX": [1, 0]}

        logging.debug(f"data, {data.shape}")

    @staticmethod
    def avg_gg(a, n=2):
        end = n * int(a.shape[1] / n)
        return np.mean(a[:, :end, :].reshape((a.shape[0], -1, n, a.shape[2])), axis=2)

    def baselines_abs_and_phase(
        self, station_pairs, cluster_id, out_dir=None, prefix=""
    ):
        d = (
            self.data[:, :, :, :, cluster_id, 0]
            + 1j * self.data[:, :, :, :, cluster_id, 1]
        )
        self.d = d.reshape(*(list(d.shape[:-1]) + [2, 2]))
        logging.debug(f"d, {self.d.shape}")
        fig, axs = plt.subplots(
            ncols=4,
            nrows=len(station_pairs),
            figsize=(12, 2.5 * len(station_pairs)),
            sharex=True,
            sharey=False,
        )
        fig2, axs2 = plt.subplots(
            ncols=4,
            nrows=len(station_pairs),
            figsize=(12, 2.5 * len(station_pairs)),
            sharex=True,
            sharey=False,
        )

        for i, (s1, s2) in enumerate(station_pairs):
            g1 = self.d[:, s1]
            g2 = self.d[:, s2]
            c = np.array([[1, 0], [0, 1]])

            for j, (name, (k, l)) in enumerate(self.pols.items()):
                gg = GainsUtils.g_mul(g1, g2, c)[:, :, k, l]

                if i == 0 and j == 0:
                    vmax = np.median(abs(gg)) + 10 * astats.mad_std(abs(gg).flatten())
                    vmin = 1e-1 * vmax

                g_map = axs[i, j].pcolormesh(
                    self.freqs,
                    self.time,
                    abs(gg),
                    norm=LogNorm(vmin=vmin, vmax=vmax),
                )
                p_map = axs2[i, j].pcolormesh(
                    self.freqs,
                    self.time,
                    np.angle(gg),
                    vmin=-np.pi,
                    vmax=np.pi,
                )

                axs[i, j].text(
                    0.05,
                    0.15,
                    "Median: %.4g" % np.median(abs(gg)),
                    transform=axs[i, j].transAxes,
                    va="top",
                    ha="left",
                )
                axs2[i, j].text(
                    0.05,
                    0.15,
                    "Median: %.4g" % np.median(np.angle(gg)),
                    transform=axs2[i, j].transAxes,
                    va="top",
                    ha="left",
                )

                for ax in [axs, axs2]:
                    if j != 0:
                        ax[i, j].yaxis.set_ticklabels([])
                    if j == 0:
                        ax[i, j].set_ylabel("Time [hrs]\n(stations %s - %s)" % (s1, s2))
                    if i == len(station_pairs) - 1:
                        ax[i, j].set_xlabel("Frequency [MHz]")
                    if i == 0:
                        ax[i, j].set_title("g1g2*_%s" % name)
                    if j == len(self.pols) - 1 and i == 0:
                        cbs = ColorbarSetting(
                            ColorbarInnerPosition(height="70%", pad=0.7)
                        )
                        cbs.add_colorbar(g_map, axs[i, j])
                        cbs = ColorbarSetting(
                            ColorbarInnerPosition(height="70%", pad=0.7)
                        )
                        cbs.add_colorbar(p_map, axs2[i, j])

        fig.suptitle("Sagecal solutions abs(g1g2*)", va="bottom", y=1)
        fig2.suptitle("Sagecal solutions phase(g1g2*)", va="bottom", y=1)

        fig.tight_layout(pad=0.4)
        fig2.tight_layout(pad=0.4)

        if out_dir:
            fig.savefig(os.path.join(out_dir, f"{prefix}_sols_abs_gg.pdf"))
            fig2.savefig(os.path.join(out_dir, f"{prefix}_sols_phase_gg.pdf"))

        return fig, fig2

    def mean_and_std(
        self,
        avg_type="stations",
        n_time_avg=40,
        max_station=62,
        cluster_id=0,
        prefix="",
        out_dir=None,
    ):
        d = (
            self.data[:, :, :, :, cluster_id, 0]
            + 1j * self.data[:, :, :, :, cluster_id, 1]
        )
        self.d = d.reshape(*(list(d.shape[:-1]) + [2, 2]))
        logging.debug(f"d, {self.d.shape}")

        c = np.array([[1, 0], [0, 1]])
        gg = np.array(
            [
                GainsUtils.g_mul(self.d[:, i], self.d[:, i], c)
                for i in range(self.d.shape[1])
            ]
        )

        if avg_type == "stations":
            avg_col_idx = 0
            other_col = "Frequency [MHz]"
            extent = [self.freqs[0], self.freqs[-1], 0, self.time.max()]
        elif avg_type == "frequency":
            avg_col_idx = 2
            other_col = "Stations"
            extent = [0, gg.shape[0] - 1, 0, self.time.max()]

        fig, axs = plt.subplots(
            ncols=4, nrows=4, figsize=(12, 10), sharex=True, sharey=False
        )

        for j, (name, (k, l)) in enumerate(self.pols.items()):
            logging.debug(
                f"{name}, {k}, {l}, {gg.shape, gg[:max_station, :, :, k, l].shape}"
            )
            g_m = np.median(np.abs(gg[:max_station, :, :, k, l]), axis=avg_col_idx)
            g_d_t = astats.mad_std(
                np.real(np.diff(gg[:max_station, :, :, k, l], axis=1)), axis=avg_col_idx
            ) * np.sqrt(0.5)
            g_d_f = astats.mad_std(
                np.real(np.diff(gg[:max_station, :, :, k, l], axis=2)), axis=avg_col_idx
            ) * np.sqrt(0.5)
            g_d_f_avg = astats.mad_std(
                np.real(
                    self.avg_gg(
                        np.diff(gg[:max_station, :, :, k, l], axis=2), n_time_avg
                    )
                ),
                axis=avg_col_idx,
            ) * np.sqrt(0.5 * n_time_avg)

            if j == 0:
                vmax = np.median(g_m) + 10 * astats.mad_std(g_m.flatten())
                vmin = 1e-3 * vmax

            for ax, a in zip(axs[:, j], [g_m, g_d_t, g_d_f, g_d_f_avg]):
                if avg_type == "frequency":
                    a = a.T

                im_map = ax.imshow(
                    a, aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent
                )
                ax.text(
                    0.05,
                    0.15,
                    "Median: %.4g" % np.median(a),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                )

            axs[3, j].set_xlabel(other_col)
            axs[0, j].set_title("gg*_%s" % name)

        for ax in axs[:, 1:].flatten():
            ax.yaxis.set_ticklabels([])

        axs[0, 0].set_ylabel("Mean along %s\n\nTime [hrs]" % (avg_type))
        axs[1, 0].set_ylabel(
            "Std along %s\nof the time diff gains\n\nTime [hrs]" % (avg_type)
        )
        axs[2, 0].set_ylabel(
            "Std along %s\nof the SB diff gains\n\nTime [hrs]" % (avg_type)
        )
        axs[3, 0].set_ylabel(
            "400 s time avg\nstd along %s\nof the SB diff gains\n\nTime [hrs]"
            % (avg_type)
        )

        cbs = ColorbarSetting(ColorbarInnerPosition(height="70%", pad=0.7))
        cbs.add_colorbar(im_map, axs[0, j])

        fig.tight_layout(pad=0.4)

        if out_dir:
            fig.savefig(
                os.path.join(out_dir, f"{prefix}_sol_mean_std_{avg_type}_gg.pdf")
            )

        return fig

    def dynamic_spectrum(
        self,
        gain_data,
        dd_stations=[],
        dd_clusters=[],
        clusters_names=[],
        vmin=1e-1,
        vmax=1e1,
        pol_name="XX",
        action_fct=np.abs,
        log_norm=True,
        file_name=None,
        out_dir=None,
        # ref_phase=False,
    ):
        dd_stations, dd_clusters, clusters_names = (
            self.parse_dd_clusters_and_stations_to_plot(
                dd_stations, dd_clusters, clusters_names
            )
        )
        x, y = len(dd_stations), len(dd_clusters)

        fig, axs = plt.subplots(
            ncols=x,
            nrows=y,
            facecolor="white",
            figsize=(3 * x, 2.5 * y),
            sharex=True,
            sharey=False,
            gridspec_kw={"wspace": 0.03, "hspace": 0.03},
        )
        if log_norm:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # if ref_phase:
        #     refph = action_fct(gain_data[(dd_stations[0], dd_clusters[0])][pol_name])

        for i, (cluster, cname) in enumerate(zip(dd_clusters, clusters_names)):

            if (dd_stations[0], cluster) not in gain_data.keys():
                logging.debug(f"{(dd_stations[0], cluster)} not in {gain_data.keys()}")
                continue

            for j, station in enumerate(dd_stations):
                g = action_fct(gain_data[(station, cluster)][pol_name])
                # if ref_phase:
                #     g -= refph
                #     g[g > np.pi] -= 2 * np.pi
                #     g[g < -np.pi] += 2 * np.pi

                g_map = axs[i, j].imshow(
                    g, norm=norm, aspect="auto"
                )  # extent=[freqs[0], freqs[-1], 0, g.shape[0] - 1]
                if j != 0:
                    axs[i, j].yaxis.set_ticklabels([])
                if j == 0:
                    axs[i, j].set_ylabel(f"Time slot\n(cluster {cname})")
                if i == len(self.clusters_ids) - 1:
                    axs[i, j].set_xlabel("Frequency [MHz]")
                if i == 0:
                    axs[i, j].set_title(f"Station {station}")

                if j == len(dd_stations) - 1 and i == 0:
                    cbs = ColorbarSetting(ColorbarInnerPosition(height="70%", pad=0.7))
                    cbs.add_colorbar(g_map, axs[i, j])

                axs[i, j].text(
                    0.05,
                    0.95,
                    "Median: %.4g" % np.median(g),
                    transform=axs[i, j].transAxes,
                    va="top",
                    ha="left",
                )

        # fig.suptitle(pol_name)
        fig.tight_layout(pad=0.4)
        if file_name:
            logging.info(f"Saving DD plot to {file_name}.pdf")
            fig.savefig(os.path.join(out_dir, f"{file_name}.pdf"), bbox_inches="tight")

        return fig

    def parse_dd_clusters_and_stations_to_plot(
        self, stations_to_plot, clusters_to_plot, clusternames
    ):
        if not stations_to_plot:
            stations = self.stations
        else:
            # stations = [self.stations[int(i)] for i in stations_to_plot[0].split(",")]
            stations = [self.stations[int(i)] for i in stations_to_plot.split(",")]

        if not clusters_to_plot:
            clusters, clusters_names = self.clusters_ids, list(
                range(len(self.clusters_ids))
            )
        else:
            # clusters = [int(clst) for clst in clusters_to_plot[0].split(",")]
            clusters = [int(clst) for clst in clusters_to_plot.split(",")]

            if isinstance(clusternames, list):
                clusters_names = [str(clstn) for clstn in clusternames]

            elif isinstance(clusternames, str):
                clusters_names = [str(clstn)[1:-1] for clstn in clusternames.split(",")]

            else:
                clusters_names = [str(s) for s in clusters]

            logging.debug(f"Cluster names: {clusters_names}, indices: {clusters}")

            assert isinstance(clusters_names, list)
            assert len(clusters) == len(clusters_names)

        logging.info(f"Plotting stations: {stations}")
        logging.info(f"Plotting cluster IDs: {clusters}")
        logging.info(f"Plotting cluster names: {clusters_names}")

        return stations, clusters, clusters_names

    def cluster_gains_noise(
        self,
        gain_data,
        clusters_info,
        station: str = None,
        pol="XX",
        discarded: list = [
            84,
            85,
            88,
            91,
            97,
            100,
            111,
            112,
            115,
            117,
            118,
            119,
            120,
            121,
        ],
    ):

        # discarded = list( set(self.clusters_ids.keys()).intersection(set(discarded)) )

        logging.debug(f"Plotting gains noise for station {station}")

        s_gain = np.array(
            [
                mad_std(
                    np.diff(np.array(gain_data[(station, c)][pol]).real, axis=0)
                ).mean()
                for c in range(len(self.clusters_ids))
            ]
        )
        m_gain = np.median(
            np.array(
                [
                    np.abs(np.array(gain_data[(station, c)][pol])).mean(axis=0)
                    for c in range(len(self.clusters_ids))
                ]
            ),
            axis=1,
        )

        c_sel = np.where(s_gain > 1e-3)[0]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

        cm1 = plt.get_cmap("viridis")
        norm = LogNorm(vmin=1, vmax=10)
        ncp = SkyCoord(ra=0 * u.deg, dec=90.0 * u.deg)

        for c_id in c_sel:
            c_sources = clusters_info[c_id]["sources"].values()
            c_pos = SkyCoord(
                ra=np.array([s["Ra"] for s in c_sources]) * u.rad,
                dec=np.array([s["Dec"] for s in c_sources]) * u.rad,
            )
            c_pos_offsets = ncp.spherical_offsets_to(c_pos)
            im = ax.scatter(
                -c_pos_offsets[0].deg,
                c_pos_offsets[1].deg,
                norm=norm,
                c=[m_gain[c_id] / s_gain[c_id]] * len(c_pos_offsets[0].deg),
                cmap=cm1,
                s=0.5,
            )
        cb = plt.colorbar(im, ax=ax)
        cb.ax.set_xlabel("gain SNR", labelpad=10)

        for uv in [4.5, 9, 13.5]:  # change for other redshifts
            ax.add_artist(
                plt.Circle([0, 0], uv, ls="--", fc=None, ec="k", fill=False, lw=0.4)
            )
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_xlabel("l [degrees]")
            ax.set_ylabel("m [degrees]")
        ax.set_title(f"Station: {station}")
        fig.tight_layout()

        return fig

    def cluster_stats(self, gain_data, pol="XX"):

        med, rms_dt, rms_dnu = self.get_stats(gain_data, pol=pol, axis="clusters")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            figsize=(17, 12), dpi=300, nrows=4, sharex=True
        )

        _ = ax2.boxplot(
            med.T, positions=[*self.clusters_ids.keys()], showfliers=True, whis=(0, 100)
        )
        _ = ax3.boxplot(
            rms_dt.T,
            positions=[*self.clusters_ids.keys()],
            showfliers=True,
            whis=(0, 100),
        )
        _ = ax4.boxplot(
            rms_dnu.T,
            positions=[*self.clusters_ids.keys()],
            showfliers=True,
            whis=(0, 100),
        )

        ax1.plot([*self.clusters_ids.keys()], self.eff_nr, ls="", marker="o")
        plt.xticks(rotation=70, fontsize=7)

        ax4.set_xlabel("Cluster ID")
        ax1.set_ylabel("Effective number of clusters")
        ax2.set_ylabel(f"Medn {pol}")
        ax3.set_ylabel(f"Time difference {pol} rms")
        ax4.set_ylabel(f"Frequency difference {pol} rms")

        ax2.set_ylim(bottom=0)
        for ax in (ax1, ax2, ax3, ax4):
            ax.grid(axis="x")

        fig.tight_layout()

        return fig, rms_dt, med, rms_dnu

    def get_stats(self, gain_data, pol, axis="stations"):
        rms_dt = []
        med = []
        rms_dnu = []

        if axis == "clusters":
            for c in np.arange(0, len(self.clusters_ids)):
                rms_dt.append(
                    [
                        np.std(np.diff(gain_data[(k, c)][pol], axis=0))
                        for k in self.stations
                    ]
                )
                med.append(
                    [np.median(abs(gain_data[(k, c)][pol])) for k in self.stations]
                )
                rms_dnu.append(
                    [
                        np.std(np.diff(gain_data[(k, c)][pol], axis=1))
                        for k in self.stations
                    ]
                )

        elif axis == "stations":
            c_ii = range(len(self.clusters_ids))
            for s_id in self.stations:
                rms_dt.append(
                    [np.std(np.diff(gain_data[(s_id, k)][pol], axis=0)) for k in c_ii]
                )
                med.append([np.median(abs(gain_data[(s_id, k)][pol])) for k in c_ii])
                rms_dnu.append(
                    [np.std(np.diff(gain_data[(s_id, k)][pol], axis=1)) for k in c_ii]
                )

        rms_dt = np.array(rms_dt)
        med = np.array(med)
        rms_dnu = np.array(rms_dnu)

        return med, rms_dt, rms_dnu

    def stations_stats(self, gain_data, pol="XX"):

        med, rms_dt, rms_dnu = self.get_stats(gain_data, pol, axis="stations")

        fig, (ax2, ax3, ax4) = plt.subplots(
            figsize=(17, 10), dpi=300, nrows=3, sharex=True
        )

        st_ids = list(range(len(self.stations)))

        ax2.boxplot(med.T, positions=st_ids, showfliers=True, whis=(0, 100))
        ax3.boxplot(rms_dt.T, positions=st_ids, showfliers=True, whis=(0, 100))
        ax4.boxplot(rms_dnu.T, positions=st_ids, showfliers=True, whis=(0, 100))

        plt.xticks(rotation=70, fontsize=9)

        ax4.set_xlabel("Station ID")
        ax2.set_ylabel(f"Medn {pol}")
        ax3.set_ylabel(f"Time difference {pol} rms")
        ax4.set_ylabel(f"Frequency difference {pol} rms")

        ax2.set_ylim(bottom=0)
        ax4.set_xticks(st_ids)
        ax4.set_xticklabels(self.stations)

        for ax in (ax2, ax3, ax4):
            ax.grid(axis="x")

        fig.tight_layout()

        return fig, rms_dt, med, rms_dnu
