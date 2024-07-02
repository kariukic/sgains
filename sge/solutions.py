"""
Module to read, transform, write and plot sagecal solutions
"""

import os
import numpy as np

from filesIO import FileIOHandler, CollectGainsFiles
from processor import GainsUtils, makeGainsDict
from plotter import PlotGains

import logging
import logging.config

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.config"))
logging = logging.getLogger("sagecal_solutions")


class Solutions:

    def __init__(
        self,
        cluster_file: str,
        sample_ms: str,
        solsfiles: str = None,
        nodes: str = None,
        obsid: str = None,
        indir: str = None,
        pid: str = None,
        cluster_to_read=[],
    ) -> None:

        self.file_handler = FileIOHandler()
        self.gains_processor = GainsUtils()

        self.cluster_file = cluster_file
        self.cluster_to_read = cluster_to_read
        self.sample_ms = sample_ms

        self.eff_nr = self.get_eff_nr()

        if solsfiles:
            logging.info(f"Reading {solsfiles}")
            self.solutions_files_list = self.file_handler.read_lines_from_file(
                solsfiles
            )
        elif all([obsid, nodes, indir]):
            logging.info(f"Collecting solutions files")
            gains_collector = CollectGainsFiles(
                obsid=obsid, nodes=nodes, indir=indir, pid=pid
            )
            self.solutions_files_list = gains_collector.collect_solutions_files()

        logging.info(f"Combining solutions into numpy format")
        (
            self.data,
            self.all_freqs,
            self.nClustEff,
            self.bandwidth,
            self.sol_timestep,
            self.nClust,
        ) = self.gains_processor.convert_solutions(
            self.solutions_files_list, self.cluster_to_read
        )

        _ntimes = self.data.shape[0]
        self.time_s = np.linspace(0, self.sol_timestep * _ntimes, _ntimes) * 60
        logging.info(f"Getting observation metadata from: {self.sample_ms}")
        (
            self.timerange,
            self.data_timestep,
            self.pointing,
            self.stations,
            self.station_pos,
        ) = self.file_handler.getMSinfo(self.sample_ms)

        self.meantimestep = (self.timerange[1] - self.timerange[0]) / (
            self.data.shape[0] - 1
        )
        logging.info(f"MS mean timestep (s): {self.meantimestep}")
        logging.info("Reading done!")

        self.cluster_ids = self.file_handler.get_cluster_ids(self.cluster_file)

        self.plot = PlotGains(
            self.data,
            self.all_freqs,
            self.time_s,
            self.stations,
            self.cluster_ids,
            self.eff_nr,
        )

    def get_clusters_details(self, sky_model):
        clusters_info, _ = self.file_handler.getClusters(self.cluster_file, sky_model)
        return clusters_info

    def make_gains_dict(self):
        mgd = makeGainsDict(self.data, self.stations, self.eff_nr)
        return mgd.get_all_gains(clusters=self.cluster_ids)

    def get_eff_nr(self):
        nrs = []
        for data in open(self.cluster_file):
            if data.strip() and not data.strip()[0] == "#":
                nr = int(data.strip().split()[1])
                nrs.append(nr)
        return np.asarray(nrs)

    # def plot_bl_gains(self, baselines, cluster_id=0):
    #     pg = PlotGains(self.data, self.all_freqs)
    #     return pg.baselines_abs_and_phase(baselines, cluster_id=cluster_id)

    # def plot_mean_and_std(
    #     self,
    #     avg_type="stations",
    #     n_time_avg=40,
    #     max_station=62,
    #     cluster_id=0,
    #     prefix="",
    #     out_dir=None,
    # ):
    #     pg = PlotGains(self.data, self.all_freqs)
    #     pg.mean_and_std(
    #         cluster_id=cluster_id,
    #         avg_type=avg_type,
    #         n_time_avg=n_time_avg,
    #         max_station=max_station,
    #         prefix=prefix,
    #         out_dir=out_dir,
    #     )

    # def plot_that_dd(self):
    #     gains_dict = self.make_gains_dict()
    #     pg = PlotGains(self.data, self.all_freqs)
    #     pg.do_plot(
    #         gains_dict,
    #     )

    def write_to_disk(self, global_sols_file="", outdir=""):

        self.file_handler.make_dir(outdir)
        file_prefix = f"{outdir}/"

        self.eff_outfile = f"{outdir}/eff_nr_{os.path.basename(self.cluster_file)}.npy"
        self.cluster_Nsols_per_Tstep = self.file_handler.write_eff_nr(
            self.cluster_file, self.eff_outfile
        )

        logging.info(f"Effective cluster size: {self.cluster_Nsols_per_Tstep}")
        logging.info(f"Wrote effective cluster size file: {self.eff_outfile}")

        self.file_handler.save_to_numpy_format(
            self.data,
            file_prefix,
            self.all_freqs,
            self.cluster_Nsols_per_Tstep,
            self.nClustEff,
            self.timerange,
            self.sol_timestep,
            self.pointing,
            self.stations,
            self.station_pos,
            self.meantimestep,
        )

        self.gains_outfile = f"{file_prefix}.npy"
        logging.info(f"Wrote gains to: {self.gains_outfile}")

        self.meta_data_outfile = f"{file_prefix}.npz"
        logging.info(f"Wrote gains metadata to: {self.meta_data_outfile}")

        # if os.path.isfile(global_sols_file):
        #     logging.info(f"Reading global solutions: {global_sols_file}")
        #     (
        #         self.n_freqs,
        #         self.p_order,
        #         self.n_stat,
        #         self.n_clus,
        #         self.n_eff_clus,
        #         self.z_sol,
        #     ) = self.file_handler.read_global_solutions(
        #         global_sols_file, len(self.all_freqs)
        #     )

        #     logging.info(f"Converting global solutions")
        #     self.a_poly_sol = self.gains_processor.convert_global_solutions(
        #         self.z_sol,
        #         self.n_eff_clus,
        #         self.n_clus,
        #         self.n_freqs,
        #         self.p_order,
        #         self.n_stat,
        #         eff_nr=self.eff_outfile,
        #     )

        #     self.global_gains_outfile = f"{file_prefix}_global_solutions.npy"
        #     np.save(self.global_gains_outfile, self.a_poly_sol)
        #     logging.info(f"Wrote global solutions to: {self.global_gains_outfile }")

        logging.info("Writing done!")
