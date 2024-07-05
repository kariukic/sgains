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
            self.freq_mhz,
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

        logging.info(f"Reading cluster IDs")
        self.cluster_ids = self.file_handler.get_cluster_ids(self.cluster_file)

        logging.info(f"Making gains_dict object")
        mgd = makeGainsDict(self.data, self.stations, self.eff_nr)
        self.gains_dict = mgd.get_all_gains(clusters=self.cluster_ids)

        logging.info(f"Averaging out effective cluster sizes")
        self.gains = self.get_equal_ntstep_gains()

        logging.info(f"Making the plotting object")
        self.plot = PlotGains(
            self.gains,
            self.freq_mhz,
            self.time_s,
            self.stations,
            self.cluster_ids,
            self.eff_nr,
            self.bandwidth,
        )

        logging.info("Reading done!")

    def get_equal_ntstep_gains(self):
        ggs = np.zeros(
            (
                len(self.time_s),
                len(self.stations),
                len(self.freq_mhz),
                4,
                len(self.cluster_ids),
            ),
            dtype=np.complex64,
        )

        for p, pol in enumerate(["XX", "XY", "YX", "YY"]):
            for s, station in enumerate(self.stations):
                sgs = np.zeros(
                    (len(self.time_s), len(self.freq_mhz), len(self.cluster_ids)),
                    dtype=np.complex128,
                )
                for clst, eff in zip(self.cluster_ids.values(), self.eff_nr):
                    g = self.gains_dict[(station, clst)][pol].reshape(
                        eff, -1, len(self.freq_mhz)
                    )
                    gs = np.average(g, axis=0)
                    sgs[..., clst] = gs

                ggs[:, s, :, p, :] = sgs

        return ggs

    def get_clusters_details(self, sky_model):
        clusters_info, _ = self.file_handler.getClusters(self.cluster_file, sky_model)
        return clusters_info

    # def make_gains_dict(self):
    #     mgd = makeGainsDict(self.data, self.stations, self.eff_nr)
    #     return mgd.get_all_gains(clusters=self.cluster_ids)

    def get_eff_nr(self):
        nrs = []
        for data in open(self.cluster_file):
            if data.strip() and not data.strip()[0] == "#":
                nr = int(data.strip().split()[1])
                nrs.append(nr)
        return np.asarray(nrs)

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
            self.freq_mhz,
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
        #         global_sols_file, len(self.freq_mhz)
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
