# import h5py
import time
import glob
import os
import sys
import numpy as np
import logging
from pathlib import Path
from casacore import tables as tab


class CollectGainsFiles:
    def __init__(
        self, obsid: str = None, nodes: str = None, indir: str = None, pid: str = None
    ) -> None:
        self.obsid = obsid
        self.nodes = nodes
        self.indir = indir
        self.pid = pid

    @staticmethod
    def parse_nodes(nodelist):
        nodes = [
            "node%03d" % i
            for j in nodelist
            if ".." in j
            for i in range(int(j.split("..")[0]), int(j.split("..")[1]) + 1)
        ]

        nodes += [j for j in nodelist if not ".." in j]

        return nodes

    def collect_solutions_files_from_single_node(self, node):

        fname_glob_pattern = "%s/%s*%03d*.MS.solutions" % (
            self.indir,
            self.obsid,
            self.pid,
        )

        # If process ID is zero, skip process ID in path to file
        if self.pid == 0:
            fname_glob_pattern = "%s/%s*MS.solutions" % (self.indir, self.obsid)
        if self.pid < 0:
            fname_glob_pattern = "%s/%s*%02d*.MS.solutions" % (
                self.indir,
                self.obsid,
                -1 * self.pid,
            )

        if "/net/" not in self.indir:
            fname_glob_pattern = f"/net/{node}/" + fname_glob_pattern

        # Get a list of files from each node
        solutions_files_list = glob.glob(fname_glob_pattern)
        solutions_files_list = [s for s in solutions_files_list if "filtered" not in s]

        # assert (len(solutions_files_list) > 0), f"No solutions files found on node {node} at {self.indir}. \nLooked for {fname_glob_pattern}"
        if not solutions_files_list:
            sys.exit(
                f"Fatal Error: No solutions files found on node {node} at {self.indir}. Looked for {fname_glob_pattern}"
            )

        return solutions_files_list

    def collect_solutions_files(self) -> list:

        nodes = self.parse_nodes(nodes)
        logging.debug(f"Nodes: {nodes}")

        logging.debug(f"Collecting solutions files from each node at: {self.indir}")
        solutions_files = []
        for inode in self.nodes:
            solutions_files += self.collect_solutions_files_from_single_node(inode)

        logging.info(f"Total solutions files found: {len(solutions_files)}")

        return solutions_files


class FileIOHandler:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_cluster_ids(cluster_file):
        clusters = []
        with open(cluster_file) as f:
            for line in f.readlines():
                if line.strip().startswith("#"):
                    continue
                # print(line)
                s = line.split(" ")
                if s == ["\n"]:
                    continue
                clusters.append({"ID": int(s[0]), "NS": int(s[1]), "S": s[2:]})

        return dict(zip([v["ID"] for v in clusters], np.arange(len(clusters))))

    @staticmethod
    def write_eff_nr(cluster_file, file_name):
        nrs = []
        for data in open(cluster_file):
            if data.strip() and not data.strip()[0] == "#":
                nr = int(data.strip().split()[1])
                nrs.append(nr)
        np.save(file_name, nrs)
        return nrs

    def read_npy_file(npy_file):
        return np.load(npy_file)

    @staticmethod
    def read_sagecal_solutions_file(fname):
        myf = open(fname)

        # The first lines contain some metadata such as frequencies, bandwidth, time resolution, number of stations, number of clusters, number
        for i in range(3):
            a = myf.readline()
        freq, bw, timestep, nStations, nClust, nClustEff = tuple(
            [float(i) for i in a.split()]
        )

        myf.close()
        data = np.loadtxt(
            fname,
            skiprows=3,
            usecols=tuple(range(1, int(nClustEff) + 1)),
            unpack=True,
        )

        datas = []

        # If we only calibrated into 1 direction / 1 cluster
        if nClustEff == 1.0:
            a = data.reshape((-1, int(nStations), 4, 2))
            cdata = a[:, :, :, 0] + 1.0j * a[:, :, :, 1]
            datas.append(cdata)

        # Else, loop over the number of diretions / clusters
        else:
            for i in range(int(nClustEff)):
                a = data[i].reshape((-1, int(nStations), 4, 2))
                cdata = a[:, :, :, 0] + 1.0j * a[:, :, :, 1]
                datas.append(cdata)

        return datas, freq, nClustEff, bw, timestep, nClust

    @staticmethod
    def save_to_numpy_format(
        data,
        outdir,
        freqs,
        cluster_Nsols_per_Tstep,
        nClustEff,
        timerange,
        timestep,
        pointing,
        stations,
        station_pos,
        meantimestep,
    ):

        # if the number of solutions per time interval per cluster is not different for each cluster
        if np.sum(cluster_Nsols_per_Tstep) == nClustEff:
            np.savez(
                outdir,
                freqs=np.array(freqs) * 1.0e6,
                timerange=timerange,
                timestep=timestep,
                meantimestep=meantimestep,
                stations=stations,
                stat_pos=station_pos,
                pointing=pointing,
                eff_nr=cluster_Nsols_per_Tstep,
            )
        else:
            np.savez(
                outdir,
                freqs=np.array(freqs) * 1.0e6,
                timerange=timerange,
                timestep=timestep,
                meantimestep=meantimestep,
                stations=stations,
                stat_pos=station_pos,
                pointing=pointing,
            )

        np.save(outdir, data)

        return

    @staticmethod
    def getMSinfo(MS=None):
        if MS is None:
            print("No measurement set given")
            return

        if isinstance(MS, list):
            MS = MS[0]
            logging.debug(f"Got a list of MS files: getting info only from {MS}")

        if os.path.isdir(MS):
            myMS = tab.table(MS)
        else:
            logging.info("Do not understand the format of MS", MS, "bailing out")
            return
        timerange = [
            np.amin(myMS.getcol("TIME_CENTROID")),
            np.amax(myMS.getcol("TIME_CENTROID")),
        ]
        timestep = myMS.getcell("INTERVAL", 0)

        pointing = tab.table(myMS.getkeyword("FIELD")).getcell("PHASE_DIR", 0)
        stations = tab.table(myMS.getkeyword("ANTENNA")).getcol("NAME")
        station_pos = tab.table(myMS.getkeyword("ANTENNA")).getcol("POSITION")

        logging.info(f"MS Timerange (s): {timerange}")
        logging.info(f"MS timestep (s): {timestep}")
        logging.info(f"MS pointing (ra, dec): {pointing}")
        logging.info(f"MS stations total: {len(stations)}, {stations}")
        # logging.info(f"MS stations positions (X, Y, Z): {station_pos}")

        return (timerange, timestep, pointing.flatten(), stations, station_pos)

    @staticmethod
    def read_global_solutions(solutions_file, n_freqs):
        # reading only the 2nd row which has the headers of the data arrangement
        with open(solutions_file) as f:
            for i, line in enumerate(f.readlines()):
                if i == 2:
                    _, p_order, n_stat, n_clus, n_eff_clus = line.strip().split()
                    p_order = int(p_order)
                    n_stat = int(n_stat)
                    n_clus = int(n_clus)
                    n_eff_clus = int(n_eff_clus)
                    break
        # loading the rest of the data from line 3 to the end into an array
        z_sol = np.loadtxt(solutions_file, skiprows=3)

        logging.info(f"Number of frequency channels: {n_freqs}")
        logging.info(f"Bpol order: {p_order}")
        logging.info(f"Number of stations: {n_stat}")
        logging.info(f"Number of sky clusters: {n_clus}")
        logging.info(f"Number of effective clusters: {n_eff_clus}")

        return n_freqs, p_order, n_stat, n_clus, n_eff_clus, z_sol

    @staticmethod
    def read_sagecal_solutions_from_numpy_file(npy_solutions_file):
        data = np.load(npy_solutions_file)
        data = data[:, :, :, :, :, 0] + 1j * data[:, :, :, :, :, 1]
        data = data.transpose(0, 1, 2, 4, 3)
        data = data.reshape(*(list(data.shape[:-1]) + [2, 2]))

        return data

    @staticmethod
    def getClusters(clusterf, skymodel, max_nr_clusters=1000):
        # get clusters
        sky = open(skymodel)
        sources = {}
        # first get indices
        clusters = []
        count = 0
        tot_nr_sol = 0
        nrSB = 0
        for line in sky:
            if not line.strip():
                continue
            if not "name" in line.lower():
                continue
            splitted = line.lower().strip().split()[1:]

            if not splitted[:11] == [
                "name",
                "h",
                "m",
                "s",
                "d",
                "m",
                "s",
                "i",
                "q",
                "u",
                "v",
            ]:
                print("Do not understand the format", splitted[:11], "bailing out")
                return clusters, -1
            nr_sp = 1
            if splitted[12] == "spectral_index1" or splitted[12] == "si1":
                nr_sp += 1
                if splitted[13] == "spectral_index2" or splitted[13] == "si2":
                    nr_sp += 1
            break
        for line in sky:
            if not line.strip() or line.strip()[0] == "#":
                continue
            splitted = line.split()
            sources[splitted[0].strip()] = {}
            sources[splitted[0].strip()]["Ra"] = (
                np.pi
                / 12.0
                * (
                    float(splitted[1])
                    + float(splitted[2]) / 60.0
                    + float(splitted[3]) / 3600.0
                )
            )
            sources[splitted[0].strip()]["Dec"] = (
                np.pi
                / 180.0
                * (
                    float(splitted[4])
                    + float(splitted[5]) / 60.0
                    + float(splitted[6]) / 3600.0
                )
            )
            sources[splitted[0].strip()]["I"] = float(splitted[7])
            sources[splitted[0].strip()]["sp"] = [splitted[11]] + [
                str(float(splitted[i + 11]) * (np.log(10)) ** i)
                for i in range(1, nr_sp)
            ]
            sources[splitted[0].strip()]["freq0"] = float(splitted[-1])

        clusterfile = open(clusterf)

        for i, line in enumerate(clusterfile):
            if not line.strip() or line.strip()[0] == "#":
                continue
            splitted = line.split()
            if count >= max_nr_clusters:
                tot_nr_sol += int(splitted[1])
                continue
            #         print "adding cluster", splitted[0]
            clusters.append({})
            clusters[count]["cl_id"] = int(splitted[0])
            clusters[count]["id"] = count
            clusters[count]["nrsol"] = int(splitted[1])
            clusters[count]["real"] = []
            clusters[count]["imag"] = []
            clusters[count]["sources"] = {}
            clusters[count]["store_data"] = True
            avg_ra = 0
            avg_dec = 0
            sum_weight = 0
            for src in splitted[2:]:
                clusters[count]["sources"][src.strip()] = sources[src.strip()]
                weight = sources[src.strip()]["I"]
                avg_ra += np.exp(1j * sources[src.strip()]["Ra"]) * weight
                avg_dec += np.exp(1j * sources[src.strip()]["Dec"]) * weight
                sum_weight += weight
            if sum_weight > 0:
                clusters[count]["Ra"] = np.angle(avg_ra / sum_weight)
                clusters[count]["Dec"] = np.angle(avg_dec / sum_weight)
            else:
                clusters[count]["Ra"] = np.angle(avg_ra)
                clusters[count]["Dec"] = np.angle(avg_dec)
            maxdist = 0
            x1 = (0.5 * np.pi - clusters[count]["Dec"]) * np.sin(clusters[count]["Ra"])
            y1 = (0.5 * np.pi - clusters[count]["Dec"]) * np.cos(clusters[count]["Ra"])
            for src in clusters[count]["sources"].keys():

                mydiff = np.sqrt(
                    (
                        x1
                        - (0.5 * np.pi - clusters[count]["sources"][src]["Dec"])
                        * np.sin(clusters[count]["sources"][src]["Ra"])
                    )
                    ** 2
                    + (
                        y1
                        - (0.5 * np.pi - clusters[count]["sources"][src]["Dec"])
                        * np.cos(clusters[count]["sources"][src]["Ra"])
                    )
                    ** 2
                )
                if mydiff > maxdist:
                    maxdist = mydiff
            clusters[count]["size"] = maxdist
            tot_nr_sol += clusters[count]["nrsol"]
            count += 1

        return clusters, tot_nr_sol

    @staticmethod
    def make_dir(dir_path) -> None:
        if not Path(dir_path).exists():
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                logging.debug(f"Made output directory: {dir_path}")
            except OSError as e:
                logging.error(
                    f"Failed to create output directory: {dir_path}. Error: {e}"
                )

    @staticmethod
    def read_lines_from_file(file_path):
        """
        Reads all lines from the specified file and returns them as a list.

        Parameters:
        file_path (str): The path to the input file.

        Returns:
        list: A list of strings, each representing a line from the file.
        """
        with open(file_path, "r") as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
