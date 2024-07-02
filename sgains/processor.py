import os
import sys
import logging
import numpy as np
from scipy.interpolate import BPoly
from filesIO import FileIOHandler as fh


class makeGainsDict:

    def __init__(self, data, stations, eff_nr):
        self.data = GainsUtils.reshape_gains_data(data)
        self.stations = stations
        self.eff_nr = eff_nr
        self.pols = {"XX": [0, 0], "YY": [1, 1], "XY": [0, 1], "YX": [1, 0]}
        self.pol_stokes = {"I": [0, 0], "V": [1, 1], "U": [0, 1], "Q": [1, 0]}

    def get_gains(self, cluster, station):
        if cluster == 0:
            c_id = slice(None, int(self.eff_nr.cumsum()[cluster]))
        else:
            c_id = slice(
                int(self.eff_nr.cumsum()[cluster - 1]),
                int(self.eff_nr.cumsum()[cluster]),
            )

        s = self.data[:, station, :, c_id]
        s = s.transpose(0, 2, 1, 3, 4)
        s = np.concatenate(s)

        return s

    def get_all_gains(self, clusters):
        """
        read sagecal calibration gains data from the output solutions file

        Parameters
        ----------
        stations : list
            The stations names
        clusters : range object/list
            Clusters ids
        d : array
            gains data
        eff_nr : _type_
            _description_

        Returns
        -------
        Dict
            A nested dictionary with the gains data per station per cluster.
            e.g.
            dict.keys: dict_keys([('CS001HBA0', 0), ('CS001HBA0', 1), ('CS001HBA0', 2), ('CS001HBA0', 3)...
            dict[('CS001HBA0', 0)].keys():  dict_keys(['XX', 'gg_XX', 'YY',
            'gg_YY', 'XY', 'gg_XY', 'YX', 'gg_YX', 'gg_I', 'gg_V', 'gg_U', 'gg_Q'])
        """

        gains = dict()
        for s_idx, s in enumerate(self.stations):
            for c in range(len(clusters)):
                gains[(s, c)] = dict()
                # ref_station = 0
                # ref_g = self.get_gains(d, c, ref_station, self.eff_nr)
                g = self.get_gains(c, s_idx)  # / ref_g
                gg = GainsUtils.g_mul(g, g, np.eye(2))
                I, Q, U, V = GainsUtils.cov2stokes(gg)
                gg_stokes = np.stack(
                    [np.stack([I, Q], axis=-1), np.stack([U, V], axis=-1)], axis=-1
                )

                for pol_name, (i, j) in self.pols.items():
                    gains[(s, c)][pol_name] = g[:, :, i, j]
                    gains[(s, c)]["gg_" + pol_name] = gg[:, :, i, j]
                for pol_name, (i, j) in self.pol_stokes.items():
                    gains[(s, c)]["gg_" + pol_name] = gg_stokes[:, :, i, j]

        return gains


class GainsUtils:

    @staticmethod
    def reshape_gains_data(data):
        data = data[..., 0] + 1j * data[..., 1]
        data = data.transpose(0, 1, 2, 4, 3)
        return data.reshape(*(list(data.shape[:-1]) + [2, 2]))

    @staticmethod
    def cov2stokes(R):
        stokesI = 0.5 * (R[:, :, 0, 0] + R[:, :, 1, 1])
        stokesV = 0.5 * (-1j * (R[:, :, 0, 1] - R[:, :, 1, 0]))
        stokesQ = 0.5 * (R[:, :, 0, 0] - R[:, :, 1, 1])
        stokesU = 0.5 * (R[:, :, 0, 1] + R[:, :, 1, 0])

        return stokesI, stokesQ, stokesU, stokesV

    @staticmethod
    def g_mul(g1, g2, c):
        return np.matmul(np.matmul(g1, c), g2.conj().transpose(0, 1, 3, 2))

    @staticmethod
    def get_ps(d, dx, w="blackmanharris"):
        from scipy.signal import get_window

        def is_odd(num):
            return num & 0x1

        w = get_window(w, d.shape[2])[None, None, :]

        ps = abs(np.fft.fftshift(np.fft.fft(d * w, axis=2), axes=2)) ** 2
        M = ps.shape[2]
        if is_odd(M):
            ps = 0.5 * (ps[:, :, M // 2 + 1 :] + ps[:, :, : M // 2][:, :, ::-1])
        else:
            ps = 0.5 * (ps[:, :, M // 2 + 1 :] + ps[:, :, 1 : M // 2][:, :, ::-1])

        delay = (-np.fft.fftfreq(d.shape[2], dx)[d.shape[2] // 2 + 1 :])[::-1]

        return delay, ps

    @staticmethod
    def convert_solutions(solutions_files_list, cluster_start_end_indices):
        all_data = []
        freqs = []

        for fname in solutions_files_list:
            try:
                datas, freq, nClustEff, bw, timestep, nClust = (
                    fh.read_sagecal_solutions_file(fname)
                )
                logging.info(f"Successfully read file: {fname}")
            except:
                logging.warning(f"Failed to read file: {fname}")
                continue

            freqs.append(freq)
            all_data.append(datas)

        logging.info(f"Single channel bandwidth: {bw}")
        logging.info(f"Single solution timestep: {timestep}")
        logging.info(f"Number of sky directions solved for: {int(nClust)}")
        logging.info(f"Number of effective sky directions solved for: {int(nClustEff)}")
        logging.info(f"NUmber of frequency steps: {len(freqs)}")

        # Sort data in ascending frequency order
        all_freqs_sorted, all_data_sorted = zip(*sorted(zip(freqs, all_data)))

        # The calibration data are in shape (nfreq, ncluster, ntime, nstations, npol), after transpose (ntime, nstations, nfreq, npol, ncluster)
        if cluster_start_end_indices:
            cdata = (
                np.array(all_data_sorted)[
                    :, cluster_start_end_indices[0] : cluster_start_end_indices[1]
                ]
            ).transpose((2, 3, 0, 4, 1))
        else:
            cdata = np.array(all_data_sorted).transpose((2, 3, 0, 4, 1))

        # For some reason the real part and imaginary part are saved to last axis with dimension 2
        data = np.zeros(cdata.shape + (2,), dtype=np.float64)
        data[:, :, :, :, :, 0] = np.real(cdata)
        data[:, :, :, :, :, 1] = np.imag(cdata)

        # data = data[..., 0] + 1j * data[..., 1]

        logging.info(f"Total time steps: {data.shape[0]}")
        logging.info(f"Number of stations: {data.shape[1]}")
        logging.info(f"Number of polarizations: {data.shape[3]}")
        logging.info(
            f"Complex gains solutions dimensions [ntimesteps, nstations, nfreqsteps, npols, ndirections, real+imag]:  {data.shape}"
        )

        return data, all_freqs_sorted, nClustEff, bw, timestep, nClust

    def convert_global_solutions(
        self, z_sol, n_eff_clus, n_clus, n_freqs, p_order, n_stat, eff_nr=None
    ):
        assert z_sol.shape[1] - 1 == n_eff_clus

        if n_eff_clus != n_clus:
            if not eff_nr:
                logging.error("n_clus != n_eff_clus; eff_nr required !")
                sys.exit(0)
            eff_nr = np.load(eff_nr)

            assert len(eff_nr) == n_clus
            assert eff_nr.sum() == n_eff_clus

            eff_nr_idx = []
            i = 0
            for n_eff in eff_nr:
                eff_nr_idx.extend(np.arange(i + n_eff - 1, i - 1, -1).astype(int))
                i += n_eff
            eff_nr_idx = np.array(eff_nr_idx)

        x = np.linspace(0, 1, n_freqs)

        a_poly_sol = []

        for i_c in np.arange(n_eff_clus):
            z_sol_mat = z_sol[:, i_c + 1].reshape((-1, p_order, n_stat, 4, 2))
            z_sol_mat = z_sol_mat.transpose((1, 0, 2, 3, 4))

            poly_sol = BPoly(z_sol_mat[:, None], [0, 1])(x)

            poly_sol = poly_sol.transpose((1, 0, 2, 3, 4))
            poly_sol = poly_sol.transpose((0, 2, 1, 3, 4))

            a_poly_sol.append(poly_sol)

        a_poly_sol = np.stack(a_poly_sol, -1)
        a_poly_sol = a_poly_sol.transpose((0, 1, 2, 3, 5, 4))

        if n_eff_clus != n_clus:
            a_poly_sol = a_poly_sol[:, :, :, :, eff_nr_idx, :]

        logging.info(f"Dimensions of the global solutions data:  {a_poly_sol.shape}")
        return a_poly_sol


class GainsStats:

    def get_all_stations_amplitude_stats(gains_data, clusters, stations, pol="XX"):
        rms_dt = []
        med = []
        rms_dnu = []

        c_ii = range(len(clusters))

        for s_id, stn in enumerate(stations):
            rms_dt.append(
                [np.std(np.diff(gains_data[(stn, k)][pol], axis=0)) for k in c_ii]
            )
            med.append([np.median(abs(gains_data[(stn, k)][pol])) for k in c_ii])
            rms_dnu.append(
                [np.std(np.diff(gains_data[(stn, k)][pol], axis=1)) for k in c_ii]
            )

        rms_dt = np.array(rms_dt)
        med = np.array(med)
        rms_dnu = np.array(rms_dnu)

        return med, rms_dt, rms_dnu

    def get_all_clusters_amplitude_stats(gains_data, clusters, stations, pol="XX"):
        rms_dt = []
        med = []
        rms_dnu = []

        for c in np.arange(0, len(clusters)):
            rms_dt.append(
                [np.std(np.diff(gains_data[(k, c)][pol], axis=0)) for k in stations]
            )
            med.append([np.median(abs(gains_data[(k, c)][pol])) for k in stations])
            rms_dnu.append(
                [np.std(np.diff(gains_data[(k, c)][pol], axis=1)) for k in stations]
            )

        rms_dt = np.array(rms_dt)
        med = np.array(med)
        rms_dnu = np.array(rms_dnu)

        return med, rms_dt, rms_dnu
