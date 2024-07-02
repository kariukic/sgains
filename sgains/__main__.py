"""
Module to read, transform, write and plot sagecal solutions
"""

import os
import numpy as np
from solutions import Solutions

import logging
import logging.config

logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.config"))
logging = logging.getLogger("sagecal_solutions")

from argparse import ArgumentParser

parser = ArgumentParser("Sagecal Gains Explorer")

parser.add_argument(
    "-o",
    "--obsid",
    help="Observation ID, for example L254871",
    dest="obsid",
    required=False,
)

# parser.add_argument(
#     "-s",
#     "--cal_stage",
#     help="Calibration stage",
#     dest="cal_stage",
#     required=True,
#     choices=["DI", "BP", "DD"],
# )

parser.add_argument(
    "-s",
    "--solution_files",
    help="txt file listing the .MS.solutions files",
    dest="solution_files",
    required=True,
)

parser.add_argument(
    "-m",
    "--sample_ms",
    nargs="+",
    help="The full path to the measurement set files that contain the metadata information (only one is used), e.g. /path/to/file/*.MS",
    dest="sample_ms",
    required=True,
)

parser.add_argument(
    "-l",
    "--sky_model",
    help="The sky model file used during calibration.",
    dest="sky_model",
    required=False,
)

parser.add_argument(
    "-c",
    "--cluster_file",
    help="Clusters file used during calibration",
    dest="cluster_file",
    required=True,
)

parser.add_argument("-p", "--plot", dest="plot", default=True, action="store_true")

# parser.add_argument(
#     "-r", "--write_to_h5", dest="write_to_h5", default=True, action="store_true"
# )

# parser.add_argument(
#     "-g",
#     "--global_sols_file",
#     help="The sagecal_Zsol file incase of sagecal-MPI",
#     dest="global_sols_file",
#     required=False,
# )

parser.add_argument(
    "-x",
    "--cluster",
    nargs=2,
    help="The indices of the directions (clusters) to read from sagecal solution file (start_index, end_index). Default: [], e.g. to specify the 2 DI clusters use [0 1], which is the first direction/cluster for direction-independent calibration.",
    dest="cluster",
    default=[],
    type=int,
)

parser.add_argument(
    "-i",
    "--indir",
    help="The directory of the sagecal solutions files. Use full path including `/net/node..' if all solutions files are in one directory nstead of being distributed across multiple nodes.",
    dest="indir",
    default="/data/users/lofareor/sarod/pipeline/",
)

parser.add_argument(
    "--pid",
    type=int,
    help="Process step ID. Default: 2, which is after direction-independent calibration. For unprocessed data use 0.)",
    dest="pid",
    default=2,
)

parser.add_argument(
    "-d",
    "--outdir",
    help="The directory to save all output files",
    dest="outdir",
    default=".",
)

parser.add_argument(
    "-n",
    "--nodes",
    nargs="+",
    help="The list of nodes where sagecal solutions files are saved. On EOR-cluster: specify consecutive multiple nodes as e.g. 116..131 (will use 'node116' to \node'131)",
    dest="nodes",
    default=[os.getenv("HOSTNAME")],
    required=False,
)

parser.add_argument(
    "--station_pairs",
    type=list,
    default=[(0, 1), (5, 8), (15, 18), (36, 39), (52, 55), (55, 59)],
    help="di stations to plot gains of",
)

parser.add_argument(
    "--clusters_to_plot",
    type=str,
    default="1,4,24,5,13,106,10,51,96,40,84,91,30,105",
    help="clusters indices to plot gains of",
)

parser.add_argument(
    "--cluster_names",
    type=str,
    default="'~NCP','3C61.1','3deg','4deg','5deg','6deg','7deg','8deg','9deg','10deg','11deg','14deg','Cas-A','Cyg-A'",
    help="clusters names to plot",
    required=False,
)

parser.add_argument(
    "--stations_to_plot",
    type=str,
    default="0,5,10,20,40,50,55",
    help="station indices to plot gains of",
    required=False,
)

parser.add_argument(
    "-v",
    "--verbose",
    help="verbosity. Can be v, vv or vvv",
    dest="verbose",
    default="v",
)


if __name__ == "__main__":
    args = parser.parse_args()

    sols = Solutions(
        cluster_file=args.cluster_file,
        sample_ms=args.sample_ms,
        solsfiles=args.solution_files,
        nodes=args.nodes,
        obsid=args.obsid,
        indir=args.indir,
        pid=args.pid,
        cluster_to_read=args.cluster,
    )

    if args.plot:
        if args.station_pairs:
            fig1, fig2 = sols.plot.baselines_abs_and_phase(
                args.station_pairs, cluster_id=0
            )
            fig1.savefig(
                f"{args.outdir}/baselines_gains_amplitude.png", bbox_inches="tight"
            )
            fig2.savefig(
                f"{args.outdir}/baselines_gains_phase.png", bbox_inches="tight"
            )

        fig3 = sols.plot.mean_and_std(cluster_id=0, avg_type="stations", n_time_avg=1)
        fig3.savefig(
            f"{args.outdir}/gains_mean_and_std_per_frequency.png",
            bbox_inches="tight",
        )
        fig4 = sols.plot.mean_and_std(cluster_id=0, n_time_avg=40, avg_type="frequency")
        fig4.savefig(
            f"{args.outdir}/gains_mean_and_std_per_station.png",
            bbox_inches="tight",
        )

        solsdict = sols.make_gains_dict()

        fig5 = sols.plot.dynamic_spectrum(
            gain_data=solsdict,
            dd_stations=args.stations_to_plot,
            dd_clusters=args.clusters_to_plot,
            clusters_names=args.cluster_names,
            vmin=0.1,
            vmax=10,
        )
        fig5.savefig(
            f"{args.outdir}/gains_dynamic_spectra_amplitude.png",
            bbox_inches="tight",
        )

        fig6 = sols.plot.dynamic_spectrum(
            gain_data=solsdict,
            dd_stations=args.stations_to_plot,
            dd_clusters=args.clusters_to_plot,
            clusters_names=args.cluster_names,
            action_fct=np.angle,
            log_norm=False,
            vmin=-np.pi,
            vmax=np.pi,
        )
        fig6.savefig(
            f"{args.outdir}/gains_dynamic_spectra_phase.png",
            bbox_inches="tight",
        )

        fig7, rms_dt, med, rms_dnu = sols.plot.cluster_stats(solsdict, pol="XX")
        fig7.savefig(
            f"{args.outdir}/gains_clusters_stats.png",
            bbox_inches="tight",
        )

        fig8, rms_dt, med, rms_dnu = sols.plot.stations_stats(solsdict, pol="XX")
        fig8.savefig(
            f"{args.outdir}/gains_stations_stats.png",
            bbox_inches="tight",
        )

        if args.sky_model:
            st = sols.stations[0]
            cluster_info = sols.get_clusters_details(args.sky_model)
            fig9 = sols.plot.cluster_gains_noise(
                solsdict, clusters_info=cluster_info, station=st, pol="XX"
            )
            fig9.savefig(
                f"{args.outdir}/{st}_clusters_snr.png",
                bbox_inches="tight",
            )
