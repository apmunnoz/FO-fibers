from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_root = rank == 0


def parprint(*args):
    if is_root:
        print("[FD]", *args, flush=True)

