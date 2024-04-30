#! /usr/bin/env python

from re import U
import numpy as np
import argparse
import time
from multiprocessing import Pool, TimeoutError
from julia_curve import c_from_group

# Update according to your group size and number (see TUWEL)
GROUP_SIZE   = 2
GROUP_NUMBER = 4

# do not modify BENCHMARK_C
BENCHMARK_C = complex(-0.2, -0.65)

# section 2.1
def compute_patch(args):

    x, y, patch, meta_information = args
    xmin, xmax, ymin, ymax, im_width, im_height, c = meta_information
    
    zabs_max = 10
    nit_max = 300
    xwidth  = xmax - xmin
    yheight = ymax - ymin

    julia_patch = np.zeros((patch, patch))
    for ix in range(patch):
        for iy in range(patch):
            px = x + ix
            py = y + iy
            nit = 0
            z = complex(px / im_width * xwidth + xmin,
                        py / im_height * yheight + ymin)
            
            while abs(z) <= zabs_max and nit < nit_max:
                z = z**2 + c
                nit += 1
            ratio = nit / nit_max
            julia_patch[ix,iy] = ratio

    return (x, y, julia_patch)

def compute_julia_in_parallel(size, xmin, xmax, ymin, ymax, patch, nprocs, c):

    task_list = []
    for x in range (0, size, patch):
        for y in range (0, size, patch):
            task_list.append((x, y, patch, (xmin, xmax, ymin, ymax, size, size, c)))

    pool = Pool(processes = nprocs)
    completed_patches = pool.map(compute_patch, task_list, chunksize=1)
    pool.close()
    pool.join()
    
    julia_img = np.zeros((size, size))
    
    for p in completed_patches:
        x, y, julia_patch = p
        julia_img[x:min(x+patch, size)  , y:min(y+patch,size)] = julia_patch[:min(size - x, patch), :min(size - y, patch)]

    return julia_img

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="image size in pixels (square images)", type=int, default=500)
    parser.add_argument("--xmin", help="", type=float, default=-1.5)
    parser.add_argument("--xmax", help="", type=float, default=1.5)
    parser.add_argument("--ymin", help="", type=float, default=-1.5)
    parser.add_argument("--ymax", help="", type=float, default=1.5)
    parser.add_argument("--group-size", help="", type=int, default=None)
    parser.add_argument("--group-number", help="", type=int, default=None)
    parser.add_argument("--patch", help="patch size in pixels (square images)", type=int, default=20)
    parser.add_argument("--nprocs", help="number of workers", type=int, default=1)
    parser.add_argument("--draw-axes", help="Whether to draw axes", action="store_true")
    parser.add_argument("-o", help="output file")
    parser.add_argument("--benchmark", help="Whether to execute the script with the benchmark Julia set", action="store_true")
    args = parser.parse_args()

    #print(args)
    if args.group_size is not None:
        GROUP_SIZE = args.group_size
    if args.group_number is not None:
        GROUP_NUMBER = args.group_number

    # assign c based on mode
    c = None
    if args.benchmark:
        c = BENCHMARK_C 
    else:
        c = c_from_group(GROUP_SIZE, GROUP_NUMBER) 

    stime = time.perf_counter()
    julia_img = compute_julia_in_parallel(
        args.size,
        args.xmin, args.xmax, 
        args.ymin, args.ymax, 
        args.patch,
        args.nprocs,
        c)
    rtime = time.perf_counter() - stime

    print(f"{args.size};{args.patch};{args.nprocs};{rtime}")

    if not args.o is None:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        fig, ax = plt.subplots()
        ax.imshow(julia_img, interpolation='nearest', cmap=plt.get_cmap("hot"))

        if args.draw_axes:
            # set labels correctly
            im_width = args.size
            im_height = args.size
            xmin = args.xmin
            xmax = args.xmax
            xwidth = args.xmax - args.xmin
            ymin = args.ymin
            ymax = args.ymax
            yheight = args.ymax - args.ymin

            xtick_labels = np.linspace(xmin, xmax, 7)
            ax.set_xticks([(x-xmin) / xwidth * im_width for x in xtick_labels])
            ax.set_xticklabels(['{:.1f}'.format(xtick) for xtick in xtick_labels])
            ytick_labels = np.linspace(ymin, ymax, 7)
            ax.set_yticks([(y-ymin) / yheight * im_height for y in ytick_labels])
            ax.set_yticklabels(['{:.1f}'.format(-ytick) for ytick in ytick_labels])
            ax.set_xlabel("Imag")
            ax.set_ylabel("Real")
        else:
            # disable axes
            ax.axis("off") 

        plt.tight_layout()
        plt.savefig(args.o, bbox_inches='tight')
        #plt.show()