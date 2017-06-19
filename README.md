# EAST
EAST: Efficient and Accurate Detection of Topologically Associating Domains from Contact Maps

# Dependencies:

- Python 3 or higher
- numba library

# Input format and Parameters

Input files are matrices in the format of Dixon et al. at http://chromosome.sdsc.edu/mouse/hi-c/download.html. 
- The maxW parameter: determines the maximum size of a TAD allowed in the algorithm (maximum size of a TAD = resoultion*maxW).
- The normalization parameter N: larger values of N  lead to smaller TADs. 
- Alhpa and beta are experimentally determined and set to 0.2.

