# EAST
EAST: Efficient and Accurate Detection of Topologically Associating Domains from Contact Maps

# Current Release: EAST 2.0
- EAST 2.0 is faster and more memory efficient.
- EAST 2.0 can work with both Dixon and Rao data formats.
- In case you don't have enough memory to store a contact map in a dense matrix, comment the line 'chr1 = chr1.todense()'.
- GUI will be added in the next release (2.1).
# Dependencies:

- Python 3 or higher
- numba library

# Input format 

Input files are matrices of format:

 1- Dixon et al. at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35156 or  
 2- Rao et al. at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525

# Parameters

- maxL: max length of a TAD allowed
- Nfactor: normalization factor

