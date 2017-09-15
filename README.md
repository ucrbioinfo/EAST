# EAST
EAST: Efficient and Accurate Detection of Topologically Associating Domains from Contact Maps

# Current Release: EAST 2.0
- EAST 2.0 is faster and more memory efficient. As an example, for the WHOLE GENOME of Rao et al. with resolution 5kb, it takes 2m23s to read data from files, 1m34s to compute the SAT matrices (integral images) and only 2m15s to compute TAD boundaries on Intel(R) CORE(TM) i7-7500U CPU @ 2.70GHz. For Dixon et al. data with resoultion 40kb, it takes 1 minute to read the data from files, 1.25s to compute the SAT matrices and 2.34s to compute the TAD boundaries.
- EAST 2.0 can work with both Dixon and Rao data formats.
- In case you don't have enough memory to store a contact map in a dense matrix, comment the line 'chr1 = chr1.todense()'.
- GUI will be added in the next release (2.1).
# Dependencies:

- Python 3 or higher
- numba package
- pandas package

# Input format 

Input files are matrices of format:

 1- Dixon et al. at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35156 or  
 2- Rao et al. at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525

# Parameters

- maxL: max length of a TAD allowed
- Nfactor: normalization factor

