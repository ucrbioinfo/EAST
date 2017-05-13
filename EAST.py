import numpy as np
from timeit import time
from numba import guvectorize,float64,int64
import os

maxW = 80
maxL = 2*maxW + 1
minL = 3
sampleSize = 2*maxL

SPECIES = 'hES'
if SPECIES=='hIMR90' or SPECIES=='hES':
    NUM_OF_CHRMS = 23
elif SPECIES == 'mES' or 'mCO':
    NUM_OF_CHRMS = 20
for CHRM in range(22,NUM_OF_CHRMS+1):
    print('Loading Chromosome '+ str(CHRM))
    st = time.time()
    chr1 = np.loadtxt(os.path.abspath(os.sep)+'Users/Abbas/Google Drive/Research/Dataset/'+SPECIES+'/nij/nij.chr'+str(CHRM))
    print('time to read chromosome ',CHRM,':',time.time()-st)
    CUT = chr1.shape[0]
    #print('********************** TAD DETECTION *****************************')
    # TAD Detection
    @guvectorize([(float64[:,:], int64[:], int64[:], float64[:])], '(m,p),(),()->()',target='parallel')
    def score(intgMAT,i,l,res):
        i_indent = 2*maxW+1
        j_indent = 2*maxW+1
        wScore = 0

        c1 = 1.0 
        c2 = 0.2
        c3 = 0.2  #penalty for inter-TAD interactions
        if l[0] % 2 == 0:
            w = (l[0])/2
            w2 = np.math.ceil(w/5)

            A = [int(i[0]-2*w+i_indent),int(i[0]-w+j_indent)]
            B = [int(i[0]-2*w+i_indent),int(i[0]+w+j_indent)]
            C = [int(i[0]-w+i_indent),int(i[0]-w+j_indent)]
            D = [int(i[0]-w+i_indent),int(i[0]+w+j_indent)]
            E = [int(i[0]+w+i_indent),int(i[0]-w+j_indent)]
            F = [int(i[0]+w+i_indent),int(i[0]+w+j_indent)]
            G = [int(i[0]+2*w+i_indent),int(i[0]-w+j_indent)]
            H = [int(i[0]+2*w+i_indent),int(i[0]+w+j_indent)]
                       
            O = [int(i[0]-w-1+i_indent),int(i[0]+w-w2+j_indent)]
            P = [int(i[0]-w+w2-1+i_indent),int(i[0]+w-w2+j_indent)]
            Q = [int(i[0]-w+w2-1+i_indent),int(i[0]+w+j_indent)]

            ABGH = (intgMAT[A[0], A[1]] + intgMAT[H[0], H[1]]) - (intgMAT[B[0], B[1]] + intgMAT[G[0], G[1]])
            CDEF = (intgMAT[C[0], C[1]] + intgMAT[F[0], F[1]]) - (intgMAT[D[0], D[1]] + intgMAT[E[0], E[1]])
            ODPQ = (intgMAT[O[0], O[1]] + intgMAT[Q[0], Q[1]]) - (intgMAT[D[0], D[1]] + intgMAT[P[0], P[1]])
            wScore = c1*CDEF + c2*ODPQ - c3*(ABGH - CDEF)
        else:
            w = (l[0]-1)/2
            w2 = np.math.ceil(w/5)

            A = [int(i[0]-2*w-1+i_indent),int(i[0]-w-1+j_indent)]
            B = [int(i[0]-2*w-1+i_indent),int(i[0]+w+j_indent)]
            C = [int(i[0]-w-1+i_indent),int(i[0]-w-1+j_indent)]
            D = [int(i[0]-w-1+i_indent),int(i[0]+w+j_indent)]
            E = [int(i[0]+w+i_indent),int(i[0]-w-1+j_indent)]
            F = [int(i[0]+w+i_indent),int(i[0]+w+j_indent)]
            G = [int(i[0]+2*w+i_indent),int(i[0]-w-1+j_indent)]
            H = [int(i[0]+2*w+i_indent),int(i[0]+w+j_indent)]
           
            O = [int(i[0]-w-1+i_indent),int(i[0]+w-w2+j_indent)]
            P = [int(i[0]-w+w2-1+i_indent),int(i[0]+w-w2+j_indent)]
            Q = [int(i[0]-w+w2-1+i_indent),int(i[0]+w+j_indent)]

            ABGH = (intgMAT[A[0], A[1]] + intgMAT[H[0], H[1]]) - (intgMAT[B[0], B[1]] + intgMAT[G[0], G[1]])
            CDEF = (intgMAT[C[0], C[1]] + intgMAT[F[0], F[1]]) - (intgMAT[D[0], D[1]] + intgMAT[E[0], E[1]])
            ODPQ = (intgMAT[O[0], O[1]] + intgMAT[Q[0], Q[1]]) - (intgMAT[D[0], D[1]] + intgMAT[P[0], P[1]])
            wScore = c1*CDEF + c2*ODPQ - c3*(ABGH - CDEF)
        
        res[0] = wScore/np.power(l[0],0.55)

    @guvectorize([(float64[:,:],float64[:], int64[:],int64[:], float64[:])], '(m,m),(m),(),(n)->(n)',target='parallel')
    def parDP(scores,A,j,k, res):
        for counter in range(k.shape[0]):
            res[counter] = A[k[counter]] + scores[k[counter]+1,j[0]]#A[k[counter]+1,i[counter]+l[0]-1]

    def findTADs(intgMAT):
        st = time.time()
        N = intgMAT.shape[0]
        intgMAT = np.lib.pad(intgMAT, ((2 * maxW + 1, 2*maxW), (2*maxW + 1, 2*maxW)), 'constant', constant_values=0)
        print_flag = int((N-minL)/20)
        scores = np.zeros([N,N],dtype=np.float64)
        ctcf_scores = np.zeros([N,N],dtype=np.float64)

        T = np.zeros(N,dtype=np.float64)
        backT = np.zeros(N,dtype=np.float64)
        for i in range(N):
            backT[i] = i
        # pre-compute scores for sub-matrices
        long_i = np.repeat(np.arange(int(np.floor(minL/2)-(minL+1)%2),N-int(np.floor(minL/2))-1+1),maxL-minL+1) #change it to N-minL/2    
        long_l = np.tile(np.arange(minL,maxL+1),int(N-minL)+1) 
        temp_s = score(intgMAT,long_i,long_l)
        sc = np.zeros([N+maxL,N+maxL],dtype=np.float64)
        sc[(long_i-np.floor(long_l/2)+(long_l+1)%2).astype(int)+maxL,(long_i+np.floor(long_l/2)).astype(int)] = temp_s            
        scores = sc[maxL:N+maxL,0:N]

        for j in range(minL-1,N):
            k_size = np.min(list([j,maxL-1]))
            kk = np.arange(j-k_size,j)     
            vals = parDP(scores,T,j,kk)
            T[j] = np.max([np.max(vals),T[j-1]])
            if(T[j]>T[j-1]):
                backT[j] = kk[np.argmax(vals)]+1

        # Extracting TADs based on backT
        counter = 0
        q = []
        TAD = []
        TAD.append(0)
        TAD.append(backT.shape[0]-1)
        q.append(backT.shape[0]-1)
        while len(q)>0:
            j = q.pop()
            k = backT[int(j)]
            if k != j:
                q.append(k-1)
                #vars.append((scores[int(k),int(j)]))
                if((scores[int(k),int(j)])>250 ): 
                    counter+=1
                    TAD.append(k)
                        
            elif j != 0:
                q.append(j-1)
        # threshold = np.mean(vars) - 1*np.sqrt( np.var(vars))
        print('Number of TADs:',counter)
        TAD.sort()
        out = np.asarray(TAD)
        out.reshape([len(TAD),1])
        print('Time to Identify TADs: ',time.time()-st)
        print()
        np.savetxt('TAD_'+SPECIES+'_nij_chr'+str(CHRM)+'_'+str(CUT),out,delimiter=' ',fmt='%d')

    diagSum = np.zeros(sampleSize)
    diagMean = np.zeros(sampleSize)
    rectMean = np.zeros(sampleSize)
    counter = 0
    S = 0
    for delta in range(sampleSize):
        S = 0
        counter = 0
        for i in range(chr1.shape[0]-delta):
            counter+=1
            S+= chr1[i,i+delta]

        diagMean[delta] = S/(counter+1)
        rectMean[delta] = np.dot(diagMean[0:delta+1],np.arange(delta+1,0,-1)) 
    
    for i in range(chr1.shape[0]):
        for j in range(i,np.min([i+sampleSize,chr1.shape[1]])):
            chr1[i,j]=chr1[i,j]/diagMean[j-i]
    # compute the integral image
    intgMat1 = np.zeros(chr1.shape,dtype=np.float64)
    intgMat1[0,0]=chr1[0,0]
    st = time.time()
    for i in range(1,intgMat1.shape[1]):
        intgMat1[0,i] = intgMat1[0,i-1]+chr1[0,i]
    for i in range(1,intgMat1.shape[0]):
        intgMat1[i,0] = intgMat1[i-1,0]+chr1[i,0]
    for i in range(1,intgMat1.shape[0]):
        for j in range(1,intgMat1.shape[1]):
            intgMat1[i,j] = intgMat1[i-1,j]+intgMat1[i,j-1]-intgMat1[i-1,j-1]+ chr1[i,j]
    
    print('time to compute the integral image:',time.time()-st)
    findTADs(intgMat1)
