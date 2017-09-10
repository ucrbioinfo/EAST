import numpy as np
from timeit import time
from numba import guvectorize,float64,int64
import os
import itertools
from scipy import sparse
from detect_peaks import detect_peaks
import importlib
pnd = importlib.find_loader('pandas')
PANDAS_INSTALLED = pnd is not None
if PANDAS_INSTALLED:
    import pandas as pd

#minL = 9
#sampleSize = 2*maxL
RESOLUTION = 500000
#RES_THRESH = 5000
W = 10
maxL = 2*int(np.round(3200000/RESOLUTION)) + 1
Nfactor = 0.35

class species:
    K526 = 'K526'
    hES = 'hES'
    mES = 'mES'
class dataType:
    Dixon = 'Dixon'
    Rao = 'Rao'

SPECIES = species.K526
DataType = dataType.Rao 

if SPECIES=='hIMR90' or SPECIES=='hES' or SPECIES=='K526':
    NUM_OF_CHRMS = 22
elif SPECIES == 'mES' or 'mCO':
    NUM_OF_CHRMS = 20
for CHRM in range(1,NUM_OF_CHRMS+1):
    if DataType == dataType.Rao:
        print('Loading Chromosome '+ str(CHRM))
        st = time.time()
        if PANDAS_INSTALLED:
            if RESOLUTION < 1000000:
                name = str(int(RESOLUTION/1000))+'kb'
            else:
                name = str(int(RESOLUTION/1000000))+'mb'
            chr1Data = pd.read_csv(os.path.abspath(os.sep)+'Users/Abbas/Google Drive/Research/Dataset/'+SPECIES+'/'+name+'_resolution_intrachromosomal/chr'+str(CHRM)+'/MAPQGE30/chr'+str(CHRM)+'_'+name +'.RAWobserved',sep='\t',header=None)
            chr1Data = chr1Data.values
            chr1Data[:,0:2] = np.floor(chr1Data[:,0:2]/RESOLUTION)
            knorm = pd.read_csv(os.path.abspath(os.sep)+'Users/Abbas/Google Drive/Research/Dataset/'+SPECIES+'/'+name+'_resolution_intrachromosomal/chr'+str(CHRM)+'/MAPQGE30/chr'+str(CHRM)+'_'+ name+'.Rawexpected',sep='\t',header=None)

        else:
            chr1Data = np.genfromtxt(os.path.abspath(os.sep)+'Users/Abbas/Google Drive/Research/Dataset/'+SPECIES+'/'+str(int(RESOLUTION/1000))+'kb_resolution_intrachromosomal/chr'+str(CHRM)+'/MAPQGE30/chr'+str(CHRM)+'_'+name+'.RAWobserved')
            chr1Data[:,0:2] = np.round(chr1Data[:,0:2]/RESOLUTION)
            knorm = np.genfromtxt(os.path.abspath(os.sep)+'Users/Abbas/Google Drive/Research/Dataset/'+SPECIES+'/'+str(int(RESOLUTION/1000))+'kb_resolution_intrachromosomal/chr'+str(CHRM)+'/MAPQGE30/chr'+str(CHRM)+'_'+name+'.Rawexpected')
        # Normalizing the data
        for i in range(chr1Data.shape[0]):
            chr1Data[i,2] = chr1Data[i,2] / knorm[0][chr1Data[i,1] - chr1Data[i,0]]
        chr1 = sparse.csr_matrix((chr1Data[:,2],(chr1Data[:,0],chr1Data[:,1])))
        # comment this line if you don't have enough memory to store the dense matrix for higher resolution
        chr1 = chr1.todense()
        print('time to read the chromosome',CHRM,'and normalizing it:',time.time()-st)
    elif DataType == dataType.Dixon: # Dixon data Type 
        print('Loading Chromosome '+ str(CHRM))
        st = time.time()
        if PANDAS_INSTALLED:
            chr1 = pd.read_csv(os.path.abspath(os.sep)+'Users/Abbas/Google Drive/Research/Dataset/'+SPECIES+'/nij/nij.chr'+str(CHRM),sep='\t',header=None)
            chr1 = chr1.values
        else:
            chr1 = np.genfromtxt(os.path.abspath(os.sep)+'Users/Abbas/Google Drive/Research/Dataset/'+SPECIES+'/nij/nij.chr'+str(CHRM))
        print('time to read the chromosome ',CHRM,':',time.time()-st)
    N = chr1.shape[0]

    # compute the integral image (We don't care about the values on the diagonal)
    st = time.time()

    #if RESOLUTION >= RES_THRESH:
    #    #for i in range(chr1.shape[0]):
    #    #    intgMat[i,i] = chr1[i,i]
    #    intgMat = np.zeros(chr1.shape,dtype=np.float64)
    #    for delta in range(1,2*maxL):
    #        for i in range(chr1.shape[0]-delta):
    #            intgMat[i,i+delta] = chr1[i,i+delta] + intgMat[i+1,i+delta] + intgMat[i,i+delta-1] - intgMat[i+1,i+delta-1]
    #else: 
        #for i in range(chr1.shape[0]):
        #    intgMat[0,i] = chr1[i,i]
    intgMat = np.zeros([2*maxL,N],dtype=np.float64)
    for i in range(chr1.shape[0]-1): # delta = 1                
        intgMat[1,i+1] = chr1[i,i+1] + intgMat[0,i+1] + intgMat[0,i] 
    for delta in range(2,2*maxL):
        for i in range(chr1.shape[0]-delta):               
            intgMat[delta,i+delta] = chr1[i,i+delta] + intgMat[delta-1,i+delta] + intgMat[delta-1,i+delta-1] - intgMat[delta-2,i+delta-1] 
        
    

    #np.savetxt('intgMat',intgMat,delimiter=' ',fmt='%s')
    print('time to compute the integral image:',time.time()-st)
    #print('********************** TAD DETECTION *****************************')
    # TAD Detection
    @guvectorize([(float64[:,:], int64[:], int64[:], float64[:])], '(m,p),(),()->()',target='parallel')
    def score(intgMAT,i,l,res):
        i_indent = maxL
        j_indent = maxL
        wScore = 0
        pixel = i[0]
        c1 = 1.0 
        alpha = 0.1  #penalty for inter-TAD interactions
        if(l[0]<=maxL or l[0]<5):
            if l[0] % 2 == 0:
                
                w = (l[0])/2
                pixel = pixel + w
                w2 = np.math.ceil(w/5)

                A = [int(pixel-w-l[0]+i_indent),int(pixel-w+j_indent)]
                B = [int(pixel-w-l[0]+i_indent),int(pixel+w+j_indent)]
                D = [int(pixel-w+i_indent),int(pixel+w+j_indent)]
                E = [int(pixel-w+i_indent),int(pixel+w+l[0]+j_indent)]
                F = [int(pixel+w+i_indent),int(pixel+w+l[0]+j_indent)]
            else:
                w = (l[0]-1)/2
                pixel = pixel + w
                w2 = np.math.ceil(w/5)

                A = [int(pixel-w-l[0]-1+i_indent),int(pixel-w-1+j_indent)]
                B = [int(pixel-w-l[0]-1+i_indent),int(pixel+w+j_indent)]
                D = [int(pixel-w-1+i_indent),int(pixel+w+j_indent)]
                E = [int(pixel-w-1+i_indent),int(pixel+w+l[0]+j_indent)]
                F = [int(pixel+w+i_indent),int(pixel+w+l[0]+j_indent)]

            #if RESOLUTION >= RES_THRESH:
            #   wScore = (2+2*alpha)*intgMAT[D[0], D[1]] - alpha*(intgMAT[B[0], B[1]] +intgMAT[E[0], E[1]] -intgMAT[A[0], A[1]]-intgMAT[F[0], F[1]]  )
            #   #wScore = intgMAT[D[0], D[1]] 
            #else:
            #wScore = (2+2*alpha)*intgMAT[D[1]-D[0]+i_indent, D[1]] - alpha*(intgMAT[B[1]-B[0]+i_indent, B[1]] +intgMAT[E[1]-E[0]+i_indent, E[1]] -intgMAT[A[1]-A[0]+i_indent, A[1]]-intgMAT[F[1]-F[0]+i_indent, F[1]]  )
            wScore = intgMAT[D[1]-D[0]+i_indent, D[1]]
            res[0] = wScore/np.power(l[0],Nfactor)
        else:
            res[0]=0

    @guvectorize([(float64[:,:], int64[:], int64[:], float64[:])], '(m,p),(),(o)->(o)',target='parallel')
    def det_score(intgMAT,i,w,res):
        i_indent = maxL
        j_indent = maxL
        #A = [int(i[0]-w[0]+i_indent),int(i[0]-w[0]+j_indent)]
        B = [int(i[0]-w[0]+i_indent),int(i[0]+j_indent)]
        C = [int(i[0]-w[0]+i_indent),int(i[0]+w[0]+j_indent)]
        #D = [int(i[0]-100+i_indent),int(i[0]+100+j_indent)]
        #D = [int(i[0]+i_indent),int(i[0]-w[0]+j_indent)]
        #E = [int(i[0]+i_indent),int(i[0]+j_indent)]
        F = [int(i[0]+i_indent),int(i[0]+w[0]+j_indent)]
        #G = [int(i[0]+w[0]+i_indent),int(i[0]+j_indent)]
        #H = [int(i[0]+w[0]+i_indent),int(i[0]+w[0]+j_indent)]
        
        #ABDE = (intgMAT[A[0], A[1]] + intgMAT[E[0], E[1]]) - (intgMAT[B[0], B[1]] + intgMAT[D[0], D[1]])
        #EFGH = (intgMAT[E[0], E[1]] + intgMAT[H[0], H[1]]) - (intgMAT[F[0], F[1]] + intgMAT[G[0], G[1]])
        #BCEF = (intgMAT[B[0], B[1]] + intgMAT[F[0], F[1]]) - (intgMAT[C[0], C[1]] + intgMAT[E[0], E[1]])

        #res[0] = BCEF
        #res[1] = ABDE
        #res[2] = EFGH
        #if RESOLUTION >= RES_THRESH:
        #    a = intgMAT[B[0], B[1]] #left
        #    f = intgMAT[F[0], F[1]] #right
        #    res[0] = intgMAT[C[0], C[1]] - a - f #center
        #    res[1] = (a - res[0])
        #    res[2] = (f - res[0])
        #    res[3] = intgMAT[C[0], C[1]]
        #    #/max(1,intgMAT[D[0], D[1]])
        #    #/max(1,intgMAT[D[0], D[1]])
        #    #res[0] =  intgMAT[C[0], C[1]] #2*res[1]+2*res[2]-
        #    #res[0] =  np.sqrt((res[0] -res[1])**2 + (res[0] - res[2])**2)
        #else:
        a = intgMAT[B[1]-B[0]+i_indent, B[1]] #left
        f = intgMAT[F[1]-F[0]+i_indent, F[1]] #right
        res[0] = intgMAT[C[1]-C[0] +i_indent, C[1]] - a - f #center
        res[1] = (a - res[0])
        res[2] = (f - res[0])
        res[3] =  intgMAT[C[1]-C[0], C[1]]


    @guvectorize([(float64[:,:],int64[:],int64[:],float64[:], int64[:], int64[:],int64[:],int64[:],int64[:], float64[:])], '(u,v),(s),(s),(r),(p),(q),(p),(),(n)->(n)',target='parallel')
    def parDP(scores,i2I,j2J,T,startPeaks,endPeaks,prev_end,j,k, res):
        for counter in range(k.shape[0]):
            res[counter] = T[prev_end[k[counter]]] + scores[i2I[startPeaks[k[counter]]], j2J[endPeaks[j[0]]]]#scores[startPeaks[k[counter]], endPeaks[j[0]]]
           
    def findTADs(intgMAT):
        st = time.time()
        intgMAT2 = np.lib.pad(intgMAT, ((maxL, maxL), (maxL, maxL)), 'constant', constant_values=0)       
        det_scores = np.zeros([N,4],dtype=np.float64)
        # pre-compute scores for sub-matrices
        #long_i = np.repeat(np.arange(int(np.floor(minL/2)-(minL+1)%2),N-int(np.floor(minL/2))-1+1),maxL-minL+1) #change it to N-minL/2    
        #long_l = np.tile(np.arange(minL,maxL+1),int(N-minL)+1) 
        det_scores = det_score(intgMAT2,np.arange(N),[W,W,W,W])
        #print('first part takes: ',time.time()-st)
        #st2 = time.time()
        #st2 = time.time()
        S = np.sum(det_scores[:,0])/N
        det_scores[:,1] = det_scores[:,1]/S #np.divide(det_scores[:,1],det_scores[:,3]+1)
        det_scores[:,2] = det_scores[:,2]/S #np.divide(det_scores[:,2],det_scores[:,3]+1)
        #np.savetxt('long',det_scores[:,2],delimiter=' ',fmt='%s')

        start_peaks = np.asarray(detect_peaks(det_scores[:,2],  mpd=2),dtype=np.int64) #np.median(det_scores[:,1])
        end_peaks   = np.asarray(detect_peaks(det_scores[:,1],  mpd=2),dtype=np.int64) #np.median(det_scores[:,2])
        #print(start_peaks,end_peaks)
        #print('second part takes: ',time.time()-st2)
        #st2 = time.time()
                # we need to know the preceding ends right before each start
        prev_end = np.zeros(len(start_peaks),dtype=np.int64)
        k_old = 0
        for i in range(len(start_peaks)):
            k = k_old
            for j in range(k,len(end_peaks)):
                if end_peaks[k] <= start_peaks[i]:
                    k = k + 1
                else:
                    break
            if k==0:
                prev_end[i] = -1
                k_old = 0
            elif end_peaks[k-1] < start_peaks[0]:
                prev_end[i] = -1
                k_old = k-1
            else:
                prev_end[i] = k-1
                k_old = k - 1
        # make prev_start
        prev_start = np.zeros(len(end_peaks),dtype=np.int64)
        k_old = 0
        for i in range(len(end_peaks)):
            k = k_old
            for j in range(k,len(start_peaks)):
                if start_peaks[k] <= end_peaks[i]:
                    k = k + 1
                else:
                    break
            if k==0:
                prev_start[i] = -1
                k_old = 0 
            else:
                prev_start[i] = k-1
                k_old = k - 1


        
        # pre-compute scores for start/end combinations
        #intgMAT2 = np.lib.pad(intgMAT, ((2 * maxW + 1, 2*maxW), (2*maxW + 1, 2*maxW)), 'constant', constant_values=0)
        M1 = len(start_peaks)
        M2 = len(end_peaks)

        rep = []
        indx = []
        for i in range(M1):
            ind = prev_end[i]
            indx.append(min(ind+1,len(end_peaks)))
            if ind == -1:
                rep.append(M2)
            elif ind == M2:
                rep.append(0)
            else:
                rep.append(M2 - ind - 1)
            #rep.append(max(0, min(len(end_peaks)-min(ind+1, len(end_peaks)), len(end_peaks))))
            
        long_i = np.repeat(start_peaks,rep)
        long_j = np.zeros(np.sum(rep),dtype=np.int64)   
        for i in range(M1):
            #print(i,M1 - 1,rep[0])
            start = sum(rep[0:i])
            #print(st,st+rep[i]-1)
            #print(indx[i],len(end_peaks))
            long_j[start:start+rep[i]] = end_peaks[indx[i]:len(end_peaks)] 

        long_l = long_j - long_i + 1
        temp_s = score(intgMAT2,long_i,long_l) #long_i is the center of a domain; long_l is the length of that domain
        #scores = np.zeros([N,N],dtype=np.float64)
        #scores[long_i,long_j] = temp_s

        # build a dense representation of the 'scores' but much smaller than N*N (for higher resolutions)
        i2I = np.zeros(N,dtype=np.int64)
        for i in range(len(start_peaks)):
            i2I[start_peaks[i]] = i
        j2J = np.zeros(N,dtype=np.int64)
        for j in range(len(end_peaks)):
            j2J[end_peaks[j]] = j

        scores = np.zeros([len(start_peaks),len(end_peaks)],dtype=np.float64)
        scores[i2I[long_i],j2J[long_j]] = temp_s
        
        #cc = 0
        #for i in range(len(long_i)):
        ##test = 3380
        #    if long_j[i] - long_i[i] + 1 != long_l[i]:
        #        cc = cc + 1
        #        #print(i,long_i[i],long_j[i],long_l[i])
        #print(len(long_i),cc)
                
        ##print(i2I[long_i[test]],j2J[long_j[test]],scores[i2I[long_i[test]],j2J[long_j[test]]])
        #exit(0)

        #for k in range(len(temp_s)):
        #    scores[i2I[long_i[k]],j2J[long_j[k]]] = temp_s[k]
        # Dynamic Programming applies on potential start/end TAD boundaries we have found above
        T = np.zeros(len(end_peaks),dtype=np.float64)
        backT = np.zeros(len(end_peaks),dtype=np.int64) # shows the index of the start point (out of all start points) for each end point
        #for i in range(N):
        #    backT[i] = i
        k_old = 0
        for j in range(len(end_peaks)):
            if prev_start[j] == -1:
                backT[j] = -1
                continue
            #k_size = np.min(list([j,maxL-1]))
            k = k_old
            #print('j:',end_peaks[j],start_peaks[k])
            #while start_peaks[k] < end_peaks[j]:
            #    k=k+1;

            kk = np.arange(int(prev_start[j])+1)  
             
            #for ii in range(len(kk)):
            #    print(start_peaks[kk[ii]],end_peaks[j],scores[start_peaks[kk[ii]],end_peaks[j]])
            #start_peaks = np.arange(10)
            #end_peaks = np.arange(10)
            #prev_end = np.arange(3)
            vals = parDP(scores,i2I,j2J,T,start_peaks,end_peaks,prev_end,j,kk)
            #vals = parDP(scores,T,np.arange(5),np.arange(5),np.arange(5),np.arange(5),np.arange(5))
            #print(vals)
            T[j] = np.max([np.max(vals),T[j-1]])
            #if(T[j]>T[j-1]):
            backT[j] = np.argmax(vals)
            #print(start_peaks[backT[j]],end_peaks[j])
            k_old = k
        #np.savetxt('T',backT,delimiter=' ',fmt='%d')
        #exit(0)
        #[(start_peaks[i], end_peaks[j]) for i in range(len(start_peaks)) for j in xrange(len(end_peaks))]        #list1=[1,2,3,4]
        #list2=[2,3,4,5]
        #a = [zip(x,list2) for x in itertools.permutations(list1,len(list2))]

        #np.savetxt('score1.txt',scores[:,1],delimiter=' ',fmt='%d')
        #np.savetxt('score2.txt',scores[:,2],delimiter=' ',fmt='%d')
     
    
        #def_scores = np.zeros(N)
        #def_scores[0] = scores[0,0]*-2 + scores[1,0]*2
        #for i in range(1,N-1):
        #    def_scores[i] = scores[i,0]*-2 + scores[i+1,0] + scores[i-1,0]
        #def_scores[N-1] = scores[N-1,0]*-2 + scores[N-2,0]*2
        ##np.savetxt('A.txt',scores,delimiter=' ',fmt='%d')

        # Extracting TADs based on backT
        counter = len(end_peaks) - 1
        tadCount = 0
        TADx1 = []
        TADx2 = []
        while True:# backT[counter] != 0 and backT[counter] != -1:
            
            #print(counter,backT[counter],prev_end[backT[counter]])
            if backT[counter] == -1:
                break
            elif backT[counter] == 0:
                TADx2.append(end_peaks[counter])
                TADx1.append(start_peaks[0])
                tadCount = tadCount + 1
                break
            elif counter > 0:
                TADx2.append(end_peaks[counter])
                TADx1.append(start_peaks[backT[counter]])
                tadCount = tadCount + 1
                counter = prev_end[backT[counter]]

                
                if prev_end[backT[counter]] == counter:
                    counter = counter - 1
                if counter == -1:
                    break
            elif counter == 0:
                TADx2.append(end_peaks[counter])
                TADx1.append(start_peaks[backT[counter]])
                tadCount = tadCount + 1
                break
        # threshold = np.mean(vars) - 1*np.sqrt( np.var(vars))
        print('Number of TADs:',tadCount)
        TADx1.sort()
        TADx2.sort()
        TADx1 = (np.asarray(TADx1)).reshape([len(TADx1),1])
        TADx2 = (np.asarray(TADx2)).reshape([len(TADx2),1])
        TAD = np.concatenate((TADx1,TADx2),axis=1)
        #print('Time to Identify part of TADs: ',time.time()-st2)
        print('Time to Identify TADs: ',time.time()-st)
        #print()
        ## Find potential boundaries using peak calling
        #print('Detect peaks with minimum height and distance filters.')
        #print(scores[:,0],np.max(scores[:,0]) - scores[:,0])

        #

        #nbrhood_test = np.logical_or(scores[peaks_loc,1] > 18 , scores[peaks_loc,2] > 18)
        
        #print(peaks_loc)
        #print(nbrhood_test)
        #print(peaks_loc[nbrhood_test])
        #np.concatenate((start_peaks,end_peaks),axis=0)
        np.savetxt(SPECIES+'_nij_chr'+str(CHRM)+'_'+str(N),TAD,delimiter=' ',fmt='%d')
        #np.savetxt('Scores_'+SPECIES+'_nij_chr'+str(CHRM)+'_'+str(CUT),scores,delimiter=' ',fmt='%d')

    ##
    #diagSum = np.zeros(sampleSize)
    #diagMean = np.zeros(sampleSize)
    #rectMean = np.zeros(sampleSize)
    #counter = 0
    #S = 0
    #for delta in range(sampleSize):
    #    S = 0
    #    counter = 0
    #    for i in range(chr1.shape[0]-delta):
    #        counter+=1
    #        S+= chr1[i,i+delta]

    #    diagMean[delta] = S/(counter+1)
    #    rectMean[delta] = np.dot(diagMean[0:delta+1],np.arange(delta+1,0,-1)) 
    
    #for i in range(chr1.shape[0]):
    #    for j in range(i,np.min([i+sampleSize,chr1.shape[1]])):
    #        chr1[i,j]=chr1[i,j]/diagMean[j-i]
    findTADs(intgMat)
