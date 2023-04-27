
import numpy as np


def ctc_loss(params, seq, epsilon=0, is_prob = True):

    seqLen = seq.shape[0] # Length of label sequence (# chars)
    L = 2*seqLen + 1 # Length of label sequence with blanks
    T = params.shape[1] # Length of utterance (time)

    alphas = np.zeros((L,T))
    betas = np.zeros((L,T))

    # Initialize alphas and forward pass
    alphas[0,0] = params[epsilon,0]
    alphas[1,0] = params[seq[0],0]

    c = np.sum(alphas[:,0])
    alphas[:,0] = alphas[:,0] / c

    for t in range(1,T):
        start = max(0, L - 2 * ( T - t))
        end = min(2*t+2,L)
        for s in range(start,L):
            l = (s-1)//2
            # epsilon
            if s%2 == 0:
                if s==0:
                    alphas[s,t] = alphas[s,t-1] * params[epsilon,t]
                else:
                    alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[epsilon,t]
            # repeated labels
            elif s == 1 or seq[l] == seq[l-1]:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1]) * params[seq[l],t]
            else:
                alphas[s,t] = (alphas[s,t-1] + alphas[s-1,t-1] + alphas[s-2,t-1]) * params[seq[l],t]

	# normalize at current time (prevent underflow)
    c = np.sum(alphas[start:end,t])
    alphas[start:end,t] = alphas[start:end,t] / c
    np.savetxt('alpha.csv', alphas, delimiter=',')

    # Initialize betas for backwards pass
    betas[-1,-1] = params[epsilon,-1]
    betas[-2,-1] = params[seq[-1],-1]
    c = np.sum(betas[:,-1])
    betas[:,-1] = betas[:,-1] / c


    for t in range(T-2,-1,-1):
        start = max(0,L-2*(T-t))
        end = min(2*t+2,L)

        for s in range(end-1,-1,-1):
            l = (s-1)//2

    	    # blank
            if s%2 == 0:
                if s == L-1:
                    betas[s,t] = betas[s,t+1] * params[epsilon,t]
                else:
                    betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[epsilon,t]

            # repeated labels
            elif s == L-2 or seq[l] == seq[l+1]:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1]) * params[seq[l],t]
            else:
                betas[s,t] = (betas[s,t+1] + betas[s+1,t+1] + betas[s+2,t+1]) * params[seq[l],t]

    c = np.sum(betas[start:end,t])
    betas[start:end,t] = betas[start:end,t] / c
    np.savetxt('beta.csv', alphas, delimiter=',')



    #gama
    gammas = np.zeros((L, T))
    for t in range(T):
        start = max(0, L - 2 * (T - t))
        end = min(2 * t + 2, L)
        for s in range(start, end):
            gammas[s, t] = alphas[s, t] * betas[s, t]

        # normalize gamma values for current time step
        c = np.sum(gammas[start:end, t])
        gammas[start:end, t] = gammas[start:end, t] / c

    np.savetxt('gamma.csv', alphas, delimiter=',')
    logll=k=0
    for t in range(T):
        for s in range(L):
            l = (s - 1) // 2
            if s%2 == 0:
                k += (gammas[s, t] / params[epsilon,t])
            else:
                k += (gammas[s, t] / params[seq[l], t])
        logll += np.sum(np.log(np.sum(k)))

    return logll

if __name__=='__main__':

    for _ in range(1):
        numchars = 4
        seqLen = 5
        uttLen = 12
        seq = np.floor(np.random.rand(seqLen,1)*numchars)
        seq = seq.astype(np.int32)
        params = np.random.randn(numchars,uttLen)
        params = np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), 0, params)
        lossvalue = ctc_loss(params,seq)
        np.savetxt('loss.csv', lossvalue, delimiter=',')
        print(f"Total loss value is: {lossvalue}")



