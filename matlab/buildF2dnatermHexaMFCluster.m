function F2dnaterm=buildF2dnatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3dn)
BaseL=length(K1up);
F2dnaterm=zeros(BaseL,BaseL);

for K=1:BaseL
    
    k1up=K1up(K); k1dn=K1dn(K);k2up=K2up(K); k2dn=K2dn(K);
    k3up=K3up(K); k3dn=K3dn(K);k4up=K4up(K); k4dn=K4dn(K);
    k5up=K5up(K); k5dn=K5dn(K);k6up=K6up(K); k6dn=K6dn(K);

    %off-diagonal kinetic hopping part of Hamiltonian
    for L=1:BaseL
		l1up=K1up(L); l1dn=K1dn(L);l2up=K2up(L); l2dn=K2dn(L);
        l3up=K3up(L); l3dn=K3dn(L);l4up=K4up(L); l4dn=K4dn(L);
        l5up=K5up(L); l5dn=K5dn(L);l6up=K6up(L); l6dn=K6dn(L);
        
  %====================spin-dn meanfield terms with a=============================      
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn-1 && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			F2dnaterm(K,L)=F2dnaterm(K,L)-conj(t3dn)*sqrt(l2dn);
        end
    end
end
    
end