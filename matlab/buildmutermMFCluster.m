
function muterm=buildmutermMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,mu0)
BaseL=length(K1up);
muterm=zeros(BaseL,BaseL);

for K=1:BaseL

    %diagonal interaction part of Hamiltonian
    n1up=K1up(K); n1dn=K1dn(K);n2up=K2up(K); n2dn=K2dn(K);
    n3up=K3up(K); n3dn=K3dn(K);n4up=K4up(K); n4dn=K4dn(K);
    n5up=K5up(K); n5dn=K5dn(K);n6up=K6up(K); n6dn=K6dn(K);
    
    muterm(K,K)=-mu0*(n1up+n1dn+n2up+n2dn+n3up+n3dn+n4up+n4dn+n5up+n5dn+n6up+n6dn);
    
end
    
end