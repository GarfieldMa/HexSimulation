
function Uterm=buildUtermMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,U0)
BaseL=length(K1up);
Uterm=zeros(BaseL,BaseL);

for K=1:BaseL
    %diagonal interaction part of Hamiltonian
    n1up=K1up(K); n1dn=K1dn(K);n2up=K2up(K); n2dn=K2dn(K);
    n3up=K3up(K); n3dn=K3dn(K);n4up=K4up(K); n4dn=K4dn(K);
    n5up=K5up(K); n5dn=K5dn(K);n6up=K6up(K); n6dn=K6dn(K);
    
    Uterm(K,K)=U0/2*(n1up*(n1up-1)+n1dn*(n1dn-1))+U0/2*(n2up*(n2up-1)+n2dn*(n2dn-1))+U0/2*(n3up*(n3up-1)+n3dn*(n3dn-1))+U0/2*(n4up*(n4up-1)+n4dn*(n4dn-1))+U0/2*(n5up*(n5up-1)+n5dn*(n5dn-1))+U0/2*(n6up*(n6up-1)+n6dn*(n6dn-1));
    

end
    
end