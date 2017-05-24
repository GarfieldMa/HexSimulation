
function Tterm=buildTtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1up,t1dn,t2up,t2dn,t3up,t3dn)
BaseL=length(K1up);
Tterm=zeros(BaseL,BaseL);

for K=1:BaseL
    k1up=K1up(K); k1dn=K1dn(K);k2up=K2up(K); k2dn=K2dn(K);
    k3up=K3up(K); k3dn=K3dn(K);k4up=K4up(K); k4dn=K4dn(K);
    k5up=K5up(K); k5dn=K5dn(K);k6up=K6up(K); k6dn=K6dn(K);
   
    %off-diagonal kinetic hopping part of Hamiltonian
    for L=1:BaseL
		l1up=K1up(L); l1dn=K1dn(L);l2up=K2up(L); l2dn=K2dn(L);
        l3up=K3up(L); l3dn=K3dn(L);l4up=K4up(L); l4dn=K4dn(L);
        l5up=K5up(L); l5dn=K5dn(L);l6up=K6up(L); l6dn=K6dn(L);
        
 %=======================spin-up intra-cluster tunneling terms terms ai^{dagger}aj===================               
        if k1up==l1up+1 && k1dn==l1dn && k2up==l2up-1 && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t2up)*sqrt(k1up*l2up);
        end
        if k1up==l1up-1 && k1dn==l1dn && k2up==l2up+1 && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t2up*sqrt(k2up*l1up);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up+1 && k2dn==l2dn && k3up==l3up-1 && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t1up*sqrt(k2up*l3up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up-1 && k2dn==l2dn && k3up==l3up+1 && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t1up)*sqrt(k3up*l2up);
        end
        
        if k1up==l1up-1 && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up+1 && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t3up*sqrt(k6up*l1up);
        end
        if k1up==l1up+1 && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up-1 && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t3up)*sqrt(k1up*l6up);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up+1 && k5dn==l5dn && k6up==l6up-1 && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t1up)*sqrt(k5up*l6up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up-1 && k5dn==l5dn && k6up==l6up+1 && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t1up*sqrt(k6up*l5up);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up+1 && k4dn==l4dn && k5up==l5up-1 && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t2up*sqrt(k4up*l5up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up-1 && k4dn==l4dn && k5up==l5up+1 && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t2up)*sqrt(k5up*l4up);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up+1 && k3dn==l3dn && k4up==l4up-1 && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t3up)*sqrt(k3up*l4up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up-1 && k3dn==l3dn && k4up==l4up+1 && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t3up*sqrt(k4up*l3up);
        end

   %=======================spin-dn intra-cluster tunneling terms ai^{dagger}aj===================         
        if k1up==l1up && k1dn==l1dn+1 && k2up==l2up && k2dn==l2dn-1 && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t2dn)*sqrt(k1dn*l2dn);
        end
        if k1up==l1up && k1dn==l1dn-1 && k2up==l2up && k2dn==l2dn+1 && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t2dn*sqrt(k2dn*l1dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn+1 && k3up==l3up && k3dn==l3dn-1 && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t1dn*sqrt(k2dn*l3dn);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn-1 && k3up==l3up && k3dn==l3dn+1 && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t1dn)*sqrt(k3dn*l2dn);
        end
        
        if k1up==l1up && k1dn==l1dn-1 && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn+1
			Tterm(K,L)=Tterm(K,L)-t3dn*sqrt(k6dn*l1dn);
        end
        if k1up==l1up && k1dn==l1dn+1 && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn-1
			Tterm(K,L)=Tterm(K,L)-conj(t3dn)*sqrt(k1dn*l6dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn+1 && k6up==l6up && k6dn==l6dn-1
			Tterm(K,L)=Tterm(K,L)-conj(t1dn)*sqrt(k5dn*l6dn);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn-1 && k6up==l6up && k6dn==l6dn+1
			Tterm(K,L)=Tterm(K,L)-t1dn*sqrt(k6dn*l5dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn+1 && k5up==l5up && k5dn==l5dn-1 && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t2dn*sqrt(k4dn*l5dn);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn-1 && k5up==l5up && k5dn==l5dn+1 && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t2dn)*sqrt(k5dn*l4dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn+1 && k4up==l4up && k4dn==l4dn-1 && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-conj(t3dn)*sqrt(k3dn*l4dn);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn-1 && k4up==l4up && k4dn==l4dn+1 && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			Tterm(K,L)=Tterm(K,L)-t3dn*sqrt(k4dn*l3dn);
        end
        
        
end	
end
    
end