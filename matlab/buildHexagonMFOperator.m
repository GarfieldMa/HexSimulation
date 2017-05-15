function  [b1up,b1dn,b2up,b2dn,b3up,b3dn,b4up,b4dn,b5up,b5dn,b6up,b6dn]=buildHexagonMFOperator(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn)
NN=length(K1up);
b1up=zeros(NN,NN);
b1dn=zeros(NN,NN);
b2up=zeros(NN,NN);
b2dn=zeros(NN,NN);

b3up=zeros(NN,NN);
b3dn=zeros(NN,NN);
b4up=zeros(NN,NN);
b4dn=zeros(NN,NN);

b5up=zeros(NN,NN);
b5dn=zeros(NN,NN);
b6up=zeros(NN,NN);
b6dn=zeros(NN,NN);

for I=1:NN
	k1up=K1up(I); k1dn=K1dn(I); k2up=K2up(I); k2dn=K2dn(I);
    k3up=K3up(I); k3dn=K3dn(I); k4up=K4up(I); k4dn=K4dn(I);
    k5up=K5up(I); k5dn=K5dn(I); k6up=K6up(I); k6dn=K6dn(I);
	for L=1:NN
		l1up=K1up(L); l1dn=K1dn(L); l2up=K2up(L); l2dn=K2dn(L);
        l3up=K3up(L); l3dn=K3dn(L); l4up=K4up(L); l4dn=K4dn(L);
        l5up=K5up(L); l5dn=K5dn(L); l6up=K6up(L); l6dn=K6dn(L);
        
		if k1up==l1up-1 && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b1up(I,L)=b1up(I,L)+sqrt(l1up);
		end
		if k1up==l1up && k1dn==l1dn-1 && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b1dn(I,L)=b1dn(I,L)+sqrt(l1dn);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up-1 && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b2up(I,L)=b2up(I,L)+sqrt(l2up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn-1 && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b2dn(I,L)=b2dn(I,L)+sqrt(l2dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up-1 && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b3up(I,L)=b3up(I,L)+sqrt(l3up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn-1 && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b3dn(I,L)=b3dn(I,L)+sqrt(l3dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up-1 && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b4up(I,L)=b4up(I,L)+sqrt(l4up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn-1 && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b4dn(I,L)=b4dn(I,L)+sqrt(l4dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up-1 && k5dn==l5dn && k6up==l6up && k6dn==l6dn
			b5up(I,L)=b5up(I,L)+sqrt(l5up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn-1 && k6up==l6up && k6dn==l6dn
			b5dn(I,L)=b5dn(I,L)+sqrt(l5dn);
        end
        
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up-1 && k6dn==l6dn
			b6up(I,L)=b6up(I,L)+sqrt(l6up);
        end
        if k1up==l1up && k1dn==l1dn && k2up==l2up && k2dn==l2dn && k3up==l3up && k3dn==l3dn && k4up==l4up && k4dn==l4dn && k5up==l5up && k5dn==l5dn && k6up==l6up && k6dn==l6dn-1
			b6dn(I,L)=b6dn(I,L)+sqrt(l6dn);
        end
        
        
	end
	
end
end