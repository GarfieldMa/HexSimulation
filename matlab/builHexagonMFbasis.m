function [K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn]=builHexagonMFbasis(Nmax)%"up" ("down") is for Boson pseudospin
NN=(Nmax+1)*(Nmax+1)*(Nmax+1)*(Nmax+1);%dimension of the basis state, as well as the Hilbert space
K1up=zeros(NN,1); K1dn=zeros(NN,1);K2up=zeros(NN,1); K2dn=zeros(NN,1);
K3up=zeros(NN,1); K3dn=zeros(NN,1);K4up=zeros(NN,1); K4dn=zeros(NN,1);
K5up=zeros(NN,1); K5dn=zeros(NN,1);K6up=zeros(NN,1); K6dn=zeros(NN,1);

count=1;

for k1=0:Nmax
	for k2=0:Nmax
        for k3=0:Nmax
            for k4=0:Nmax
                for k5=0:Nmax
                    for k6=0:Nmax
                        for k7=0:Nmax
                            for k8=0:Nmax
                                for k9=0:Nmax
                                    for k10=0:Nmax
                                        for k11=0:Nmax
                                            for k12=0:Nmax
			K1up(count)=k1;
			K1dn(count)=k2;
            K2up(count)=k3;
			K2dn(count)=k4;
            
            K3up(count)=k5;
			K3dn(count)=k6;
            K4up(count)=k7;
			K4dn(count)=k8;
            
            K5up(count)=k9;
			K5dn(count)=k10;
            K6up(count)=k11;
			K6dn(count)=k12;
            
            count=count+1;
                                            
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
end

