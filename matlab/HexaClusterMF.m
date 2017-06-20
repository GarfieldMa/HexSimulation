% Program (BoseHubbardHoneycomb.m) for the phase diagram of 2D Bose-Hubbard
% model with Cluster Meanfield Approach-Hexagonal Ring with Hardcore Boson
% case

clear all;
Nmax=1;
%import basis
[K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn]=builHexagonMFbasis(Nmax);
%import the bosonic operators for "Pseudospin-up" and "Pseudospin-down" for
%both A and B sublattices
[b1up,b1dn,b2up,b2dn,b3up,b3dn,b4up,b4dn,b5up,b5dn,b6up,b6dn]=buildHexagonMFOperator(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn);


EVHexmin=[];
%range setting of hopping strength 
tfirst=0.2;tsecond=0.4;n1=3;n2=3;
tA=[];
ta=linspace(0.05,tfirst,n1);%ta-part1,near phase transition boundary, need to be calculated more densely;
tb=linspace(tfirst,tsecond,n2);%tb-part2


%setting tunneling terms
%phase winding factor W:
W=2*pi/3;
t0=1;
t1=t0;
t2=t0*exp(1i*W);
t3=t0*exp(-1i*W);


t1up=t1;
t2up=t2;
t3up=t3;

t1dn=conj(t1up);
t2dn=conj(t2up);
t3dn=conj(t3up);


%setting chemical potential
mu0=1;
murange=1.5;
Ma=linspace(-0.5,1.5,5);%the range of mu, chemical potential

%setting on-site interactions
%range of on-site interaction of the two same pseudospin particles, fix U first
%U=linspace(0,V0,50);
U0=1;
U=U0;
%range of on-site interaction of the two different pseudospin particles, fix V first
%V=linspace(0,V0,50);
V0=1;
V=V0;

%buliding Hamiltonian terms 
Uterm=buildUtermMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,U0);
Vterm=buildVtermMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,V0);
muterm=buildmutermMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,mu0);
Tterm=buildTtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1up,t1dn,t2up,t2dn,t3up,t3dn);
tic

F1upaterm=buildF1upatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1up);
F2upaterm=buildF2upatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3up);
F3upaterm=buildF3upatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2up);
F4upaterm=buildF4upatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1up);
F5upaterm=buildF5upatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3up);
F6upaterm=buildF6upatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2up);

F1upadgterm=buildF1upadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1up);
F2upadgterm=buildF2upadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3up);
F3upadgterm=buildF3upadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2up);
F4upadgterm=buildF4upadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1up);
F5upadgterm=buildF5upadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3up);
F6upadgterm=buildF6upadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2up);


F1dnaterm=buildF1dnatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1dn);
F2dnaterm=buildF2dnatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3dn);
F3dnaterm=buildF3dnatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2dn);
F4dnaterm=buildF4dnatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1dn);
F5dnaterm=buildF5dnatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3dn);
F6dnaterm=buildF6dnatermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2dn);

F1dnadgterm=buildF1dnadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1dn);
F2dnadgterm=buildF2dnadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3dn);
F3dnadgterm=buildF3dnadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2dn);
F4dnadgterm=buildF4dnadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t1dn);
F5dnadgterm=buildF5dnadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t3dn);
F6dnadgterm=buildF6dnadgtermHexaMFCluster(K1up,K1dn,K2up,K2dn,K3up,K3dn,K4up,K4dn,K5up,K5dn,K6up,K6dn,t2dn);

DigH=eye((Nmax+1)^12);
toc

toc1=zeros();toc2=zeros();
T1=0;T2=0;

%the range of order parameters trial solution, the trial OrderParameter is Complex with Pa(i,j)=Pr*exp(i*theta)
Pr=linspace(0.01,sqrt(Nmax),10);
% theta=linspace(0,2*pi,20);
% Pa=zeros();
% for i=1:length(Pr)
%     for j=1:length(theta)
%         Pa(i,j)=Pr(i)*exp(1i*theta(j));
%     end
% end

p=1;
while p<=n1
    tA(p)=ta(p);
    p=p+1;
end
q=n1+1;
 while q<=(n2+n1)
    tA(q)=tb(q-n1);
    q=q+1;
 end
 
Psi1up=zeros(length(Ma),n1+n2);Psi2up=zeros(length(Ma),n1+n2); 
Psi1dn=zeros(length(Ma),n1+n2);Psi2dn=zeros(length(Ma),n1+n2);
Psi3up=zeros(length(Ma),n1+n2);Psi4up=zeros(length(Ma),n1+n2); 
Psi3dn=zeros(length(Ma),n1+n2);Psi4dn=zeros(length(Ma),n1+n2);
Psi5up=zeros(length(Ma),n1+n2);Psi6up=zeros(length(Ma),n1+n2); 
Psi5dn=zeros(length(Ma),n1+n2);Psi6dn=zeros(length(Ma),n1+n2);



Psi12up=zeros(length(Ma),n1+n2);Psi12dn=zeros(length(Ma),n1+n2);
Psi1updn=zeros(length(Ma),n1+n2);Psi2updn=zeros(length(Ma),n1+n2);        
Psi12updn=zeros(length(Ma),n1+n2);Psi12dnup=zeros(length(Ma),n1+n2);        
Psi1upanddn=zeros(length(Ma),n1+n2);Psi2upanddn=zeros(length(Ma),n1+n2);

        N1up=zeros(length(Ma),n1+n2);
        N1dn=zeros(length(Ma),n1+n2);
        
        N2up=zeros(length(Ma),n1+n2);
        N2dn=zeros(length(Ma),n1+n2);
        
        N1squareup=zeros(length(Ma),n1+n2);
        N2squareup=zeros(length(Ma),n1+n2);
       
        N1squaredn=zeros(length(Ma),n1+n2);
        N2squaredn=zeros(length(Ma),n1+n2);

error=1.0e-5;%the error for self-consistency

for k=1:length(ta) %first part1
    
%setting hopping parameter
t=ta(k);

for j=1:length(Ma)
		tic; mu=Ma(j);
%initial Dmin��Dmin is the minimum of eigenvalue
        DHexmin=1.0e5;
		for lp=1:length(Pr)
%             for lpp=1:length(theta)
%     
%             psi1up=Pa(lp,lpp);
%             psi1dn=Pa(lp,lpp);
%             psi2up=Pa(lp,lpp);
%             psi2dn=Pa(lp,lpp);
            
            psi1up=Pr(lp);
            psi1dn=Pr(lp);
            psi2up=Pr(lp);
            psi2dn=Pr(lp);
            
            psi3up=Pr(lp);
            psi3dn=Pr(lp);
            psi4up=Pr(lp);
            psi4dn=Pr(lp);
            
            psi5up=Pr(lp);
            psi5dn=Pr(lp);
            psi6up=Pr(lp);
            psi6dn=Pr(lp);
            
%import the 6 single-site meanfield Hamiltonians for a Honeycomb lattice with two species of Pseudospins

HHexa=t*psi4up'*F1upaterm+t*psi5up'*F2upaterm+t*psi6up'*F3upaterm+t*psi1up'*F4upaterm+t*psi2up'*F5upaterm+t*psi3up'*F6upaterm...
    +t*psi4up*F1upadgterm+t*psi5up*F2upadgterm+t*psi6up*F3upadgterm+t*psi1up*F4upadgterm+t*psi2up*F5upadgterm+t*psi3up*F6upadgterm...
    +t*psi4dn'*F1dnaterm+t*psi5dn'*F2dnaterm+t*psi6dn'*F3dnaterm+t*psi1dn'*F4dnaterm+t*psi2dn'*F5dnaterm+t*psi3dn'*F6dnaterm...
    +t*psi4dn*F1dnadgterm+t*psi5dn*F2dnadgterm+t*psi6dn*F3dnadgterm+t*psi1dn*F4dnadgterm+t*psi2dn*F5dnadgterm+t*psi3dn*F6dnadgterm...
    +t*Tterm+Uterm+Vterm+mu*muterm+DigH*t*(-real(t1up*psi1up'*psi4up)-real(conj(t3up)*psi2up'*psi5up)-real(t2up*psi3up'*psi6up)-real(conj(t1up)*psi4up'*psi1up)-real(t3up*psi5up'*psi2up)-real(conj(t2up)*psi6up'*psi3up)-real(t1dn*psi1dn'*psi4dn)-real(conj(t3dn)*psi2dn'*psi5dn)-real(t2dn*psi3dn'*psi6dn)-real(conj(t1dn)*psi4dn'*psi1dn)-real(t3dn*psi5dn'*psi2dn)-real(conj(t2dn)*psi6dn'*psi3dn));

%     HS=sparse(HAB);
%     [Vec,D]=eigs(HS,1,'SR');    % to solve the smallest algebraic eigenvalues
%     
%     VAB0(:,1)=Vec/sqrt(Vec'*Vec);
%     DAB0=D;


%solve the Hamilton with Eigen Vectors and Eigenvalues
            [VecHex,DHex]=eig(HHexa);
%denote the lowest eigenvalue for each site            
            DHex0=DHex(1,1);
            %normalization
            VHex0(:,1)=VecHex(:,1)/sqrt((VecHex(:,1))'*VecHex(:,1));


%find phi1up(down)---the trial solution corresponding to the lowest eigenvalues of Hsite
            if DHex0<DHexmin
            DHexmin=DHex0;
            VHexmin=VHex0;
            
            phi1up=psi1up;
            phi1dn=psi1dn;
            phi2up=psi2up;
            phi2dn=psi2dn;
            
            phi3up=psi3up;
            phi3dn=psi3dn;
            phi4up=psi4up;
            phi4dn=psi4dn;
            
            phi5up=psi5up;
            phi5dn=psi5dn;
            phi6up=psi6up;
            phi6dn=psi6dn;
            end
     
%             end
        end
        
    PHI1up=0; PHI1dn=0;
    PHI2up=0; PHI2dn=0;
    PHI3up=0; PHI3dn=0;
    PHI4up=0; PHI4dn=0;
    PHI5up=0; PHI5dn=0;
    PHI6up=0; PHI6dn=0;
% Values of Order parameters corresponding to the trial solution of ground state above  
	PHI1up=VHexmin'*b1up*VHexmin;PHI1dn=VHexmin'*b1dn*VHexmin;
    PHI2up=VHexmin'*b2up*VHexmin;PHI2dn=VHexmin'*b2dn*VHexmin;
    
    PHI3up=VHexmin'*b3up*VHexmin;PHI3dn=VHexmin'*b3dn*VHexmin;
    PHI4up=VHexmin'*b4up*VHexmin;PHI4dn=VHexmin'*b4dn*VHexmin;
    
    PHI5up=VHexmin'*b5up*VHexmin;PHI5dn=VHexmin'*b5dn*VHexmin;
    PHI6up=VHexmin'*b6up*VHexmin;PHI6dn=VHexmin'*b6dn*VHexmin;

  %value difference for designated order parameters with the trial solutions
        dif1up=abs(phi1up-PHI1up);dif1dn=abs(phi1dn-PHI1dn);
        dif2up=abs(phi2up-PHI2up);dif2dn=abs(phi2dn-PHI2dn);
        dif3up=abs(phi3up-PHI3up);dif3dn=abs(phi3dn-PHI3dn);
        dif4up=abs(phi4up-PHI4up);dif4dn=abs(phi4dn-PHI4dn);
        dif5up=abs(phi5up-PHI5up);dif5dn=abs(phi5dn-PHI5dn);
        dif6up=abs(phi6up-PHI6up);dif6dn=abs(phi6dn-PHI6dn);
        
        step=0;%if step>100,the self-consistency fails.
        while (dif1up>error)||(dif1dn>error)||(dif2up>error)||(dif2dn>error)||(dif3up>error)||(dif3dn>error)||(dif4up>error)||(dif4dn>error)||(dif5up>error)||(dif5dn>error)||(dif6up>error)||(dif6dn>error)
           if step<16
               
               psi1up=PHI1up;psi1dn=PHI1dn;
               psi2up=PHI2up;psi2dn=PHI2dn;
               psi3up=PHI3up;psi3dn=PHI3dn;
               psi4up=PHI4up;psi4dn=PHI4dn;
               psi5up=PHI5up;psi5dn=PHI5dn;
               psi6up=PHI6up;psi6dn=PHI6dn;
               
HHexa=t*psi4up'*F1upaterm+t*psi5up'*F2upaterm+t*psi6up'*F3upaterm+t*psi1up'*F4upaterm+t*psi2up'*F5upaterm+t*psi3up'*F6upaterm...
    +t*psi4up*F1upadgterm+t*psi5up*F2upadgterm+t*psi6up*F3upadgterm+t*psi1up*F4upadgterm+t*psi2up*F5upadgterm+t*psi3up*F6upadgterm...
    +t*psi4dn'*F1dnaterm+t*psi5dn'*F2dnaterm+t*psi6dn'*F3dnaterm+t*psi1dn'*F4dnaterm+t*psi2dn'*F5dnaterm+t*psi3dn'*F6dnaterm...
    +t*psi4dn*F1dnadgterm+t*psi5dn*F2dnadgterm+t*psi6dn*F3dnadgterm+t*psi1dn*F4dnadgterm+t*psi2dn*F5dnadgterm+t*psi3dn*F6dnadgterm...
    +t*Tterm+Uterm+Vterm+mu*muterm+DigH*t*(-real(t1up*psi1up'*psi4up)-real(conj(t3up)*psi2up'*psi5up)-real(t2up*psi3up'*psi6up)-real(conj(t1up)*psi4up'*psi1up)-real(t3up*psi5up'*psi2up)-real(conj(t2up)*psi6up'*psi3up)-real(t1dn*psi1dn'*psi4dn)-real(conj(t3dn)*psi2dn'*psi5dn)-real(t2dn*psi3dn'*psi6dn)-real(conj(t1dn)*psi4dn'*psi1dn)-real(t3dn*psi5dn'*psi2dn)-real(conj(t2dn)*psi6dn'*psi3dn));


%     HS=sparse(HAB);
%     [Vec,D]=eigs(HS,1,'SR');    % to solve the smallest algebraic eigenvalues
%     VAB0(:,1)=Vec/sqrt(Vec'*Vec);

%solve the Hamilton with Eigen Vectors and Eigenvalues
            [VecHex,DHex]=eig(HHexa);
            %normalization
            VHex0(:,1)=VecHex(:,1)/sqrt((VecHex(:,1))'*VecHex(:,1));
            
% Designated ground states of the Cluster           
VHexmin=VHex0;

    PHI1up=0; PHI1dn=0;
    PHI2up=0; PHI2dn=0;
    PHI3up=0; PHI3dn=0;
    PHI4up=0; PHI4dn=0;
    PHI5up=0; PHI5dn=0;
    PHI6up=0; PHI6dn=0;
 
	PHI1up=VHexmin'*b1up*VHexmin;PHI1dn=VHexmin'*b1dn*VHexmin;
    PHI2up=VHexmin'*b2up*VHexmin;PHI2dn=VHexmin'*b2dn*VHexmin;
    
    PHI3up=VHexmin'*b3up*VHexmin;PHI3dn=VHexmin'*b3dn*VHexmin;
    PHI4up=VHexmin'*b4up*VHexmin;PHI4dn=VHexmin'*b4dn*VHexmin;
    
    PHI5up=VHexmin'*b5up*VHexmin;PHI5dn=VHexmin'*b5dn*VHexmin;
    PHI6up=VHexmin'*b6up*VHexmin;PHI6dn=VHexmin'*b6dn*VHexmin;


% Input for the next iteration loop
            phi1up=psi1up;
            phi1dn=psi1dn;
            phi2up=psi2up;
            phi2dn=psi2dn;
            
            phi3up=psi3up;
            phi3dn=psi3dn;
            phi4up=psi4up;
            phi4dn=psi4dn;
            
            phi5up=psi5up;
            phi5dn=psi5dn;
            phi6up=psi6up;
            phi6dn=psi6dn;
                        
        dif1up=abs(phi1up-PHI1up);dif1dn=abs(phi1dn-PHI1dn);
        dif2up=abs(phi2up-PHI2up);dif2dn=abs(phi2dn-PHI2dn);
        dif3up=abs(phi3up-PHI3up);dif3dn=abs(phi3dn-PHI3dn);
        dif4up=abs(phi4up-PHI4up);dif4dn=abs(phi4dn-PHI4dn);
        dif5up=abs(phi5up-PHI5up);dif5dn=abs(phi5dn-PHI5dn);
        dif6up=abs(phi6up-PHI6up);dif6dn=abs(phi6dn-PHI6dn);
        
            step=step+1;
           else
                              disp('not converge');
                              PHI2up=NaN;
               break
           end
        end %after the self-cosistency judgement��get the the  optimal value
        
        
        PHI12up=VHexmin'*b1up'*b2up*VHexmin; PHI12dn=VHexmin'*b1dn'*b2dn*VHexmin;
        
        PHI1updn=VHexmin'*b1up'*b1dn*VHexmin;       PHI2updn=VHexmin'*b2up'*b2dn*VHexmin;
        
        PHI12updn=VHexmin'*b1up'*b2dn*VHexmin;     PHI12dnup=VHexmin'*b1dn'*b2up*VHexmin;
        
        PHI1upanddn=VHexmin'*(b1up+b1dn)*VHexmin;  PHI2upanddn=VHexmin'*(b2up+b2dn)*VHexmin;
        
        n1up=VHexmin'*(b1up'*b1up)*VHexmin;
        n1dn=VHexmin'*(b1dn'*b1dn)*VHexmin;
        
        n2up=VHexmin'*(b2up'*b2up)*VHexmin;
        n2dn=VHexmin'*(b2dn'*b2dn)*VHexmin;
        
       n1squareup=VHexmin'*(b1up'*b1up)*(b1up'*b1up)*VHexmin;
       n1squaredn=VHexmin'*(b1dn'*b1dn)*(b1dn'*b1dn)*VHexmin;
       
       n2squareup=VHexmin'*(b2up'*b2up)*(b2up'*b2up)*VHexmin;
       n2squaredn=VHexmin'*(b2dn'*b2dn)*(b2dn'*b2dn)*VHexmin;
        
        
         
 %save the final optimal value of both order parameters��also save the
 %corresponding state��eigenvector��
		EVHexmin=[EVHexmin,VHexmin]; 
        Psi1up(j,k)=PHI1up; Psi1dn(j,k)=PHI1dn;
        Psi2up(j,k)=PHI2up; Psi2dn(j,k)=PHI2dn;
        
        Psi12up(j,k)=PHI12up;Psi12dn(j,k)=PHI12dn;
        
        Psi1updn(j,k)=PHI1updn; Psi2updn(j,k)=PHI2updn;
        
        Psi12updn(j,k)=PHI12updn;  Psi12dnup(j,k)=PHI12dnup;
        
        Psi1upanddn(j,k)=PHI1upanddn; Psi2upanddn(j,k)=PHI2upanddn;
        
        N1up(j,k)=n1up;
        N1dn(j,k)=n1dn;
        
        N2up(j,k)=n2up;
        N2dn(j,k)=n2dn;
        
       N1squareup(j,k)=n1squareup;
       N2squareup(j,k)=n2squareup;
       
       N1squaredn(j,k)=n1squaredn;
       N2squaredn(j,k)=n2squaredn;
        
		
        disp('First loop');disp([k j]);toc;
        toc1(j)=toc/60;
        T1=T1+toc1(j)
	end
	
end

La=length(ta);Lb=length(tb);
for k=La+1:La+Lb %then part2

    
t=tb(k-La);

for j=1:length(Ma)
		tic; mu=Ma(j);
%initial Dmin��Dmin is the minimum of eigenvalue
        DHexmin=1.0e5;
		for lp=1:length(Pr)
%             for lpp=1:length(theta)
%     
%             psi1up=Pa(lp,lpp);
%             psi1dn=Pa(lp,lpp);
%             psi2up=Pa(lp,lpp);
%             psi2dn=Pa(lp,lpp);
            
            psi1up=Pr(lp);
            psi1dn=Pr(lp);
            psi2up=Pr(lp);
            psi2dn=Pr(lp);
            
            psi3up=Pr(lp);
            psi3dn=Pr(lp);
            psi4up=Pr(lp);
            psi4dn=Pr(lp);
            
            psi5up=Pr(lp);
            psi5dn=Pr(lp);
            psi6up=Pr(lp);
            psi6dn=Pr(lp);
            

HHexa=t*psi4up'*F1upaterm+t*psi5up'*F2upaterm+t*psi6up'*F3upaterm+t*psi1up'*F4upaterm+t*psi2up'*F5upaterm+t*psi3up'*F6upaterm...
    +t*psi4up*F1upadgterm+t*psi5up*F2upadgterm+t*psi6up*F3upadgterm+t*psi1up*F4upadgterm+t*psi2up*F5upadgterm+t*psi3up*F6upadgterm...
    +t*psi4dn'*F1dnaterm+t*psi5dn'*F2dnaterm+t*psi6dn'*F3dnaterm+t*psi1dn'*F4dnaterm+t*psi2dn'*F5dnaterm+t*psi3dn'*F6dnaterm...
    +t*psi4dn*F1dnadgterm+t*psi5dn*F2dnadgterm+t*psi6dn*F3dnadgterm+t*psi1dn*F4dnadgterm+t*psi2dn*F5dnadgterm+t*psi3dn*F6dnadgterm...
    +t*Tterm+Uterm+Vterm+mu*muterm+DigH*t*(-real(t1up*psi1up'*psi4up)-real(conj(t3up)*psi2up'*psi5up)-real(t2up*psi3up'*psi6up)-real(conj(t1up)*psi4up'*psi1up)-real(t3up*psi5up'*psi2up)-real(conj(t2up)*psi6up'*psi3up)-real(t1dn*psi1dn'*psi4dn)-real(conj(t3dn)*psi2dn'*psi5dn)-real(t2dn*psi3dn'*psi6dn)-real(conj(t1dn)*psi4dn'*psi1dn)-real(t3dn*psi5dn'*psi2dn)-real(conj(t2dn)*psi6dn'*psi3dn));


%     HS=sparse(HAB);
%     [Vec,D]=eigs(HS,1,'SR');    % to solve the smallest algebraic eigenvalues
%     
%     VAB0(:,1)=Vec/sqrt(Vec'*Vec);
%     DAB0=D;


%solve the Hamilton with Eigen Vectors and Eigenvalues
            [VecHex,DHex]=eig(HHexa);
%denote the lowest eigenvalue for each site            
            DHex0=DHex(1,1);
            %normalization
            VHex0(:,1)=VecHex(:,1)/sqrt((VecHex(:,1))'*VecHex(:,1));


%find phi1up(down)---the trial solution corresponding to the lowest eigenvalues of Hsite
            if DHex0<DHexmin
            DHexmin=DHex0;
            VHexmin=VHex0;
            
            phi1up=psi1up;
            phi1dn=psi1dn;
            phi2up=psi2up;
            phi2dn=psi2dn;
            
            phi3up=psi3up;
            phi3dn=psi3dn;
            phi4up=psi4up;
            phi4dn=psi4dn;
            
            phi5up=psi5up;
            phi5dn=psi5dn;
            phi6up=psi6up;
            phi6dn=psi6dn;
            end
     
%             end
        end
        
    PHI1up=0; PHI1dn=0;
    PHI2up=0; PHI2dn=0;
    PHI3up=0; PHI3dn=0;
    PHI4up=0; PHI4dn=0;
    PHI5up=0; PHI5dn=0;
    PHI6up=0; PHI6dn=0;
% Values of Order parameters corresponding to the trial solution of ground state above  
	PHI1up=VHexmin'*b1up*VHexmin;PHI1dn=VHexmin'*b1dn*VHexmin;
    PHI2up=VHexmin'*b2up*VHexmin;PHI2dn=VHexmin'*b2dn*VHexmin;
    
    PHI3up=VHexmin'*b3up*VHexmin;PHI3dn=VHexmin'*b3dn*VHexmin;
    PHI4up=VHexmin'*b4up*VHexmin;PHI4dn=VHexmin'*b4dn*VHexmin;
    
    PHI5up=VHexmin'*b5up*VHexmin;PHI5dn=VHexmin'*b5dn*VHexmin;
    PHI6up=VHexmin'*b6up*VHexmin;PHI6dn=VHexmin'*b6dn*VHexmin;

  %value difference for designated order parameters with the trial solutions
        dif1up=abs(phi1up-PHI1up);dif1dn=abs(phi1dn-PHI1dn);
        dif2up=abs(phi2up-PHI2up);dif2dn=abs(phi2dn-PHI2dn);
        dif3up=abs(phi3up-PHI3up);dif3dn=abs(phi3dn-PHI3dn);
        dif4up=abs(phi4up-PHI4up);dif4dn=abs(phi4dn-PHI4dn);
        dif5up=abs(phi5up-PHI5up);dif5dn=abs(phi5dn-PHI5dn);
        dif6up=abs(phi6up-PHI6up);dif6dn=abs(phi6dn-PHI6dn);
        
        step=0;%if step>100,the self-consistency fails.
        while (dif1up>error)||(dif1dn>error)||(dif2up>error)||(dif2dn>error)||(dif3up>error)||(dif3dn>error)||(dif4up>error)||(dif4dn>error)||(dif5up>error)||(dif5dn>error)||(dif6up>error)||(dif6dn>error)
           if step<100
               
               psi1up=PHI1up;psi1dn=PHI1dn;
               psi2up=PHI2up;psi2dn=PHI2dn;
               psi3up=PHI3up;psi3dn=PHI3dn;
               psi4up=PHI4up;psi4dn=PHI4dn;
               psi5up=PHI5up;psi5dn=PHI5dn;
               psi6up=PHI6up;psi6dn=PHI6dn;

HHexa=t*psi4up'*F1upaterm+t*psi5up'*F2upaterm+t*psi6up'*F3upaterm+t*psi1up'*F4upaterm+t*psi2up'*F5upaterm+t*psi3up'*F6upaterm...
    +t*psi4up*F1upadgterm+t*psi5up*F2upadgterm+t*psi6up*F3upadgterm+t*psi1up*F4upadgterm+t*psi2up*F5upadgterm+t*psi3up*F6upadgterm...
    +t*psi4dn'*F1dnaterm+t*psi5dn'*F2dnaterm+t*psi6dn'*F3dnaterm+t*psi1dn'*F4dnaterm+t*psi2dn'*F5dnaterm+t*psi3dn'*F6dnaterm...
    +t*psi4dn*F1dnadgterm+t*psi5dn*F2dnadgterm+t*psi6dn*F3dnadgterm+t*psi1dn*F4dnadgterm+t*psi2dn*F5dnadgterm+t*psi3dn*F6dnadgterm...
    +t*Tterm+Uterm+Vterm+mu*muterm+DigH*t*(-real(t1up*psi1up'*psi4up)-real(conj(t3up)*psi2up'*psi5up)-real(t2up*psi3up'*psi6up)-real(conj(t1up)*psi4up'*psi1up)-real(t3up*psi5up'*psi2up)-real(conj(t2up)*psi6up'*psi3up)-real(t1dn*psi1dn'*psi4dn)-real(conj(t3dn)*psi2dn'*psi5dn)-real(t2dn*psi3dn'*psi6dn)-real(conj(t1dn)*psi4dn'*psi1dn)-real(t3dn*psi5dn'*psi2dn)-real(conj(t2dn)*psi6dn'*psi3dn));


%     HS=sparse(HAB);
%     [Vec,D]=eigs(HS,1,'SR');    % to solve the smallest algebraic eigenvalues
%     VAB0(:,1)=Vec/sqrt(Vec'*Vec);

%solve the Hamilton with Eigen Vectors and Eigenvalues
            [VecHex,DHex]=eig(HHexa);
            %normalization
            VHex0(:,1)=VecHex(:,1)/sqrt((VecHex(:,1))'*VecHex(:,1));
            
% Designated ground states of the Cluster           
VHexmin=VHex0;

    PHI1up=0; PHI1dn=0;
    PHI2up=0; PHI2dn=0;
    PHI3up=0; PHI3dn=0;
    PHI4up=0; PHI4dn=0;
    PHI5up=0; PHI5dn=0;
    PHI6up=0; PHI6dn=0;
 
	PHI1up=VHexmin'*b1up*VHexmin;PHI1dn=VHexmin'*b1dn*VHexmin;
    PHI2up=VHexmin'*b2up*VHexmin;PHI2dn=VHexmin'*b2dn*VHexmin;
    
    PHI3up=VHexmin'*b3up*VHexmin;PHI3dn=VHexmin'*b3dn*VHexmin;
    PHI4up=VHexmin'*b4up*VHexmin;PHI4dn=VHexmin'*b4dn*VHexmin;
    
    PHI5up=VHexmin'*b5up*VHexmin;PHI5dn=VHexmin'*b5dn*VHexmin;
    PHI6up=VHexmin'*b6up*VHexmin;PHI6dn=VHexmin'*b6dn*VHexmin;


% Input for the next iteration loop
            phi1up=psi1up;
            phi1dn=psi1dn;
            phi2up=psi2up;
            phi2dn=psi2dn;
            
            phi3up=psi3up;
            phi3dn=psi3dn;
            phi4up=psi4up;
            phi4dn=psi4dn;
            
            phi5up=psi5up;
            phi5dn=psi5dn;
            phi6up=psi6up;
            phi6dn=psi6dn;
                        
        dif1up=abs(phi1up-PHI1up);dif1dn=abs(phi1dn-PHI1dn);
        dif2up=abs(phi2up-PHI2up);dif2dn=abs(phi2dn-PHI2dn);
        dif3up=abs(phi3up-PHI3up);dif3dn=abs(phi3dn-PHI3dn);
        dif4up=abs(phi4up-PHI4up);dif4dn=abs(phi4dn-PHI4dn);
        dif5up=abs(phi5up-PHI5up);dif5dn=abs(phi5dn-PHI5dn);
        dif6up=abs(phi6up-PHI6up);dif6dn=abs(phi6dn-PHI6dn);
        
            step=step+1;
           else
                              disp('not converge');
                              PHI2up=NaN;
               break
           end
        end %after the self-cosistency judgement��get the the  optimal value
        
        
        PHI12up=VHexmin'*b1up'*b2up*VHexmin; PHI12dn=VHexmin'*b1dn'*b2dn*VHexmin;
        
        PHI1updn=VHexmin'*b1up'*b1dn*VHexmin;       PHI2updn=VHexmin'*b2up'*b2dn*VHexmin;
        
        PHI12updn=VHexmin'*b1up'*b2dn*VHexmin;     PHI12dnup=VHexmin'*b1dn'*b2up*VHexmin;
        
        PHI1upanddn=VHexmin'*(b1up+b1dn)*VHexmin;  PHI2upanddn=VHexmin'*(b2up+b2dn)*VHexmin;
        
        n1up=VHexmin'*(b1up'*b1up)*VHexmin;
        n1dn=VHexmin'*(b1dn'*b1dn)*VHexmin;
        
        n2up=VHexmin'*(b2up'*b2up)*VHexmin;
        n2dn=VHexmin'*(b2dn'*b2dn)*VHexmin;
        
       n1squareup=VHexmin'*(b1up'*b1up)*(b1up'*b1up)*VHexmin;
       n1squaredn=VHexmin'*(b1dn'*b1dn)*(b1dn'*b1dn)*VHexmin;
       
       n2squareup=VHexmin'*(b2up'*b2up)*(b2up'*b2up)*VHexmin;
       n2squaredn=VHexmin'*(b2dn'*b2dn)*(b2dn'*b2dn)*VHexmin;
        
        
         
 %save the final optimal value of both order parameters��also save the
 %corresponding state��eigenvector��
		EVHexmin=[EVHexmin,VHexmin]; 
        Psi1up(j,k)=PHI1up; Psi1dn(j,k)=PHI1dn;
        Psi2up(j,k)=PHI2up; Psi2dn(j,k)=PHI2dn;
        
        Psi12up(j,k)=PHI12up;Psi12dn(j,k)=PHI12dn;
        
        Psi1updn(j,k)=PHI1updn; Psi2updn(j,k)=PHI2updn;
        
        Psi12updn(j,k)=PHI12updn;  Psi12dnup(j,k)=PHI12dnup;
        
        Psi1upanddn(j,k)=PHI1upanddn; Psi2upanddn(j,k)=PHI2upanddn;
        
        N1up(j,k)=n1up;
        N1dn(j,k)=n1dn;
        
        N2up(j,k)=n2up;
        N2dn(j,k)=n2dn;
        
       N1squareup(j,k)=n1squareup;
       N2squareup(j,k)=n2squareup;
       
       N1squaredn(j,k)=n1squaredn;
       N2squaredn(j,k)=n2squaredn;
        
        disp('Second loop');disp([k j]);toc;
        toc2(j)=toc/60;
        T2=T2+toc2(j)
	end
	
end

[tA,MA]=meshgrid(tA,Ma);

Ttot=T1+T2
% display phase diagram for orderparameter
figure; surf(tA,MA,abs(Psi1up),'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi1dn),'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi1up+Psi1dn)/2,'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi2up),'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi2dn),'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi1upanddn),'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi1updn),'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi12up),'edgecolor','none');shading interp;view(0,90);
figure; surf(tA,MA,abs(Psi12dn),'edgecolor','none');shading interp;view(0,90);