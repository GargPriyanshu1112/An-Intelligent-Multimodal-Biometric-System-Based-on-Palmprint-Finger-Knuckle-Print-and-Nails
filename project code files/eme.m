function E=eme(X, patch_size)
    [m n] = size(X);

    r = m / patch_size;
    c = n / patch_size;

    E=0.;
    B1=zeros(patch_size);
    m1=1;
    for m=1: r
        n1=1;
        for n=1: c
            B1=X(m1: m1+patch_size-1, n1: n1+patch_size-1);
            b_min=min(min(B1));
            b_max=max(max(B1));
            if b_min>0
                b_ratio=b_max/b_min;
                E=E+20.*log(double(b_ratio));
            end;
            n1=n1+patch_size;
        end;
        m1=m1+patch_size;
    end;
    E=(E/r)/c;
    
    
    
    
% function E=eme(X,M,L);
% %	L=5; 
% 	how_many=floor(M/L);
% 	
% 	E=0.; 
% 	B1=zeros(L);
% 	m1=1;
% 	for m=1:how_many
% 	    n1=1;
% 	    for n=1:how_many
% 	    	B1=X(m1:m1+L-1,n1:n1+L-1);
%             b_min=min(min(B1));
%             b_max=max(max(B1));
%             if b_min>0 
%                 b_ratio=b_max/b_min;
%                 E=E+20.*log(double(b_ratio));	  
%             end;
%             n1=n1+L;	              
% 	    end;
% 	    m1=m1+L;
% 	end;
% 	E=(E/how_many)/how_many;